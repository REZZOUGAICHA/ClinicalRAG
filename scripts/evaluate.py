"""
RAGAS evaluation harness — measures retrieval + generation quality with real
numbers instead of anecdotal "it seemed to work" testing.

WHY THIS EXISTS:
Up to this point, every fix in this project (query rewriting for typos,
parent-document expansion for cross-section questions) was verified by
manually reading a handful of example responses. That's fine for catching
bugs, but it doesn't scale and it isn't a number you can put in a CV or
PhD application. This script runs a fixed set of questions through the
*actual* pipeline (src.pipeline.ask — same code the API uses) and scores
each answer on four RAGAS metrics:

  - Faithfulness      — does every claim in the answer actually appear in
                         the retrieved passages? (catches hallucination)
  - Context Precision  — of the retrieved passages, how many were actually
                         relevant to answering the question?
  - Context Recall     — did retrieval find everything needed to answer,
                         compared against the reference answer?
  - Answer Relevancy   — does the answer actually address the question
                         asked (not just cite something true but off-topic)?

EVAL SET DESIGN:
Includes both plain single-section lookups AND the two failure modes this
project specifically found and fixed:
  - a cross-section question (patient identity lives in one chunk, the
    fact asked about lives in a different chunk of the same document —
    what parent-document expansion fixes)
  - a deliberately misspelled question, paired with its correctly-spelled
    twin — what query rewriting fixes. Comparing their scores directly
    shows whether the typo still degrades retrieval quality.

Usage: python scripts/evaluate.py
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import instructor
from dotenv import load_dotenv
from google import genai

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src.pipeline import ask

load_dotenv()

# The eval JUDGE runs on Gemini, not Groq — a deliberate separation from the
# app's own generation (which stays on Groq, see generator.py). Groq's free
# tier has a 100k-tokens/day cap that a full eval run alone can exhaust, so
# scoring needs its own independent quota.
# gemini-3.6-flash's free tier caps at just 20 requests/day — nowhere near
# enough for a 32-call eval run (8 questions x 4 metrics). gemini-3.1-flash-lite
# has a much more workable free-tier allowance for this account.
JUDGE_MODEL = "gemini-3.1-flash-lite"
RESULTS_DIR = Path("data/eval")

# Ground-truth answers pulled directly from the source PDFs (see
# data/processed/chunks.json) — not guessed, so ContextRecall/Precision
# scores reflect real retrieval quality, not a mismatched reference.
EVAL_SET = [
    {
        "id": "wbc-lymphoma",
        "question": "What was the WBC count in the lymphoma patient?",
        "reference": "The WBC count was 11.2 x 10^9/L, which is high (reference range 4.0-11.0).",
    },
    {
        "id": "hba1c-diabetic",
        "question": "What is the HbA1c of the diabetic patient?",
        "reference": "The HbA1c is 9.2%, above the target of <7.0%.",
    },
    {
        "id": "meds-acute-mi-cross-section",
        "question": "What medications was the acute MI patient discharged with?",
        "reference": (
            "Aspirin 100 mg OD, Ticagrelor 90 mg BID, Atorvastatin 80 mg OD, "
            "Ramipril 5 mg OD, Bisoprolol 5 mg OD, and Pantoprazole 40 mg OD."
        ),
    },
    {
        "id": "stroke-door-to-needle",
        "question": "What was the door-to-needle time for the stroke patient's thrombolysis?",
        "reference": "27 minutes (tPA administered at 10:22).",
    },
    {
        "id": "stroke-anticoagulant-cross-section",
        "question": "What anticoagulant is the stroke patient on and why?",
        "reference": (
            "Apixaban 5 mg BID, for anticoagulation due to newly diagnosed "
            "atrial fibrillation (the cause of the stroke)."
        ),
    },
    {
        "id": "sepsis-lactate",
        "question": "What was the lactate level in the sepsis patient?",
        "reference": "4.8 mmol/L, critically high (reference <2.0), indicating tissue hypoperfusion.",
    },
    {
        "id": "lymphoma-id-cross-section",
        "question": "What is the patient ID of the lymphoma patient?",
        "reference": "SYN-2024-001.",
    },
    {
        "id": "lymphoma-id-typo",
        "question": "whats the id of lymphoma patien",  # deliberate typo — pairs with the item above
        "reference": "SYN-2024-001.",
    },
]


def _build_scorers():
    # ragas's llm_factory(provider="google", ...) doesn't work for our case:
    # it always constructs a *sync* instructor client for Google, but every
    # Collections metric here scores via .ascore() and requires an async
    # client — a real gap in ragas 0.4.3's Google provider dispatch (it just
    # doesn't forward use_async=True through to instructor.from_genai()).
    # So we build the async instructor-patched client by hand instead.
    raw_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    patched_client = instructor.from_genai(raw_client, use_async=True)
    rl = InstructorLLM(client=patched_client, model=JUDGE_MODEL, provider="google")
    emb = HuggingFaceEmbeddings(model="BAAI/bge-small-en")

    return {
        "faithfulness": Faithfulness(llm=rl),
        "context_precision": ContextPrecisionWithReference(llm=rl),
        "context_recall": ContextRecall(llm=rl),
        # strictness=1 (default 3) — AnswerRelevancy generates that many
        # candidate questions internally per score, each its own LLM call.
        # At 5 requests/minute, 3 would burst past the limit within a
        # single metric call before our own pacing even gets a chance to help.
        "answer_relevancy": AnswerRelevancy(llm=rl, embeddings=emb, strictness=1),
    }


_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s")


async def _score_with_retry(coro_fn, max_attempts: int = 6, base_delay: float = 15.0):
    """
    Free-tier LLM APIs cap tokens per minute, and each RAGAS metric call
    carries the full retrieved context in its prompt — with parent-document
    expansion that can be several chunks' worth of text. Running metrics
    concurrently (asyncio.gather) blew straight through Groq's per-minute
    cap earlier in this project; running sequentially with backoff keeps
    every eval run comfortably under whatever judge model's budget instead
    of racing several large prompts at once.

    Gemini's 429 responses include an exact suggested wait
    ("Please retry in 19.06s") — a fixed backoff schedule guessed wrong
    often enough to exhaust all retries on an error that would've cleared
    with the right wait. Parse and use Google's own number when present;
    fall back to a growing fixed schedule otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except Exception as e:
            msg = str(e)
            # 429 = rate limit, 503/UNAVAILABLE = transient "model overloaded"
            # (Google's own error message calls these "usually temporary").
            # Anything else (bad request, auth, etc.) is a real failure —
            # retrying it would just waste the rate-limit budget.
            is_transient = any(s in msg for s in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if attempt == max_attempts or not is_transient:
                raise

            match = _RETRY_DELAY_RE.search(msg)
            wait = float(match.group(1)) + 3 if match else base_delay * attempt
            print(f"    transient error, retrying in {wait:.0f}s (attempt {attempt}/{max_attempts})...")
            await asyncio.sleep(wait)


async def _score_one(scorers: dict, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    # Gemini's free tier for gemini-3.6-flash caps at 5 requests/minute —
    # 12s/request minimum sustainable rate. 15s between calls gives margin.
    # (Groq's per-minute cap was much looser; this pacing is Gemini-specific.)
    PACING_SECONDS = 15

    # Sequential, not gather() — see _score_with_retry for why.
    faithfulness = await _score_with_retry(
        lambda: scorers["faithfulness"].ascore(user_input=question, response=answer, retrieved_contexts=contexts)
    )
    await asyncio.sleep(PACING_SECONDS)
    ctx_precision = await _score_with_retry(
        lambda: scorers["context_precision"].ascore(user_input=question, reference=reference, retrieved_contexts=contexts)
    )
    await asyncio.sleep(PACING_SECONDS)
    ctx_recall = await _score_with_retry(
        lambda: scorers["context_recall"].ascore(user_input=question, retrieved_contexts=contexts, reference=reference)
    )
    await asyncio.sleep(PACING_SECONDS)
    ans_relevancy = await _score_with_retry(
        lambda: scorers["answer_relevancy"].ascore(user_input=question, response=answer)
    )

    return {
        "faithfulness": faithfulness.value,
        "context_precision": ctx_precision.value,
        "context_recall": ctx_recall.value,
        "answer_relevancy": ans_relevancy.value,
    }


CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"


def _save_checkpoint(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_eval():
    # Resumable: a question already scored in an earlier (possibly crashed)
    # run is loaded from the checkpoint and skipped, not re-scored. Free-tier
    # judge quota has proven scarce enough this session that re-doing
    # already-successful work would be wasteful.
    results = []
    done_ids = set()
    if CHECKPOINT_PATH.exists():
        results = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in results}
        if done_ids:
            print(f"Resuming from checkpoint — already have: {sorted(done_ids)}")

    scorers = _build_scorers()

    for item in EVAL_SET:
        if item["id"] in done_ids:
            continue

        print(f"Running: {item['id']} ...")
        pipeline_result = ask(item["question"], top_k=3)
        answer = pipeline_result["answer"]
        contexts = [c["text"] for c in pipeline_result["chunks"]]

        scores = await _score_one(scorers, item["question"], answer, contexts, item["reference"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "reference": item["reference"],
            "chunks_retrieved": len(contexts),
            "scores": scores,
        })
        print(f"  faithfulness={scores['faithfulness']:.2f}  "
              f"context_precision={scores['context_precision']:.2f}  "
              f"context_recall={scores['context_recall']:.2f}  "
              f"answer_relevancy={scores['answer_relevancy']:.2f}")
        _save_checkpoint(results)  # persisted immediately — a later crash won't lose this
        await asyncio.sleep(15)  # same rate-limit pacing between questions

    aggregate = {
        metric: round(sum(r["scores"][metric] for r in results) / len(results), 4)
        for metric in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
    }

    print("\n" + "=" * 60)
    print("AGGREGATE (mean across all questions)")
    for metric, value in aggregate.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(
        json.dumps({"aggregate": aggregate, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    CHECKPOINT_PATH.unlink(missing_ok=True)  # eval completed in full — checkpoint no longer needed
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())

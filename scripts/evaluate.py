"""
Answer-level eval harness: runs questions from eval/questions.json through
the real answer path (src.pipeline.ask, or src.agent.ask_agent with
--system agent) and scores what comes back.

Always computed, no judge needed:
  key_fact_recall    fraction of the item's key_facts found in the answer
                     (normalized text match, see scripts/eval_common.py)
  refused            the answer is the "not found in the documents" reply
  refused_correctly  for unanswerable items: did it refuse?
  false_refusal      for answerable items: refused and got no key fact

With --ragas, also the four RAGAS metrics judged by Gemini:
  - Faithfulness      — does every claim in the answer actually appear in
                         the retrieved passages? (catches hallucination)
  - Context Precision  — of the retrieved passages, how many were actually
                         relevant to answering the question?
  - Context Recall     — did retrieval find everything needed to answer,
                         compared against the reference answer?
  - Answer Relevancy   — does the answer actually address the question
                         asked (not just cite something true but off-topic)?
RAGAS is skipped for unanswerable items: recall/precision against a
"not in the documents" reference doesn't measure anything useful.

Runs are resumable: each finished question goes into
data/eval/checkpoint_<config>.json, and a rerun with the same --config
skips it. The checkpoint is kept after a run finishes, so adding --ragas to
a config that already has answers scores the stored answers instead of
regenerating them. Delete the checkpoint (or pick a new --config) to get
fresh answers. A checkpoint written by another --system or with different
retrieval settings (src/config.py) is refused rather than mixed in.

The deterministic scores are recomputed from the stored answers every run,
so scorer fixes apply to old answers too (scripts/rescore.py does the same
for finished result files).

Usage:
  python scripts/evaluate.py --config baseline
  python scripts/evaluate.py --config baseline --types lookup,cross_section --ragas
  python scripts/evaluate.py --config agent-v1 --system agent --types multi_hop
"""

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402
from eval_common import (  # noqa: E402
    RAGAS_METRICS,
    RUNS_DIR,
    aggregate_by_type,
    append_summary,
    call_with_groq_retry,
    deterministic_scores,
    load_questions,
    print_report,
    rescore_row,
    split_csv,
    utc_stamp,
)

from src.config import settings  # noqa: E402

load_dotenv()

# The eval JUDGE runs on Gemini, not Groq — a deliberate separation from the
# app's own generation (which stays on Groq, see generator.py). Groq's free
# tier has a 100k-tokens/day cap that a full eval run alone can exhaust, so
# scoring needs its own independent quota.
# gemini-3.1-flash-lite rather than a bigger Gemini model: the bigger ones'
# free tier allows ~20 requests/day, less than one run needs (4 per question).
JUDGE_MODEL = "gemini-3.1-flash-lite"


def _build_scorers():
    # Imported here so runs without --ragas don't need the eval extras installed.
    import instructor
    from google import genai
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms.base import InstructorLLM
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

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
            # A 503 from Gemini once left a call hanging with no error and the
            # whole run sat idle for an hour. Treat a stall as transient too.
            return await asyncio.wait_for(coro_fn(), timeout=120)
        except Exception as e:
            msg = "503 timeout" if isinstance(e, TimeoutError) else str(e)
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
    # Paced for a free-tier Gemini cap of 5 requests/minute (12s apart at
    # minimum); 15s leaves margin. Groq's cap is looser, so this is
    # Gemini-specific.
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


def _load_answer_fn(system: str):
    if system == "pipeline":
        from src.pipeline import ask
        return ask
    try:
        module = importlib.import_module("src.agent")
    except ModuleNotFoundError as e:
        if e.name not in ("src.agent", "src"):
            raise  # src.agent exists but one of its own imports is missing
        raise SystemExit("--system agent needs the src/agent package (ask_agent).") from None
    if not hasattr(module, "ask_agent"):
        raise SystemExit("src.agent was found but has no ask_agent(question) function.")
    return module.ask_agent


async def run_eval(args) -> None:
    items = load_questions(split_csv(args.types), split_csv(args.ids), args.limit)
    if not items:
        raise SystemExit("No questions selected.")
    answer_fn = _load_answer_fn(args.system)
    retrieval = settings.describe()

    checkpoint_path = RUNS_DIR / f"checkpoint_{args.config}.json"
    by_id = {}
    if checkpoint_path.exists():
        by_id = {r["id"]: r for r in json.loads(checkpoint_path.read_text(encoding="utf-8"))}
        # Answers from another system or retrieval setup would be silently
        # averaged into this run's numbers.
        stale = sorted(i for i, r in by_id.items()
                       if r.get("system") != args.system or r.get("retrieval") != retrieval)
        if stale:
            raise SystemExit(
                f"{checkpoint_path} has {len(stale)} answer(s) from a different --system or "
                f"retrieval settings (or written before settings were recorded), e.g. {stale[0]}.\n"
                f"Use another --config, or delete the checkpoint to start over.")
        resumed = sorted(set(by_id) & {q["id"] for q in items})
        if resumed:
            print(f"Resuming {args.config}: {len(resumed)} question(s) already answered")

    def save_checkpoint():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(list(by_id.values()), indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    scorers = _build_scorers() if args.ragas else None
    last_call = 0.0

    for item in items:
        row = by_id.get(item["id"])
        if row is None:
            # pace from the start of the previous call, not its end, so slow
            # answers don't add dead time on top
            wait = args.pace - (time.monotonic() - last_call)
            if last_call and wait > 0:
                time.sleep(wait)
            print(f"Running: {item['id']} ...")
            last_call = time.monotonic()
            result = call_with_groq_retry(answer_fn, item["question"])
            latency = round(time.monotonic() - last_call, 2)

            answer = result["answer"]
            row = {
                "id": item["id"],
                "type": item["type"],
                "answerable": item["answerable"],
                "question": item["question"],
                "answer": answer,
                "reference": item["reference"],
                "key_facts": item["key_facts"],
                "system": args.system,
                "retrieval": retrieval,
                "model": result.get("model"),
                "latency_s": latency,
                "chunks_retrieved": len(result["chunks"]),
                "context_sources": [{"source_file": c["source_file"], "section": c["section"]}
                                    for c in result["chunks"]],
                "contexts": [c["text"] for c in result["chunks"]],
                "scores": deterministic_scores(item, answer),
            }
            if "gathered_chunks" in result:
                row["chunks_gathered"] = len(result["gathered_chunks"])
            if "trace" in result:
                # Lets scripts/visualize_agent_run.py --from-json redraw any eval run offline.
                row["route"] = result.get("route")
                row["trace"] = result["trace"]
            by_id[item["id"]] = row
            save_checkpoint()  # persisted immediately — a later crash won't lose this

            s = row["scores"]
            if item["answerable"]:
                missing = f"  missing={s['missing_facts']}" if s["missing_facts"] else ""
                print(f"  key_fact_recall={s['key_fact_recall']:.2f}  refused={s['refused']}{missing}")
            else:
                print(f"  refused_correctly={s['refused_correctly']}")

        if scorers and item["answerable"] and not row.get("ragas"):
            print(f"  RAGAS: {item['id']} ...")
            row["ragas"] = await _score_one(scorers, row["question"], row["answer"],
                                            row["contexts"], item["reference"])
            print("  " + "  ".join(f"{m}={row['ragas'][m]:.2f}" for m in RAGAS_METRICS))
            save_checkpoint()
            await asyncio.sleep(15)  # same rate-limit pacing between questions

    # Scored here rather than trusted from the checkpoint, so resumed answers
    # are judged by the current scorer and the current eval/questions.json.
    rows = [rescore_row(by_id[q["id"]], q) for q in items]
    overall, by_type = aggregate_by_type(rows)

    print()
    print_report(overall, by_type)

    run_info = {"system": args.system, "ragas": args.ragas, "retrieval": retrieval,
                "selection": {"types": args.types, "ids": args.ids, "limit": args.limit}}
    out_path = RUNS_DIR / f"eval_{args.config}_{utc_stamp()}.json"
    out_path.write_text(json.dumps({"config": args.config, **run_info, "aggregate": overall,
                                    "by_type": by_type, "results": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    append_summary({"script": "evaluate", "config": args.config, **run_info,
                    "aggregate": overall, "by_type": by_type})
    # The checkpoint stays: a later --ragas run (or a rescore) reuses its answers.
    print(f"\nFull results saved to: {out_path}")
    print(f"Answers kept in: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="run label (default: the --system name)")
    parser.add_argument("--system", choices=("pipeline", "agent"), default="pipeline")
    parser.add_argument("--types", help="comma-separated question types, e.g. lookup,cross_section")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--limit", type=int, help="only the first N selected questions")
    parser.add_argument("--ragas", action="store_true", help="also score with RAGAS (Gemini judge)")
    parser.add_argument("--pace", type=float, default=10.0,
                        help="minimum seconds between answer calls; Groq's free tier caps tokens/minute")
    args = parser.parse_args()
    args.config = args.config or args.system
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()

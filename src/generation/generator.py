"""
LLM generation module — takes retrieved chunks and produces a cited answer.

This is the "G" in RAG. Its job is NOT to know the answer —
the retriever found the answer in the documents. The generator's job
is to READ the retrieved passages and write a fluent, cited response.

WHY THIS DISTINCTION MATTERS:
An uncited LLM answer is an unverifiable claim. The model might be
"hallucinating" — generating plausible-sounding text that isn't in any document.
By instructing the LLM to ONLY use the provided passages and to cite its source,
we get an answer that can be verified by a human in under 10 seconds
(open the PDF, find the section, check the claim).

In clinical AI this is not optional — it's the entire safety story.

THE PROMPT ENGINEERING DECISION:
We use a "context-grounded generation" prompt. The key constraint is:
  "Answer ONLY using the information in the passages below.
   If the answer is not in the passages, say so explicitly."

This instruction does two things:
1. Prevents hallucination by giving the model an "out" (it can say "not found")
2. Forces citation — the model has to reference which passage it used

LLM CHOICE — Groq (Llama 3.3 70B):
Groq is an inference provider that runs Meta's open-source Llama models
on custom hardware (LPUs — Language Processing Units). Key facts:
- Free tier: 6,000 requests/day, no credit card, no regional restrictions
- Model: llama-3.3-70b-versatile — Meta's 70 billion parameter model,
  competitive with GPT-4o on most benchmarks
- Very fast: Groq's LPUs are 10-100x faster than GPU inference
- API is OpenAI-compatible — easy to swap in another provider later

PHD ANGLE:
This approach is directly analogous to the "faithfulness" evaluation metric
in RAGAS — faithfulness measures whether each claim in the answer is
supported by the retrieved context. By constraining the prompt this way,
we're structurally maximising faithfulness before we even evaluate it.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_client() -> Groq:
    """
    Initialise the Groq client using the API key from .env.

    WHY load from environment variable and not hardcode the key?
    Security. If you hardcode the key in Python and push to GitHub,
    the key is public. Git history doesn't forget even if you delete the line.
    Environment variables keep secrets out of the codebase entirely.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Make sure your .env file exists "
            "and contains GROQ_API_KEY=gsk_your-key-here"
        )
    return Groq(api_key=api_key)


def _build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the prompt that gets sent to Groq (Llama 3.3 70B).

    PROMPT STRUCTURE — why each part is here:

    1. Role instruction ("You are a clinical document assistant...")
       Sets the model's behaviour mode. Without this, the LLM might
       draw on general medical knowledge beyond the documents — exactly
       what we don't want.

    2. Numbered passages with source labels
       Each passage is clearly marked with [1], [2], [3] and its origin.
       This gives the model a consistent citation format to reference.

    3. The strict constraint ("ONLY use the passages above")
       This is the anti-hallucination instruction. Models follow explicit
       constraints reliably when they're clear and repeated.

    4. Citation format instruction
       We tell the model exactly how to cite: [Source: filename | SECTION].
       This gives us machine-parseable citations we can extract later.

    5. The question at the end
       Putting the question after the context is intentional. Research on
       LLM prompting shows models attend more to content near the question.
    """
    passages = ""
    for i, chunk in enumerate(chunks, start=1):
        passages += (
            f"[{i}] Source: {chunk['source_file']} | Section: {chunk['section']}\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""You are a clinical document assistant. Your job is to answer questions
about clinical reports accurately and safely.

IMPORTANT RULES:
- Answer ONLY using the information in the numbered passages below.
- Do not use any outside medical knowledge — only what is explicitly stated in the passages.
- After your answer, cite which passage(s) you used in this format: [Source: filename | SECTION]
- If the answer to the question is not found in any passage, respond with:
  "The answer to this question is not found in the available clinical documents."
- Never guess. Never invent lab values, medications, or diagnoses.

PASSAGES:
{passages}
QUESTION: {query}

ANSWER (with citation):"""

    return prompt


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Send the query + retrieved chunks to Gemini and return the answer.

    Returns a dict with:
    - answer:  the full text response from Gemini
    - sources: list of (source_file, section) pairs used (from chunk metadata)
    - model:   which model produced the answer (useful for reproducibility)

    WHY return sources separately from the answer text?
    The answer text contains human-readable citations ("see report_001...").
    The sources list contains structured data we can use programmatically —
    for example, to highlight the source chunk in the UI, or to run
    automated faithfulness checks later.
    """
    client = _get_client()
    prompt = _build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # low temperature = less creativity, more factual
        max_tokens=512,
    )

    answer_text = response.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "sources": [
            {"source_file": c["source_file"], "section": c["section"]}
            for c in chunks
        ],
        "model": GROQ_MODEL,
        "chunks_used": len(chunks),
    }

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

import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# Matches the citation format we instruct the model to use:
# "[Source: filename.pdf | SECTION NAME]". The model tends to echo the
# "Section:" label from how passages are formatted in the prompt (producing
# "[Source: file.pdf | Section: SECTION NAME]") even though it isn't asked
# to — so that label is optional here, not part of the real section name.
_CITATION_RE = re.compile(r"\[Source:\s*([^|\]]+?)\s*\|\s*(?:Section:\s*)?([^\]]+?)\s*\]", re.IGNORECASE)


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


def rewrite_query(query: str) -> str:
    """
    Normalize the raw user query before it's used for retrieval: fix typos,
    tidy up phrasing — without changing what's being asked.

    WHY THIS EXISTS:
    Embedding search and BM25 keyword search both match on exact wording.
    A single typo ("patien" instead of "patient") can shift the embedding
    enough, and breaks BM25's exact-token match entirely, to drop the
    correct document out of the retrieval shortlist before reranking even
    gets a chance to see it. Neither the embedding model nor BM25 do any
    spelling correction on their own.

    This is a small, fast, separate LLM call — not the main generation
    call — so retrieval always searches on cleaned-up text, while the
    final answer is still generated against the user's original message
    (see generate_answer/stream_answer), so the reply still reflects what
    they actually typed.
    """
    client = _get_client()
    prompt = f"""Rewrite the following user query to fix any spelling or typing
errors and tidy up the phrasing. Do NOT change its meaning, do NOT add
information, and do NOT answer it. If it's already clear, return it unchanged.
Reply with ONLY the rewritten query, nothing else.

Query: {query}

Rewritten query:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip().strip('"')


def _select_cited_sources(answer_text: str, chunks: list[dict]) -> list[dict]:
    """
    Return only the chunks the model actually cited in its answer, matched
    by the (filename, SECTION) pairs parsed out of its [Source: ...] tags.

    WHY NOT JUST RETURN EVERY RETRIEVED CHUNK?
    retrieve() may return more chunks than were actually useful — either
    because a query is purely conversational (nothing was cited) or because
    of parent-document expansion (sibling sections pulled in for context
    that the model didn't end up needing). Showing all of them as citation
    chips would be misleading — a "source" the UI highlights should be a
    passage the answer genuinely relied on.
    """
    cited_keys = {
        (fname.strip(), section.strip().upper())
        for fname, section in _CITATION_RE.findall(answer_text)
    }
    if cited_keys:
        return [c for c in chunks if (c["source_file"], c["section"].upper()) in cited_keys]
    if "[Source:" in answer_text:
        # A citation marker is present but didn't match our expected format —
        # fall back to showing everything retrieved rather than nothing.
        return chunks
    return []


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

FIRST, check what kind of message this is:
- If it's a greeting, thanks, or general conversational remark (not a request for
  clinical information) — respond briefly and naturally, in plain conversational
  language. Do NOT use the passages, do NOT cite a source, and do NOT say the
  "not found" line below. This rule overrides everything after it.
- Otherwise, treat it as a clinical question and follow the rules below.

IMPORTANT RULES (for clinical questions only):
- Answer ONLY using the information in the numbered passages below.
- Do not use any outside medical knowledge — only what is explicitly stated in the passages.
- After your answer, cite which passage(s) you used in this format: [Source: filename | SECTION]
- If the answer to the question is not found in any passage, respond with:
  "The answer to this question is not found in the available clinical documents."
- Never guess. Never invent lab values, medications, or diagnoses.

PASSAGES:
{passages}
MESSAGE: {query}

RESPONSE:"""

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
    cited_chunks = _select_cited_sources(answer_text, chunks)

    return {
        "answer": answer_text,
        "sources": [
            {
                "source_file": c["source_file"],
                "section": c["section"],
                "page_number": c.get("page_number", 1),
            }
            for c in cited_chunks
        ],
        "model": GROQ_MODEL,
        "chunks_used": len(cited_chunks),
    }


def stream_answer(query: str, chunks: list[dict]):
    """
    Generator that streams the LLM response as Server-Sent Events (SSE).

    WHAT ARE SERVER-SENT EVENTS?
    SSE is a protocol where the server sends a stream of text messages to
    the browser over a single HTTP connection. Each message is formatted as:
        "data: {JSON}\n\n"
    The double newline signals the end of one event. The browser receives
    events one at a time as they arrive, rather than waiting for the full
    response.

    WHY SSE INSTEAD OF WEBSOCKETS?
    Communication here is one-directional: server sends tokens, client
    displays them. SSE maps perfectly to that. WebSockets are bidirectional
    (client can also send mid-stream) — right tool for a real-time chat
    where the user can interrupt; overkill here.

    EVENT TYPES WE EMIT:
      {"type": "token",   "content": "word "}  — one per LLM output token
      {"type": "sources", "sources": [...]}     — after the last token
      {"type": "done"}                           — signals stream end

    The frontend listens for these events and builds the answer incrementally.
    Sources arrive last because we retrieve them before calling the LLM but
    they're most useful displayed together after the full answer.
    """
    client = _get_client()
    prompt = _build_prompt(query, chunks)

    groq_stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
        stream=True,
    )

    full_answer = ""
    for event in groq_stream:
        token = event.choices[0].delta.content
        if token:
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    cited_chunks = _select_cited_sources(full_answer, chunks)
    sources = [
        {
            "source_file": c["source_file"],
            "section": c["section"],
            "text": c["text"],
            "page_number": c.get("page_number", 1),
        }
        for c in cited_chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield 'data: {"type": "done"}\n\n'

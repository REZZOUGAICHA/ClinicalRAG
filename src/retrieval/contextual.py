"""
Contextual retrieval (Anthropic, "Introducing Contextual Retrieval", Sept 2024).

A chunk like "PATIENT INFORMATION / Patient ID: SYN-2024-001 / Age: 54" says
nothing about which report it belongs to, so a query for "the lymphoma
patient's ID" can't tell it apart from the same section of nine other
reports. An LLM reads the whole document and writes one or two sentences
situating each chunk ("Patient details from the lymphoma staging report of
..."). That context is prepended to the chunk for indexing only: the dense
embedding and BM25 see context + text, while the user and the answer LLM
still get the original chunk text. The context is model-written, so it must
never be quoted as if it were in the source PDF.

One difference from the paper's recipe: the paper makes one call per chunk
(document + chunk) and relies on prompt caching to make that cheap. Groq's
free tier has no prompt caching and a small token budget, so we make one
call per document and ask for every chunk's context in a single JSON reply.
The model still sees the whole document and each chunk; it costs about a
sixth of the tokens.

Contexts are cached in data/contexts/chunk_contexts.json (tracked in git),
keyed by chunk_id and stored with a hash of the chunk text, so a re-chunked
document shows up as stale rather than silently getting the wrong context.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from src.config import settings

CONTEXT_MODEL = "openai/gpt-oss-20b"
CONTEXT_REASONING_EFFORT = "low"
# Reasoning tokens count toward this cap. Building all 37 documents, one call
# used 600-1600 tokens in total, prompt included, so this leaves headroom.
CONTEXT_BASE_TOKENS = 1024
CONTEXT_TOKENS_PER_CHUNK = 160

_PROMPT = """Below is a clinical document, split into numbered chunks.

<document title="{title}">
{chunks}
</document>

Write a short context for each chunk that situates it within the whole
document, to improve search retrieval of the chunk. Reply with only a JSON
object of this shape:

{{"document": "...", "chunks": {{"1": "...", "2": "..."}}}}

"document": one sentence saying what kind of document this is and who or
what it is about: the patient's name or ID and the main diagnosis or
procedure, if the document gives them.
"chunks": for every chunk number, one sentence (at most 30 words) saying what
that chunk covers within this document. Do not start with "Chunk N", and do
not copy the chunk's text.

Use only facts from the document. Include every chunk number."""


def _field(chunk, key: str):
    # The chunker produces dataclasses, chunks.json gives dicts; accept both.
    return chunk[key] if isinstance(chunk, dict) else getattr(chunk, key)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_contexts(path: Path | None = None) -> dict:
    path = path or settings.contexts_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_contexts(contexts: dict, path: Path | None = None) -> None:
    path = path or settings.contexts_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted save can't truncate the cache. The
    # temp name is unique so two uploads saving at once don't share a file.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(contexts, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def cached_context(chunk, contexts: dict) -> str | None:
    """The cached context for this chunk, or None if missing or stale."""
    entry = contexts.get(_field(chunk, "chunk_id"))
    if entry and entry.get("text_hash") == text_hash(_field(chunk, "text")):
        return entry["context"]
    return None


def indexed_text(text: str, context: str | None) -> str:
    """What the embedder and BM25 see for a chunk."""
    return f"{context}\n\n{text}" if context else text


def build_prompt(source_file: str, chunks: list) -> str:
    body = "\n\n".join(
        f'<chunk number="{i}" section="{_field(c, "section")}">\n{_field(c, "text")}\n</chunk>'
        for i, c in enumerate(chunks, 1)
    )
    return _PROMPT.format(title=source_file, chunks=body)


# gpt-oss writes IDs like SYN-2024-001 with U+2011 non-breaking hyphens; the chunks
# and the BM25 tokenizer use ASCII "-", so map them back.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def clean_context(text: str) -> str:
    return " ".join(text.translate(_DASHES).split())


def _parse_reply(reply: str, n: int) -> dict[int, str]:
    # json_object mode should give clean JSON, but strip a code fence just in case
    reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    data = json.loads(reply)
    # The document sentence is asked for once and prepended to every chunk's
    # sentence here. Asking for full per-chunk contexts instead, gpt-oss-20b
    # named the patient/report in the first context and dropped it from the
    # rest, which is exactly the part retrieval needs.
    doc = str(data.get("document") or "")
    out = {}
    for key, value in (data.get("chunks") or {}).items():
        if str(key).isdigit() and 1 <= int(key) <= n and isinstance(value, str) and value.strip():
            out[int(key)] = clean_context(f"{doc} {value}")
    return out


def generate_document_contexts(source_file: str, chunks: list, client=None) -> tuple[dict, int]:
    """
    One LLM call for one document. Returns ({chunk_id: context}, total_tokens).
    Chunks the model skipped are simply missing from the dict. Raises on API
    or parse errors; callers decide whether to retry or fall back.
    """
    if client is None:
        from src.generation.generator import _get_client

        client = _get_client()
    response = client.chat.completions.create(
        model=CONTEXT_MODEL,
        messages=[{"role": "user", "content": build_prompt(source_file, chunks)}],
        temperature=0.0,
        max_completion_tokens=CONTEXT_BASE_TOKENS + CONTEXT_TOKENS_PER_CHUNK * len(chunks),
        reasoning_effort=CONTEXT_REASONING_EFFORT,
        include_reasoning=False,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(f"context reply for {source_file} hit the token cap")
    parsed = _parse_reply(choice.message.content or "", len(chunks))
    contexts = {_field(c, "chunk_id"): parsed[i] for i, c in enumerate(chunks, 1) if i in parsed}
    usage = getattr(response, "usage", None)
    return contexts, getattr(usage, "total_tokens", 0) or 0


def cache_entry(chunk, context: str) -> dict:
    return {
        "source_file": _field(chunk, "source_file"),
        "text_hash": text_hash(_field(chunk, "text")),
        "context": context,
        "model": CONTEXT_MODEL,
    }


def contextualize_chunks(chunks: list) -> list[str | None]:
    """
    Contexts for freshly uploaded chunks, in order. Uses the cache where it
    can, otherwise one LLM call per document. Any failure (no key, rate
    limit, bad JSON) falls back to no context for those chunks: an upload
    should never fail because this optional step did. Failures are not
    cached, so build_contexts.py can fill them in later.
    """
    contexts = load_contexts()
    result = [cached_context(c, contexts) for c in chunks]

    missing: dict[str, list] = {}
    for c, ctx in zip(chunks, result):
        if ctx is None:
            missing.setdefault(_field(c, "source_file"), []).append(c)

    new_entries = {}
    for source_file, doc_chunks in missing.items():
        try:
            generated, _ = generate_document_contexts(source_file, doc_chunks)
        except Exception as e:
            print(f"  contextual: no context for {source_file} ({type(e).__name__}: {e})")
            continue
        for c in doc_chunks:
            if _field(c, "chunk_id") in generated:
                new_entries[_field(c, "chunk_id")] = cache_entry(c, generated[_field(c, "chunk_id")])

    if new_entries:
        contexts = load_contexts()  # re-read: another upload may have saved meanwhile
        contexts.update(new_entries)
        save_contexts(contexts)
        result = [ctx if ctx is not None else new_entries.get(_field(c, "chunk_id"), {}).get("context")
                  for c, ctx in zip(chunks, result)]
    return result


def embed_for_index(chunks: list):
    """
    Document embeddings for new chunks, honouring the configured embedder and
    contextual setting. This is the hook for the upload path; with default
    settings it equals encoding the plain chunk texts with bge-small-en.
    """
    from src.embeddings.embedder import embed_texts

    texts = [_field(c, "text") for c in chunks]
    if settings.contextual:
        texts = [indexed_text(t, ctx) for t, ctx in zip(texts, contextualize_chunks(chunks))]
    titles = [_field(c, "section") for c in chunks]
    return embed_texts(texts, titles=titles, show_progress_bar=False)

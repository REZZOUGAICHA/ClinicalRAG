"""
Tools the agent can call. Each is a thin wrapper over data that already
exists: hybrid retrieval from src.retrieval, and the chunk list in
data/processed/chunks.json. No new retrieval logic lives here.

Every tool returns (text for the model, chunks it showed). The text is
trimmed to keep a multi-step run inside Groq's per-minute token budget; the
chunks travel separately as the ToolMessage artifact so the answer step and
the grounding check work from the full, untrimmed passages.
"""

import json
import re
from pathlib import Path

from langchain_core.tools import tool

CHUNKS_PATH = Path("data/processed/chunks.json")

SEARCH_SNIPPET_CHARS = 350
SECTION_CHARS = 1500
# Each search result costs planner tokens; past 5 the extra hits were mostly
# other patients' copies of the same section. (The reranker shortlist,
# RAG_RERANK_POOL, is 20 by default, so this cap is ours, not retrieve()'s.)
MAX_K = 5
# gpt-oss-20b liked asking for k=1, then re-searching when the one hit was the
# wrong section of the right report. Three is cheap and usually covers both.
MIN_K = 3

_cache: dict = {"mtime": None, "chunks": []}


def _load_chunks() -> list[dict]:
    # Re-read when the file changes, so PDFs uploaded through the API show up.
    mtime = CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None
    if mtime != _cache["mtime"]:
        _cache["chunks"] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8")) if mtime else []
        _cache["mtime"] = mtime
    return _cache["chunks"]


def _retrieve(query: str, k: int) -> list[dict]:
    # Imported lazily: it pulls in the embedder and ChromaDB, which the tests
    # and the simple route never need.
    from src.retrieval.retriever import retrieve

    # No parent expansion: it would append a whole report per search and blow
    # the token budget. The agent can read_section when it needs more.
    return retrieve(query, top_k=k, expand_parents=False)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _resolve_file(name: str, files: list[str]) -> str | None:
    """Accept "report_003", "report_003_acute_mi" or the full filename."""
    name = name.strip()
    if name in files:
        return name
    stem = name.removesuffix(".pdf")
    matches = [f for f in files if f.removesuffix(".pdf") == stem or f.startswith(stem + "_")]
    return matches[0] if len(matches) == 1 else None


def _describe(source_file: str, chunks: list[dict]) -> str:
    by_section = {c["section"]: c["text"] for c in chunks}
    info = by_section.get("PATIENT INFORMATION")
    if info:
        # Synthetic reports: patient ID, sex and age, then the diagnosis line.
        fields = {}
        for part in info.replace("\n", "|").split("|"):
            key, _, value = part.partition(":")
            fields[key.strip()] = value.strip()
        who = ", ".join(v for v in (fields.get("Patient ID"), fields.get("Sex"),
                                    fields.get("Age") and f"age {fields['Age']}") if v)
        diagnosis = by_section.get("DIAGNOSIS", "").split("\n")[0]
        return f"{who}. {_clip(diagnosis, 90)}"
    # MTSamples: the filename slug is the note's title; add its opening line.
    title = source_file.removesuffix(".pdf").split("_", 2)[-1].replace("-", " ")
    first = chunks[0]["text"]
    if chunks[0]["section"] == "FULL_DOCUMENT":
        # These start with a title/specialty/source header; skip past it.
        first = first.split("mtsamples.com)", 1)[-1]
    return f"{title}. {_clip(first, 80)}"


@tool(response_format="content_and_artifact")
def search_documents(query: str, k: int = 4) -> tuple[str, list[dict]]:
    """Search all clinical documents with hybrid keyword + semantic retrieval.

    Returns up to k passages (k is 3-5), each labelled with its source file and
    section, text trimmed. Use short, specific queries naming one entity and
    one fact, e.g. "septic shock WBC" or "bisoprolol dose". Run separate
    searches for separate patients or drugs rather than one combined query.
    """
    k = max(MIN_K, min(int(k), MAX_K))
    chunks = _retrieve(query, k)
    if not chunks:
        return "No passages found.", []
    lines = [f"[{c['source_file']} | {c['section']}] {_clip(c['text'], SEARCH_SNIPPET_CHARS)}"
             for c in chunks]
    return "\n".join(lines), chunks


@tool(response_format="content_and_artifact")
def list_documents() -> tuple[str, list[dict]]:
    """List every document in the collection with a one-line description.

    Synthetic patient reports (report_*.pdf) show patient ID, sex, age and the
    main diagnosis; all of them have the sections PATIENT INFORMATION, CHIEF
    COMPLAINT, DIAGNOSIS, LABORATORY RESULTS, IMAGING, MEDICATIONS, TREATMENT
    PLAN and FOLLOW-UP. MTSamples notes (mts_*.pdf) show the note title and
    opening line; their section names vary. Use this to enumerate patients
    before reading the same section across several of them. The synthetic
    reports' PATIENT INFORMATION and DIAGNOSIS sections behind these lines are
    passed on to the answer, so questions about sex, age or diagnosis need no
    further reads.
    """
    grouped: dict[str, list[dict]] = {}
    for c in _load_chunks():
        grouped.setdefault(c["source_file"], []).append(c)
    if not grouped:
        return "No documents are indexed.", []
    lines = [f"{f}: {_describe(f, cs)}" for f, cs in sorted(grouped.items())]
    # The synthetic-report lines are built from these two sections, so hand
    # them over as citable passages: otherwise the agent "knows" every
    # patient's age and diagnosis but the answer step can't use any of it.
    # MTSamples lines are only a clipped opening line, so they stay uncitable.
    shown = [c for f, cs in sorted(grouped.items()) if f.startswith("report_")
             for c in cs if c["section"] in ("PATIENT INFORMATION", "DIAGNOSIS")]
    return "\n".join(lines), shown


@tool(response_format="content_and_artifact")
def read_section(source_files: list[str], section: str) -> tuple[str, list[dict]]:
    """Read the full text of one section from one or more documents.

    source_files: filenames as given by list_documents or search results
    (e.g. ["report_003_acute_mi.pdf", "report_006_stroke.pdf"]). Pass several
    to read the same section across patients in one call.
    section: section name, case-insensitive, e.g. "MEDICATIONS". If a document
    has no such section, the reply lists the sections it does have.
    """
    all_chunks = _load_chunks()
    files = sorted({c["source_file"] for c in all_chunks})
    wanted = section.strip().upper()
    parts, found = [], []
    for name in source_files:
        source = _resolve_file(name, files)
        if source is None:
            parts.append(f"[{name}] Unknown document.")
            continue
        doc_chunks = [c for c in all_chunks if c["source_file"] == source]
        hits = [c for c in doc_chunks if c["section"].upper() == wanted]
        if not hits:
            available = ", ".join(dict.fromkeys(c["section"] for c in doc_chunks))
            parts.append(f"[{source}] No section {wanted}. Sections: {available}")
            continue
        text = "\n".join(c["text"] for c in hits)
        parts.append(f"[{source} | {hits[0]['section']}] {_clip(text, SECTION_CHARS)}")
        found.extend(hits)
    return "\n\n".join(parts), found


TOOLS = [search_documents, list_documents, read_section]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def step_label(name: str, args: dict) -> str:
    """Short human-readable description of a tool call, for the UI."""
    if name == "search_documents":
        return f'Searching documents for "{args.get("query", "")}"'
    if name == "read_section":
        # "report_003_acute_mi.pdf" -> "report_003"
        files = [re.sub(r"^([a-z]+_\d+).*", r"\1", str(f)) for f in args.get("source_files", [])]
        return f"Reading {args.get('section', '').upper()} in {', '.join(files)}"
    if name == "list_documents":
        return "Listing all documents"
    return f"Calling {name}"

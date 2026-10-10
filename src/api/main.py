"""
ClinicalRAG FastAPI backend.

WHY FASTAPI?
FastAPI is the standard choice for ML-backed APIs in Python. Two main reasons:
1. Async: handles many concurrent requests without blocking on I/O
2. Automatic docs: visiting /docs gives you a working UI to test every endpoint
   for free — no Postman needed

WHAT THIS FILE DOES:
- Defines the HTTP API that the frontend will call
- Preloads all ML models at server startup (so the first request is fast)
- Streams LLM tokens to the browser as they're generated (SSE)
- Handles PDF upload → extract → chunk → embed → store

ENDPOINTS:
  GET  /health      → server status + how many chunks are indexed
  GET  /documents   → list of ingested document filenames
  POST /ask         → streaming Q&A (returns SSE stream). "mode": "pipeline"
                      (default), "agent" (LangGraph agent in src/agent), or
                      "auto" (router decides; what the web UI uses)
  POST /upload      → upload a PDF and ingest it into the pipeline

HOW TO RUN:
  uvicorn src.api.main:app --reload --port 8000

Then open http://localhost:8000/docs to see and test all endpoints.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import classify_question, stream_agent
from src.agent.llm import agent_model
from src.chunking.chunker import chunk_document
from src.embeddings.embedder import MODEL_NAME as EMBEDDING_MODEL
from src.embeddings.embedder import get_model
from src.generation.generator import GROQ_MODEL, rewrite_query, stream_answer
from src.ocr.extractor import extract
from src.retrieval.contextual import embed_for_index
from src.retrieval.retriever import RERANKER_MODEL, _get_bm25, get_reranker, reset_bm25, retrieve
from src.tracing import setup_tracing
from src.vectorstore.store import get_collection

RAW_DIR = Path("data/raw")
CHUNKS_JSON = Path("data/processed/chunks.json")

log = logging.getLogger("uvicorn.error")
# One upload at a time: each one reads and rewrites chunks.json and the
# contexts cache, and two interleaved would drop each other's chunks.
_ingest_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Startup — load all models before the first request arrives
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: code before `yield` runs at startup, code after at shutdown.

    WHY PRELOAD MODELS HERE?
    Loading bge-small-en and the cross-encoder each take 2-4 seconds the first
    time (reading from disk into RAM). If we lazy-load them on the first request,
    that user waits 5+ seconds. Preloading at startup means every request after
    that hits already-warm models — response time drops to ~1s (network to Groq).

    This is the key architectural difference between the slow CLI and the fast server:
    the CLI pays model-loading cost on every run; the server pays it once.
    """
    print("Starting ClinicalRAG server...")
    setup_tracing()  # no-op unless CLINICALRAG_TRACING is set
    print(f"  Loading embedding model ({EMBEDDING_MODEL})...")
    get_model()
    print(f"  Loading reranker ({RERANKER_MODEL})...")
    get_reranker()
    print("  Opening ChromaDB collection...")
    get_collection()
    print("  Building BM25 index from chunks.json...")
    _get_bm25()
    print(f"  Answer model: {GROQ_MODEL} (pipeline), {agent_model()} (agent)")
    print("All models loaded. Server ready.\n")
    yield
    # Nothing to clean up on shutdown


app = FastAPI(
    title="ClinicalRAG",
    description="Multimodal clinical document RAG system with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — the deployed frontend (Netlify) calls this API on another domain
# ---------------------------------------------------------------------------

# Comma-separated, e.g. "https://clinicalrag.netlify.app". Locally the frontend
# is served by this same app, so the "*" default never matters there. No
# cookies are used, so credentials stay off ("*" with credentials is invalid).
CORS_ORIGINS = [o.strip() for o in os.getenv("CLINICALRAG_CORS_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB = float(os.getenv("CLINICALRAG_MAX_UPLOAD_MB", "10"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """
    Pydantic model for the /ask request body.

    WHY PYDANTIC?
    FastAPI uses Pydantic to validate incoming JSON automatically. If the
    frontend sends {"query": 42} instead of a string, FastAPI rejects it
    with a clear error before your code even runs. No manual validation needed.
    """
    query: str
    top_k: int = 3
    mode: Literal["pipeline", "agent", "auto"] = "pipeline"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse("src/frontend/index.html")


@app.get("/health")
async def health():
    """Quick liveness check. Also tells you how many chunks are indexed."""
    col = get_collection()
    return {
        "status": "ok",
        "chunks_indexed": col.count(),
    }


@app.get("/documents")
async def list_documents():
    """
    Return the list of unique document filenames currently in the vector store.
    The frontend uses this to populate the document sidebar.
    """
    col = get_collection()
    result = col.get(include=["metadatas"])
    if not result["metadatas"]:
        return {"documents": [], "total": 0}
    files = sorted({m["source_file"] for m in result["metadatas"]})
    return {"documents": files, "total": len(files)}


@app.get("/documents/{filename}")
async def get_document(filename: str):
    """
    Serve the raw PDF for a document already in data/raw/, so the sidebar
    can open the original report in a new tab.
    """
    dest = (RAW_DIR / filename).resolve()
    if RAW_DIR.resolve() not in dest.parents or not dest.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(dest, media_type="application/pdf")


@app.post("/ask")
async def ask(req: QuestionRequest):
    """
    Main Q&A endpoint. Returns a Server-Sent Events stream.

    FLOW:
    1. retrieve()       — hybrid search (BM25 + ChromaDB + RRF + reranker)
    2. stream_answer()  — Groq streams tokens; we forward them as SSE events

    WHY StreamingResponse?
    Without streaming: browser waits ~2s for the full answer, then sees it all
    at once. With streaming: browser receives the first token in ~300ms and
    renders the answer word by word. Identical total time, dramatically better
    perceived speed.

    The `X-Accel-Buffering: no` header tells nginx (if present) not to buffer
    the stream — without it, nginx accumulates the whole response before sending.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if req.mode == "agent":
        events = _agent_events(req.query)
    elif req.mode == "auto":
        events = _auto_events(req.query, req.top_k)
    else:
        events = _pipeline_events(req.query, req.top_k)
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _pipeline_events(query: str, top_k: int):
    """
    A plain (sync) generator, like _agent_events: StreamingResponse runs it in
    a worker thread, so the rewrite, retrieval and Groq calls don't block the
    event loop. If any of them fails (a Groq 429 mid-stream, say), the client
    still gets an error event and "done" instead of a stream that just stops.
    """
    try:
        # Retrieval is sensitive to exact wording (typos can drop the right
        # document out of the shortlist) — search on a typo-corrected version,
        # but still generate the answer against the user's original message.
        search_query = rewrite_query(query)
        chunks = retrieve(search_query, top_k)
        yield from stream_answer(query, chunks)
    except Exception as e:
        # Groq error strings carry the org ID; keep the detail in the server log.
        log.exception("pipeline answer failed")
        yield _sse({"type": "error", "message": f"Answer failed: {type(e).__name__}. Please try again."})
        yield _sse({"type": "done"})


def _auto_events(query: str, top_k: int):
    """
    Classify once, then take the cheaper path when it's enough: simple
    questions go through the streaming pipeline unchanged, complex ones to
    the agent with the decision passed in so the router doesn't run twice.
    """
    route = classify_question(query)
    if route == "simple":
        yield _sse({"type": "step", "content": "Single lookup, using the standard pipeline",
                    "route": "simple", "node": "route"})
        yield from _pipeline_events(query, top_k)
    else:
        yield from _agent_events(query, route=route)


def _agent_events(query: str, route: str | None = None):
    """
    Same event protocol as stream_answer, plus "step" events while the agent
    works and a "node" event each time a graph node starts (for the live
    graph in the UI). The answer arrives as a single token event rather than streamed:
    the grounding check may rewrite or strip citations after generation, so
    streaming the draft would show text that later changes.

    A plain (sync) generator: StreamingResponse iterates it in a worker
    thread, so the blocking LLM and retrieval calls don't stall the event loop.
    """
    try:
        for ev in stream_agent(query, route=route):
            if ev["type"] == "route":
                label = ("Single lookup, using the standard pipeline" if ev["route"] == "simple"
                         else "Multi-step question, planning lookups")
                yield _sse({"type": "step", "content": label, "route": ev["route"],
                            "node": ev["node"]})
            elif ev["type"] in ("step", "node"):
                yield _sse(ev)
            elif ev["type"] == "result":
                result = ev["result"]
                yield _sse({"type": "token", "content": result["answer"]})
                yield _sse({"type": "sources", "sources": result["sources"]})
                yield _sse({"type": "trace", "route": result["route"], "trace": [
                    {"step": t["step"], "summary": t["summary"]} for t in result["trace"]]})
    except Exception as e:
        log.exception("agent run failed")
        yield _sse({"type": "error", "message": f"Agent failed: {type(e).__name__}. Please try again."})
    yield _sse({"type": "done"})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, chunk it, embed it, and add it to the
    vector store. After this call, the document is immediately queryable.

    FULL PIPELINE ON UPLOAD:
      PDF → PyMuPDF extract → section-aware chunker → bge-small-en embed
      → ChromaDB store → chunks.json append → BM25 cache reset

    WHY reset BM25 after upload?
    The BM25 index is built from chunks.json in memory. After we append new
    chunks to chunks.json, the cached index is stale — it doesn't know about
    the new chunks. reset_bm25() clears the cache so _get_bm25() rebuilds it
    from the updated file on the next query.
    """
    # Only the base name: "../../x.pdf" must not escape data/raw/.
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # On a public deployment a huge scanned PDF would run OCR for minutes on
    # someone else's compute budget.
    if file.size is not None and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDFs up to {MAX_UPLOAD_MB:g} MB only.")
    return await run_in_threadpool(_ingest_upload, filename, file.file)


def _ingest_upload(filename: str, data) -> dict:
    with _ingest_lock:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = RAW_DIR / filename
        with dest.open("wb") as f:
            shutil.copyfileobj(data, f)

        chunks = chunk_document(extract(dest))
        if not chunks:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this PDF. Is it a scanned image?"
            )

        # Filter out chunks already in ChromaDB (handles re-upload of the same file)
        col = get_collection()
        existing_ids = set(col.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if new_chunks:
            embeddings = embed_for_index(new_chunks)
            col.add(
                ids=[c.chunk_id for c in new_chunks],
                embeddings=embeddings.tolist(),
                documents=[c.text for c in new_chunks],
                metadatas=[
                    {
                        "source_file": c.source_file,
                        "section": c.section,
                        "page_number": c.page_number,
                        "char_count": c.char_count,
                    }
                    for c in new_chunks
                ],
            )

            # Append new chunks to chunks.json so BM25 knows about them
            existing_json = (
                json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
                if CHUNKS_JSON.exists()
                else []
            )
            existing_json_ids = {c["chunk_id"] for c in existing_json}
            to_append = [c.to_dict() for c in new_chunks if c.chunk_id not in existing_json_ids]
            if to_append:
                _write_json_atomic(CHUNKS_JSON, existing_json + to_append)
                reset_bm25()

        return {
            "filename": filename,
            "chunks_added": len(new_chunks),
            "total_chunks_indexed": col.count(),
        }


def _write_json_atomic(path: Path, data) -> None:
    # The agent's tools re-read chunks.json whenever it changes, so it must
    # never be seen half-written.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

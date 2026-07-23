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
  POST /ask         → streaming Q&A (returns SSE stream)
  POST /upload      → upload a PDF and ingest it into the pipeline

HOW TO RUN:
  uvicorn src.api.main:app --reload --port 8000

Then open http://localhost:8000/docs to see and test all endpoints.
"""

import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chunking.chunker import chunk_document
from src.embeddings.embedder import get_model
from src.generation.generator import stream_answer
from src.ocr.extractor import extract
from src.retrieval.retriever import _get_bm25, get_reranker, reset_bm25, retrieve
from src.vectorstore.store import get_collection

RAW_DIR = Path("data/raw")
CHUNKS_JSON = Path("data/processed/chunks.json")


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
    print("  Loading embedding model (bge-small-en)...")
    get_model()
    print("  Loading reranker (ms-marco-MiniLM-L-6-v2)...")
    get_reranker()
    print("  Opening ChromaDB collection...")
    get_collection()
    print("  Building BM25 index from chunks.json...")
    _get_bm25()
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
# CORS — allow the frontend (different port) to call this API
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
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

    # retrieve() is CPU-bound (embedding + BM25 + reranker) — run in thread pool
    # so we don't block FastAPI's async event loop
    chunks = await run_in_threadpool(retrieve, req.query, req.top_k)

    return StreamingResponse(
        stream_answer(req.query, chunks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _embed_batch(texts: list[str]):
    """
    Embed a list of texts without showing a progress bar.
    Used for upload ingestion where the bar would clutter server logs.
    """
    model = get_model()
    return model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to data/raw/
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text (CPU-bound → thread pool)
    doc = await run_in_threadpool(extract, dest)
    chunks = chunk_document(doc)

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
        # Embed new chunks (CPU-bound → thread pool)
        texts = [c.text for c in new_chunks]
        embeddings = await run_in_threadpool(_embed_batch, texts)

        col.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
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
        to_append = [
            c.to_dict() for c in new_chunks if c.chunk_id not in existing_json_ids
        ]
        if to_append:
            CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
            CHUNKS_JSON.write_text(
                json.dumps(existing_json + to_append, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            reset_bm25()

    return {
        "filename": file.filename,
        "chunks_added": len(new_chunks),
        "total_chunks_indexed": col.count(),
    }

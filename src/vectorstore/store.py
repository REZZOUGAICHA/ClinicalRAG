"""
Vector store module — stores and searches chunk embeddings using ChromaDB.

WHY DO WE NEED A VECTOR DATABASE?
After embedding 80 chunks into 384-dimensional vectors, we need somewhere
to store them and a way to search them efficiently. A vector database does
exactly this: it stores (vector, metadata) pairs and answers the query
"give me the N vectors most similar to this query vector."

WHY ChromaDB?
- Runs entirely on your local machine — no server to start, no account needed
- Stores everything in a folder on disk (data/vectorstore/)
- Python API is clean and minimal
- When we deploy, we switch to Qdrant Cloud (free tier) — that swap is ~5 lines
  of code because both follow the same store/query pattern we establish here

HOW ChromaDB FINDS SIMILAR VECTORS:
Internally, ChromaDB uses a library called HNSW (Hierarchical Navigable Small
World graphs) — an approximate nearest-neighbour index. This lets it search
millions of vectors in milliseconds without comparing against every single one.
For 80 chunks you wouldn't notice the difference with brute force, but the
architecture is production-grade from day one.

WHAT IS "COSINE SIMILARITY"?
When we normalised the vectors in embedder.py (normalize_embeddings=True),
each vector has length 1. For unit vectors, cosine similarity equals dot product.
ChromaDB's default metric is cosine similarity. Score range: -1 (opposite meaning)
to 1 (identical meaning). In practice, good matches are above 0.7.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import settings

COLLECTION_NAME = "clinical_chunks"
VECTORSTORE_DIR = settings.vectorstore_dir

_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """
    Get (or create) the ChromaDB collection — singleton pattern.

    WHY SINGLETON?
    The original version created a new PersistentClient on every call.
    That meant reopening all the database files on every query, plus
    ChromaDB firing startup telemetry events on every call.
    A module-level singleton creates the client once per Python process
    and reuses it for every subsequent query — exactly how a production
    server behaves.

    In FastAPI: the server starts, this runs once, all requests share
    the same client. Zero overhead per query.
    In the CLI: the client is created on the first query and reused for
    every subsequent question in the same run.
    """
    global _collection
    if _collection is None:
        _collection = _client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=_collection_metadata(),
        )
        _check_built_with(_collection)
    return _collection


def _client() -> chromadb.ClientAPI:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_metadata() -> dict:
    # MedCPT was trained with dot-product scores on unnormalised vectors.
    # Recording how the index was built lets get_collection() refuse a query
    # setup that doesn't match it.
    return {
        "hnsw:space": "ip" if settings.embedder == "medcpt" else "cosine",
        "embedder": settings.embedder,
        "contextual": settings.contextual,
    }


def _check_built_with(collection: chromadb.Collection) -> None:
    # Indexes built before this metadata existed are plain bge-small-en.
    meta = collection.metadata or {}
    built = (meta.get("embedder", "bge-small-en"), bool(meta.get("contextual", False)))
    wanted = (settings.embedder, settings.contextual)
    if built != wanted:
        raise RuntimeError(
            f"{VECTORSTORE_DIR} was built with embedder={built[0]}, contextual={built[1]}, "
            f"but the current settings are embedder={wanted[0]}, contextual={wanted[1]}. "
            "Rebuild it with scripts/ingest.py or point RAG_VECTORSTORE_DIR at a matching index."
        )


def ingest_chunks(chunks_json_path: Path) -> int:
    """
    Load chunks from the JSON file (built in Phase 1) and store them
    in ChromaDB alongside their embeddings.

    WHY load from JSON instead of re-running the chunker?
    Separation of concerns. The chunker's job (Phase 1) is already done.
    The ingester's job is: take finished chunks → embed → store.
    If we change the embedding model later, we re-run ingest without touching
    the chunker. Clean boundaries between pipeline stages.

    ChromaDB's add() takes:
    - ids:        unique string identifiers (our chunk_ids)
    - embeddings: the vectors (list of lists of floats)
    - documents:  the raw text (stored so we can retrieve it later)
    - metadatas:  dicts of extra info per chunk (source file, section)

    WHY store metadata?
    Metadata is what powers citations. When ChromaDB returns a match,
    it also returns the metadata — so we get "this came from
    report_001_lymphoma_staging.pdf, section LABORATORY RESULTS" for free.
    """
    from src.embeddings.embedder import embed_texts
    from src.retrieval.contextual import cached_context, indexed_text, load_contexts

    with open(chunks_json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # Drop and recreate rather than clearing the rows, so the collection's
    # metadata (embedder, contextual) always describes what's in it.
    global _collection
    client = _client()
    if COLLECTION_NAME in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        print("  Collection already exists. Clearing and re-ingesting...")
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME, metadata=_collection_metadata())
    _collection = collection

    texts = [c["text"] for c in chunks]
    # With contextual retrieval on, the vector is built from context + text,
    # but the stored document stays the original text: that's what search()
    # returns, and what the user and the answer LLM should see.
    to_embed = texts
    if settings.contextual:
        contexts = load_contexts()
        found = [cached_context(c, contexts) for c in chunks]
        applied = sum(ctx is not None for ctx in found)
        stale = sum(ctx is None and c["chunk_id"] in contexts for c, ctx in zip(chunks, found))
        missing = len(chunks) - applied - stale
        # Stale means the chunk text changed since its context was written
        # (e.g. extraction differs between machines); visible in build logs.
        print(f"  Contexts ({settings.contexts_path}): {applied} applied, {stale} stale, {missing} missing")
        if stale or missing:
            print("  WARNING: those chunks are indexed without context; run scripts/build_contexts.py")
        to_embed = [indexed_text(t, ctx) for t, ctx in zip(texts, found)]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(to_embed, titles=[c["section"] for c in chunks])

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),      # ChromaDB expects plain Python lists
        documents=texts,
        metadatas=[
            {
                "source_file": c["source_file"],
                "section": c["section"],
                "page_number": c["page_number"],
                "char_count": c["char_count"],
            }
            for c in chunks
        ],
    )

    return len(chunks)


def search(query_vector, n_results: int = 5) -> list[dict]:
    """
    Find the N most similar chunks to a query vector.

    Returns a list of dicts, each with:
    - text:        the chunk's raw text
    - source_file: which PDF it came from
    - section:     which section of that PDF
    - score:       similarity score (higher = more relevant, max 1.0)

    WHY n_results=5?
    We retrieve more than we'll ultimately send to the LLM (we'll only
    send the top 3 after reranking). Retrieving 5 gives the reranker
    enough candidates to work with. This "retrieve broad, then narrow"
    pattern is standard in production RAG systems.

    WHY does ChromaDB return "distances" and not "similarities"?
    ChromaDB returns distance (lower = more similar) even when the metric
    is cosine. We convert: similarity = 1 - distance.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source_file": results["metadatas"][0][i]["source_file"],
            "section": results["metadatas"][0][i]["section"],
            "page_number": results["metadatas"][0][i].get("page_number", 1),
            "score": round(1 - results["distances"][0][i], 4),
        })
    return output

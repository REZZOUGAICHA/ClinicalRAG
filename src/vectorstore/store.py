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

COLLECTION_NAME = "clinical_chunks"
VECTORSTORE_DIR = Path("data/vectorstore")

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
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


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

    with open(chunks_json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    collection = get_collection()

    # Check if already ingested — avoid duplicates on re-run
    existing_count = collection.count()
    if existing_count > 0:
        print(f"  Collection already has {existing_count} items. Clearing and re-ingesting...")
        collection.delete(where={"source_file": {"$ne": ""}})  # delete all

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

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

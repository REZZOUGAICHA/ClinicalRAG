"""
Retrieval module — takes a user question and returns the most relevant chunks.

This is the "R" in RAG. Its job is to bridge the gap between
"user's question" and "the right passage from the documents."

TWO-STAGE RETRIEVAL:

Stage 1 — Bi-encoder search (fast, approximate):
  Embed the question → search ChromaDB → get top-5 candidates.
  This uses the bge-small-en model we already have.
  It's fast because both question and chunks are pre-embedded.

Stage 2 — Cross-encoder reranking (slower, more precise):
  Take the top-5 candidates → re-score each one with a cross-encoder.
  The cross-encoder reads the question AND the chunk together (not
  separately like the bi-encoder), which gives it much better judgment
  about relevance. It's slower because it can't use pre-computed vectors —
  it needs to re-read the pair.

WHY TWO STAGES instead of just using the cross-encoder directly?
Because you can't run a cross-encoder against all 80 chunks on every query —
it's too slow (the cross-encoder reads both texts together, so it can't
pre-compute anything). The bi-encoder does the cheap broad search,
the cross-encoder does the expensive precise ranking on a small shortlist.
This is the standard production pattern in information retrieval.

PHD ANGLE:
This two-stage pipeline is called "retrieve-then-rerank" in the NLP literature.
Papers like "Sentence-BERT" and "ColBERT" are the theoretical foundation.
In biomedical RAG (MedRAG), the reranker consistently improves faithfulness
scores by reducing cases where a topically-related but non-answering chunk
gets sent to the LLM.
"""

from sentence_transformers import CrossEncoder
from src.embeddings.embedder import embed_query
from src.vectorstore.store import search

# Cross-encoder model. Downloads ~80MB on first run, cached after.
# ms-marco-MiniLM-L-6-v2 is trained on the MS MARCO passage ranking dataset —
# the standard benchmark for this task. "MiniLM" means it's distilled
# (small + fast) while keeping most of the quality.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"  Loading reranker model: {RERANKER_MODEL}...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve(query: str, top_k: int = 3, use_reranker: bool = True) -> list[dict]:
    """
    Main retrieval function. Given a user's question, returns the top_k
    most relevant chunks, each with source and section metadata for citations.

    Args:
        query:        the user's natural language question
        top_k:        how many chunks to return (default 3)
        use_reranker: whether to apply cross-encoder reranking (default True)
                      set to False if you want speed over precision

    Returns a list of dicts with keys: text, source_file, section, score

    WHY top_k=3?
    Sending too many chunks to the LLM increases cost and can actually
    hurt answer quality — the LLM may get distracted by less relevant
    passages. 3 is a good default that balances coverage vs. precision.
    We fetch 5 from ChromaDB, rerank, then keep the top 3.

    WHAT A SENIOR INTERVIEWER WILL ASK:
    "What happens when the answer spans multiple chunks from different documents?"
    Answer: top_k=3 allows cross-document answers. The LLM sees chunks from
    up to 3 different source files and can synthesise across them.
    """
    # Stage 1: fast approximate search
    candidates = search(embed_query(query), n_results=5)

    if not use_reranker or len(candidates) <= 1:
        return candidates[:top_k]

    # Stage 2: precise reranking
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach reranker scores and sort — higher is better
    for i, candidate in enumerate(candidates):
        candidate["reranker_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]

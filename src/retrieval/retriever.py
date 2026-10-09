"""
Retrieval module — upgraded from two-stage to three-stage HYBRID retrieval.

THE PROBLEM WITH SEMANTIC-ONLY SEARCH:
Pure semantic search (dense vectors) finds chunks whose *meaning* is similar to
the query. But medical text is full of precise terminology — abbreviations like
"WBC", drug names like "Apixaban", gene names like "BRCA1" — where the user
might phrase things differently from how the document phrases them. If the query
says "white blood cell count" and the chunk says "WBC", semantic search might
miss it. If the query says "blood thinner" and the chunk says "anticoagulant",
keyword search would miss it.

THE SOLUTION — HYBRID SEARCH:
Combine two complementary signals:
1. Dense (semantic): bi-encoder embeddings capture meaning and synonymy
2. Sparse (BM25): exact keyword match captures precise medical terminology

Then fuse them with RRF (Reciprocal Rank Fusion) before the reranker.

FULL PIPELINE:
  Stage 1a: ChromaDB dense search → top 10 semantic candidates
  Stage 1b: BM25 keyword search  → top 10 keyword candidates
  Stage 2:  RRF fusion           → merged, deduplicated shortlist of up to
                                   RAG_RERANK_POOL (20 by default)
  Stage 3:  Cross-encoder rerank → precise final ranking
(the per-stage counts are configurable, see src/config.py)
  Output:   top_k results (default 3) with source metadata

WHY BM25 SPECIFICALLY?
BM25 (Best Match 25) is the Okapi IR group's 25th iteration of the "Best Match"
algorithm family, published in 1994. It scores a document for a query by:
  - Term Frequency (tf): how often the query term appears in the document
  - Inverse Document Frequency (idf): how rare the term is across all documents
  - Length normalisation: penalises very long documents for repeating terms

It's still the baseline that modern dense retrievers are benchmarked against —
because for out-of-vocabulary terms and exact abbreviations, nothing beats it.

WHY RRF OVER SCORE AVERAGING?
ChromaDB returns cosine distances (0–2). BM25 returns relevance floats (0–N).
These scales are incompatible — you can't average them. RRF uses only rank
positions, which are always comparable: `score = Σ 1/(k + rank)` where k=60
is the standard smoothing constant from Cormack et al. 2009. A document near
the top of both lists gets a higher fused score than one that only dominates one.

PHD ANGLE:
Hybrid retrieval consistently outperforms dense-only on medical benchmarks
(BioASQ, MedQA-USMLE). The specific improvement is in Precision@1 for
clinical terminology queries — exactly the use case here. Citable as:
Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", SIGIR 2009.
"""

import json
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import RERANKERS, settings
from src.embeddings.embedder import embed_query
from src.retrieval.contextual import cached_context, indexed_text, load_contexts
from src.vectorstore.store import search

RERANKER_MODEL = RERANKERS[settings.reranker]
CHUNKS_PATH = Path("data/processed/chunks.json")

_reranker: CrossEncoder | None = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None
# The API serves requests from a thread pool and an upload resets the cache,
# so the index and its chunk list are only ever read and swapped together.
_bm25_lock = threading.Lock()
# chunk_id -> LLM-written context, filled only when RAG_CONTEXTUAL=1
_contexts: dict[str, str] = {}

# A token is a run of word characters, optionally joined by ".", "-" or "/"
# when another word character follows. That keeps "syn-2024-001", "9.2" and
# "mg/dl" whole but drops sentence punctuation ("wbc," -> "wbc").
_TOKEN_RE = re.compile(r"\w+(?:[./-]\w+)*")


def tokenize(text: str) -> list[str]:
    """
    Lowercased BM25 tokens. Used for both the index and the query, so the two
    can't drift apart.

    Hyphen/slash compounds also emit their parts, so "post-op" still matches a
    query for "post" and "nausea/vomiting" matches "nausea". Decimals are not
    split: "9" and "2" on their own would only add noise.
    """
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "-" in tok or "/" in tok:
            tokens.extend(re.split(r"[-/]", tok))
    return tokens


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"  Loading reranker: {RERANKER_MODEL} ...")
        if settings.reranker == "ms-marco-minilm":
            _reranker = CrossEncoder(RERANKER_MODEL)
        else:
            # these accept 512 tokens; set it explicitly rather than trusting
            # each repo's config
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def _get_bm25() -> tuple[BM25Okapi | None, list[dict] | None]:
    """
    Lazy-load the BM25 index from chunks.json.

    WHY LAZY?
    Building the index takes ~10ms for 80 chunks — negligible, but we don't
    want module import to depend on the data file existing. Lazy init means the
    module loads cleanly even before ingest has been run.

    WHY FROM chunks.json AND NOT ChromaDB?
    BM25 needs raw text, not vectors. chunks.json already has raw text plus
    chunk_ids. Reading from JSON is simpler than re-querying ChromaDB for all
    documents. Both sources are always in sync after ingest.
    """
    global _bm25_index, _bm25_chunks, _contexts
    with _bm25_lock:
        if _bm25_index is None:
            if not CHUNKS_PATH.exists():
                return None, None
            with open(CHUNKS_PATH, encoding="utf-8") as f:
                chunks = json.load(f)
            contexts = {}
            if settings.contextual:
                # Stale or missing contexts are skipped; that chunk is indexed on
                # its own text, same as without contextual retrieval.
                cache = load_contexts()
                contexts = {c["chunk_id"]: ctx for c in chunks
                            if (ctx := cached_context(c, cache)) is not None}
            tokenized = [tokenize(indexed_text(c["text"], contexts.get(c["chunk_id"]))) for c in chunks]
            _bm25_index, _bm25_chunks, _contexts = BM25Okapi(tokenized), chunks, contexts
        return _bm25_index, _bm25_chunks


def _bm25_search(query: str, n: int = 10) -> list[dict]:
    """
    Keyword search using BM25. Returns up to n chunks with non-zero scores.

    WHY FILTER ZERO-SCORE CHUNKS?
    BM25 scores a chunk as 0 when none of the query tokens appear in it.
    Including zero-score chunks in the fusion would unfairly boost documents
    that just happen to appear in the BM25 index but have no relevance.
    """
    bm25, chunks = _get_bm25()
    if bm25 is None:
        return []

    raw_scores = bm25.get_scores(tokenize(query))

    top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:n]

    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "source_file": chunks[i]["source_file"],
            "section": chunks[i]["section"],
            "page_number": chunks[i].get("page_number", 1),
            "score": float(raw_scores[i]),
        }
        for i in top_indices
        if raw_scores[i] > 0
    ]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunk IDs.

    Formula (Cormack 2009):
        score(d) = Σ_lists  1 / (k + rank(d, list))

    k=60 is the original paper's recommendation. It smooths the weight
    difference between rank 1 and rank 2 (1/61 vs 1/62) to avoid overfitting
    to the exact position within any single retrieval list.

    Returns IDs sorted descending by fused score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _expand_to_parent_documents(chunks: list[dict]) -> list[dict]:
    """
    Pull in sibling chunks from the same source PDF as the single
    highest-ranked chunk ("parent document expansion" / "small-to-big
    retrieval"). `chunks` must already be sorted best-match-first — true for
    both callers in retrieve() (reranker-sorted or RRF-sorted).

    WHY THIS EXISTS:
    Retrieval scores each chunk independently, so a question needing info
    split across two sections of the SAME report can correctly identify the
    right document but still miss the section that actually answers it.
    Example: "what medications is the lymphoma patient on?" — "lymphoma"
    only appears in the DIAGNOSIS chunk; the drug list lives in a separate
    MEDICATIONS chunk. Retrieval might correctly rank DIAGNOSIS #1, but
    without expansion, MEDICATIONS never makes it into the prompt.

    WHY ONLY THE TOP-RANKED DOCUMENT, NOT EVERY DOCUMENT IN THE TOP-K:
    An earlier version expanded every document represented anywhere in the
    top-k. For a generic-sounding query, the top-k often spans 2-3 different
    patients' reports (structurally similar sections score similarly), which
    expanded into 2-3 *entire* patient records — most of it irrelevant. A
    RAGAS eval caught this concretely: one question retrieved 16 chunks but
    only used 1, scoring context_precision of 0.33. Expanding only the
    best-ranked chunk's document keeps the original fix while not dragging
    in unrelated patients just because one of their sections ranked #3.
    """
    _, all_chunks = _get_bm25()
    if not all_chunks or not chunks:
        return chunks

    seen_ids = {c["chunk_id"] for c in chunks}
    top_source_file = chunks[0]["source_file"]

    expanded = list(chunks)
    for c in all_chunks:
        if c["source_file"] == top_source_file and c["chunk_id"] not in seen_ids:
            expanded.append(c)
            seen_ids.add(c["chunk_id"])

    return expanded


def retrieve(
    query: str,
    top_k: int = 3,
    use_reranker: bool = True,
    *,
    expand_parents: bool = True,
    rerank_pool: int | None = None,
    first_stage_k: int | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: dense + sparse → RRF → rerank → top_k → parent expansion.

    Args:
        query:        the user's natural language question
        top_k:        how many top-scoring chunks to select (default 3) —
                      note the final return can be larger, since sibling
                      chunks from the same source document(s) get appended
        use_reranker: set False to skip Stage 3 (faster, less precise)
        expand_parents: set False to get just the ranked top_k, without
                      sibling chunks appended (used by the retrieval eval)
        rerank_pool, first_stage_k: override RAG_RERANK_POOL /
                      RAG_FIRST_STAGE_K (see src/config.py) for this call

    Returns list of dicts: chunk_id, text, source_file, section, score,
                           reranker_score (if reranker used)

    WHAT A SENIOR INTERVIEWER WILL ASK:
    "Why cast to 10 candidates in each stage before fusion?"
    Because with two retrieval methods, we need a wider initial pool — the
    interesting candidates might be ranked 6th in semantic but 1st in BM25.
    RRF finds them; the reranker then rescores the fused shortlist.
    """
    rerank_pool = rerank_pool or settings.rerank_pool
    first_stage_k = first_stage_k or settings.first_stage_k

    # Stage 1a: dense semantic search
    semantic_results = search(embed_query(query), n_results=first_stage_k)
    semantic_ids = [r["chunk_id"] for r in semantic_results]

    # Stage 1b: sparse keyword search
    keyword_results = _bm25_search(query, n=first_stage_k)
    keyword_ids = [r["chunk_id"] for r in keyword_results]

    # Stage 2: RRF fusion — merge both ranked lists by rank position
    fused_ids = _rrf([semantic_ids, keyword_ids])

    # Build a lookup so we can resolve IDs → full chunk dicts
    chunk_lookup: dict[str, dict] = {}
    for r in semantic_results + keyword_results:
        chunk_lookup[r["chunk_id"]] = r

    # Take the top rerank_pool (default 20) after RRF, not all 10+10.
    # RRF already did the coarse ranking; the reranker only needs a short
    # list. Reranker cost grows linearly with the pool (see
    # eval/results/retrieval_experiments.md for measured numbers).
    candidates = [chunk_lookup[cid] for cid in fused_ids[:rerank_pool] if cid in chunk_lookup]

    if not use_reranker or len(candidates) <= 1:
        top = candidates[:top_k]
        return _expand_to_parent_documents(top) if expand_parents else top

    # Stage 3: cross-encoder reranking on the fused candidates
    reranker = get_reranker()
    if settings.rerank_with_context:
        _get_bm25()  # loads _contexts
        pairs = [(query, indexed_text(c["text"], _contexts.get(c["chunk_id"]))) for c in candidates]
    else:
        pairs = [(query, c["text"]) for c in candidates]
    reranker_scores = reranker.predict(pairs, batch_size=settings.rerank_batch_size)

    for i, c in enumerate(candidates):
        c["reranker_score"] = float(reranker_scores[i])

    top = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)[:top_k]
    return _expand_to_parent_documents(top) if expand_parents else top


def reset_bm25() -> None:
    """
    Invalidate the BM25 cache so it rebuilds from chunks.json on the next query.
    Call this after ingesting a new document so BM25 picks up the new chunks.
    """
    global _bm25_index, _bm25_chunks
    with _bm25_lock:
        _bm25_index, _bm25_chunks = None, None

r"""
Retrieval settings, read from environment variables in one place.

Defaults are the configuration chosen from the experiments in
eval/results/retrieval_experiments.md (ctx-pool20-b1): contextual chunks,
a 20-candidate rerank pool, reranker batch size 1. Local runs, the API and
the Docker image all use them with no env vars set.

The pre-experiment pipeline ("baseline" in eval/results) is still reachable:

    RAG_CONTEXTUAL=0 RAG_RERANK_POOL=5 RAG_RERANK_BATCH=32 \
    RAG_VECTORSTORE_DIR=data/vectorstore_exp/plain-bge-small-en \
    python scripts/eval_retrieval.py --config baseline-repro

It needs its own non-contextual index; build it with the same env vars and
scripts/ingest.py.

RAG_EMBEDDER         bge-small-en (default) | bge-small-en-v1.5 | medcpt
RAG_VECTORSTORE_DIR  Chroma folder; anything but the default embedder needs
                     its own index (see scripts/ingest.py)
RAG_RERANKER         ms-marco-minilm (default) | medcpt | bge-reranker-base
                     | bge-reranker-v2-m3
RAG_RERANK_POOL      fused candidates passed to the reranker (default 20)
RAG_FIRST_STAGE_K    candidates from each of dense and BM25 (default 10)
RAG_RERANK_BATCH     reranker batch size (default 1). Every pair in a batch
                     is padded to the longest one, so with a 20-candidate
                     pool and a few 512-token chunks, batch 1 is ~2.7x faster
                     on CPU than sentence-transformers' default of 32.
RAG_CONTEXTUAL       1 (default) = BM25 and embeddings use LLM-written chunk
                     contexts (data/contexts/chunk_contexts.json); the dense
                     index has to be built the same way
RAG_RERANK_CONTEXT   1 = the reranker also sees the context, not just the
                     chunk text (only meaningful with RAG_CONTEXTUAL=1)
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VECTORSTORE_DIR = Path("data/vectorstore")
DEFAULT_CONTEXTS_PATH = Path("data/contexts/chunk_contexts.json")

EMBEDDERS = {
    "bge-small-en": "BAAI/bge-small-en",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "medcpt": "ncbi/MedCPT-Query-Encoder",  # paired with ncbi/MedCPT-Article-Encoder
}
RERANKERS = {
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "medcpt": "ncbi/MedCPT-Cross-Encoder",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def _choice(name: str, default: str, options: dict) -> str:
    value = os.getenv(name, default).strip()
    if value not in options:
        raise ValueError(f"{name}={value!r}; expected one of {sorted(options)}")
    return value


@dataclass(frozen=True)
class RetrievalSettings:
    embedder: str = "bge-small-en"
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR
    reranker: str = "ms-marco-minilm"
    rerank_pool: int = 20
    first_stage_k: int = 10
    rerank_batch_size: int = 1
    contextual: bool = True
    rerank_with_context: bool = False
    contexts_path: Path = DEFAULT_CONTEXTS_PATH

    def describe(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}


def load_settings() -> RetrievalSettings:
    return RetrievalSettings(
        embedder=_choice("RAG_EMBEDDER", "bge-small-en", EMBEDDERS),
        vectorstore_dir=Path(os.getenv("RAG_VECTORSTORE_DIR", str(DEFAULT_VECTORSTORE_DIR))),
        reranker=_choice("RAG_RERANKER", "ms-marco-minilm", RERANKERS),
        rerank_pool=int(os.getenv("RAG_RERANK_POOL", "20")),
        first_stage_k=int(os.getenv("RAG_FIRST_STAGE_K", "10")),
        rerank_batch_size=int(os.getenv("RAG_RERANK_BATCH", "1")),
        contextual=_flag("RAG_CONTEXTUAL", default=True),
        rerank_with_context=_flag("RAG_RERANK_CONTEXT"),
        contexts_path=Path(os.getenv("RAG_CONTEXTS_PATH", str(DEFAULT_CONTEXTS_PATH))),
    )


settings = load_settings()

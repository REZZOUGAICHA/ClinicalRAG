"""
CPU latency of retrieve() under the current RAG_* settings.

    RAG_RERANK_POOL=20 python scripts/bench_retrieval.py --config pool20

Runs every cached query rewrite from the retrieval eval through retrieve()
(top_k=3, with parent expansion, i.e. what a real request does) after a
warm-up, with torch limited to 2 threads to approximate a 2-vCPU Hugging
Face Space. Reports total, query-embedding and rerank time per query, plus
model load time and parameter counts. Appends to data/eval/latency.jsonl.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import RUNS_DIR  # noqa: E402


def params(model) -> int:
    from src.embeddings.embedder import MedCPTEncoder

    if isinstance(model, MedCPTEncoder):
        return params(model.query_model) + params(model.article_model)
    inner = getattr(model, "model", model)  # CrossEncoder wraps a HF model
    return sum(p.numel() for p in inner.parameters())


def summarize(ms: list[float]) -> dict:
    ms = sorted(ms)
    return {"median": round(statistics.median(ms), 1),
            "p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
            "mean": round(statistics.fmean(ms), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, help="only the first N queries (for very slow models)")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    from src.config import settings
    from src.embeddings import embedder
    from src.retrieval import retriever
    from src.vectorstore.store import get_collection

    t0 = time.perf_counter()
    emb_model = embedder.get_model()
    reranker = retriever.get_reranker()
    get_collection()
    retriever._get_bm25()
    load_s = time.perf_counter() - t0

    timings = {"embed": [], "rerank": []}

    def timed(name, fn):
        def wrapper(*a, **kw):
            start = time.perf_counter()
            out = fn(*a, **kw)
            timings[name].append((time.perf_counter() - start) * 1000)
            return out
        return wrapper

    retriever.embed_query = timed("embed", retriever.embed_query)
    reranker.predict = timed("rerank", reranker.predict)

    queries = list(json.loads((RUNS_DIR / "rewrite_cache.json").read_text(encoding="utf-8")).values())
    queries = queries[: args.limit] if args.limit else queries
    for q in queries[:3]:
        retriever.retrieve(q)
    timings = {"embed": [], "rerank": []}

    total = []
    for _ in range(args.repeats):
        for q in queries:
            start = time.perf_counter()
            retriever.retrieve(q)
            total.append((time.perf_counter() - start) * 1000)

    record = {
        "config": args.config,
        "settings": settings.describe(),
        "threads": args.threads,
        "n_queries": len(total),
        "load_s": round(load_s, 1),
        "embedder_params_m": round(params(emb_model) / 1e6, 1),
        "reranker_params_m": round(params(reranker) / 1e6, 1),
        "total_ms": summarize(total),
        "embed_ms": summarize(timings["embed"]),
        "rerank_ms": summarize(timings["rerank"]),
    }
    print(json.dumps(record, indent=2))
    with (RUNS_DIR / "latency.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

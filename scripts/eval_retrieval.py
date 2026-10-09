"""
Retrieval-only metrics over eval/questions.json. No LLM judge, so it's cheap
enough to rerun after every retrieval change.

    python scripts/eval_retrieval.py --config baseline
    python scripts/eval_retrieval.py --config baseline --no-rewrite

Per question (answerable items only):
  doc_hit@1, doc_hit@3  any gold source_file among the top 1 / 3 ranked
                        chunks, before parent expansion
  doc_recall@3          fraction of gold files among the top 3 (only differs
                        from doc_hit for multi-document questions)
  mrr                   1 / rank of the first gold-file chunk in the ranked
                        list, 0 if absent
  section_recall        fraction of gold (file, section) pairs present in the
                        context the LLM would actually get (expanded)
  context_chunks/chars  size of that context, a cost and precision proxy

The ranked list is at most RANK_DEPTH (10) long, and never longer than the
reranker shortlist (RAG_RERANK_POOL, 20 by default; 5 in the old baseline),
since retrieve() only ranks that many RRF candidates.

Query rewriting is a Groq call in the real pipeline. Rewrites are cached in
data/eval/rewrite_cache.json, so once the cache is warm this runs offline.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import (  # noqa: E402
    RUNS_DIR,
    append_summary,
    call_with_groq_retry,
    load_questions,
    mean,
    split_csv,
)

REWRITE_CACHE_PATH = RUNS_DIR / "rewrite_cache.json"
RANK_DEPTH = 10
METRICS = ("doc_hit@1", "doc_hit@3", "doc_recall@3", "mrr", "section_recall",
           "context_chunks", "context_chars")


class RewriteCache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.misses = 0

    def get(self, question: str) -> str:
        if question not in self.data:
            # imported lazily so --no-rewrite and warm-cache runs never need a key
            from src.generation.generator import rewrite_query

            self.data[question] = call_with_groq_retry(rewrite_query, question)
            self.misses += 1
            self.save()  # save each miss, a crash shouldn't cost the calls already made
        return self.data[question]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


def score_item(item: dict, ranked: list[dict], context: list[dict]) -> dict:
    gold_files = {g["source_file"] for g in item["gold"]}
    gold_pairs = {(g["source_file"], g["section"]) for g in item["gold"]}
    ranked_files = [c["source_file"] for c in ranked]

    first_hit = next((i for i, f in enumerate(ranked_files, 1) if f in gold_files), None)
    context_pairs = {(c["source_file"], c["section"]) for c in context}

    return {
        "doc_hit@1": float(bool(gold_files & set(ranked_files[:1]))),
        "doc_hit@3": float(bool(gold_files & set(ranked_files[:3]))),
        "doc_recall@3": len(gold_files & set(ranked_files[:3])) / len(gold_files),
        "mrr": 1 / first_hit if first_hit else 0.0,
        "section_recall": len(gold_pairs & context_pairs) / len(gold_pairs),
        "context_chunks": len(context),
        "context_chars": sum(len(c["text"]) for c in context),
    }


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for m in METRICS:
        out[m] = mean(r["metrics"][m] for r in rows)
    return out


def print_table(overall: dict, by_type: dict) -> None:
    cols = ("n",) + METRICS
    header = f"{'type':<14}" + "".join(f"{c:>15}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, agg in list(by_type.items()) + [("ALL", overall)]:
        cells = []
        for c in cols:
            v = agg[c]
            cells.append(f"{v:>15}" if c == "n" else f"{v:>15.0f}" if c == "context_chars"
                         else f"{v:>15.3f}")
        print(f"{name:<14}" + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="label for this run, used in output file names")
    parser.add_argument("--no-rewrite", action="store_true", help="search on the raw question text")
    parser.add_argument("--top-k", type=int, default=3, help="top_k passed to retrieve() for the context call")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--types", help="comma-separated question types to include")
    parser.add_argument("--ids", help="comma-separated question ids to include")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from src.config import settings as retrieval_settings
    from src.retrieval.retriever import retrieve

    items = [q for q in load_questions(split_csv(args.types), split_csv(args.ids), args.limit)
             if q["answerable"]]
    cache = None if args.no_rewrite else RewriteCache(REWRITE_CACHE_PATH)
    use_reranker = not args.no_reranker

    rows = []
    for item in items:
        query = item["question"] if cache is None else cache.get(item["question"])
        ranked = retrieve(query, top_k=RANK_DEPTH, use_reranker=use_reranker, expand_parents=False)
        context = retrieve(query, top_k=args.top_k, use_reranker=use_reranker)
        metrics = score_item(item, ranked, context)
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "search_query": query,
            "metrics": metrics,
            "ranked": [{"source_file": c["source_file"], "section": c["section"],
                        "reranker_score": c.get("reranker_score")} for c in ranked],
            "context": [{"source_file": c["source_file"], "section": c["section"]} for c in context],
        })
        print(f"  {item['id']:<36} hit@1={metrics['doc_hit@1']:.0f} hit@3={metrics['doc_hit@3']:.0f} "
              f"sec_recall={metrics['section_recall']:.2f} ctx={metrics['context_chunks']}")

    by_type_rows = defaultdict(list)
    for r in rows:
        by_type_rows[r["type"]].append(r)
    by_type = {t: aggregate(rs) for t, rs in sorted(by_type_rows.items())}
    overall = aggregate(rows)

    print()
    print_table(overall, by_type)

    settings = {"rewrite": cache is not None, "top_k": args.top_k, "reranker": use_reranker,
                "rank_depth_returned": max((len(r["ranked"]) for r in rows), default=0),
                "retrieval": retrieval_settings.describe()}
    out_path = RUNS_DIR / f"retrieval_{args.config}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": args.config, "settings": settings, "aggregate": overall,
                                    "by_type": by_type, "results": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    append_summary({"script": "eval_retrieval", "config": args.config, **settings,
                    "aggregate": overall, "by_type": by_type})
    if cache is not None and cache.misses:
        print(f"\n{cache.misses} rewrite(s) fetched from Groq and cached")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

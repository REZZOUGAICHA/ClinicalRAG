"""
Re-score stored eval answers with the current deterministic scorer, without
calling any API. Use it after changing the key-fact matcher, the refusal
detector or eval/questions.json, so old runs stay comparable with new ones.

    python scripts/rescore.py data/eval/eval_final_20261008T220748Z.json
    python scripts/rescore.py data/eval/checkpoint_baseline.json --config baseline

Takes a results file written by evaluate.py or a checkpoint (a plain list of
rows). RAGAS scores already in the rows are kept and averaged as they are.
Prints the by-type table and the overall numbers as stored next to the
rescored ones, and appends the result to eval/results/summary.jsonl as
"<config>-rescored".
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from eval_common import (  # noqa: E402
    QUESTIONS_PATH,
    aggregate,
    aggregate_by_type,
    append_summary,
    print_report,
    rescore_row,
)

COMPARE = ("key_fact_recall", "all_facts_rate", "false_refusal_rate", "refusal_accuracy")


def _cell(value) -> str:
    return f"{value:>10.4f}" if value is not None else f"{'-':>10}"


def load_rows(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    return data["results"], data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="eval_<config>_*.json or checkpoint_<config>.json")
    parser.add_argument("--config", help="label for the summary line (default: from the file)")
    parser.add_argument("--no-summary", action="store_true", help="print only, don't append to summary.jsonl")
    args = parser.parse_args()

    rows, meta = load_rows(args.path)
    config = args.config or meta.get("config") or args.path.stem.removeprefix("checkpoint_")
    items = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}

    unknown = [r.get("id") for r in rows if r.get("id") not in items or "answer" not in r]
    if unknown:
        print(f"Skipping {len(unknown)} row(s) with no matching question or answer: {unknown}")
    rescored = [rescore_row(r, items[r["id"]]) for r in rows if r.get("id") in items and "answer" in r]
    if not rescored:
        raise SystemExit("Nothing to score.")

    overall, by_type = aggregate_by_type(rescored)
    print(f"{config}: {len(rescored)} answers from {args.path}\n")
    print_report(overall, by_type)

    # Same rows, scores as stored, so the comparison is like for like.
    stored = aggregate([r for r in rows if r.get("id") in items and "scores" in r])
    print(f"\n{'ALL':<20}{'stored':>10}{'rescored':>10}")
    for m in COMPARE:
        print(f"{m:<20}{_cell(stored.get(m))}{_cell(overall.get(m))}")

    if not args.no_summary:
        append_summary({"script": "rescore", "config": f"{config}-rescored", "source": str(args.path),
                        "system": meta.get("system") or rescored[0].get("system"),
                        "aggregate": overall, "by_type": by_type})


if __name__ == "__main__":
    main()

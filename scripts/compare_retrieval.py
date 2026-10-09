"""
Per-question differences between two eval_retrieval.py runs.

    python scripts/compare_retrieval.py baseline pool20 [more configs...]

Means hide what moved: with 44 questions one flip is 0.023 on doc_hit, so
this lists every question whose hit@1, hit@3, MRR or section_recall changed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import RUNS_DIR  # noqa: E402

TRACKED = ("doc_hit@1", "doc_hit@3", "mrr", "section_recall")


def load(config: str) -> dict:
    data = json.loads((RUNS_DIR / f"retrieval_{config}.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["results"]}


def flips(base: dict, other: dict) -> list[str]:
    lines = []
    for qid, b in base.items():
        o = other.get(qid)
        if o is None:
            continue
        changed = [m for m in TRACKED if abs(b["metrics"][m] - o["metrics"][m]) > 1e-9]
        if changed:
            cells = ", ".join(f"{m} {b['metrics'][m]:.2f}->{o['metrics'][m]:.2f}" for m in changed)
            lines.append(f"  {qid:<36} ({b['type']}) {cells}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("others", nargs="+")
    args = parser.parse_args()

    base = load(args.base)
    for config in args.others:
        lines = flips(base, load(config))
        print(f"{args.base} -> {config}: {len(lines)} question(s) changed")
        print("\n".join(lines) if lines else "  (none)")
        print()


if __name__ == "__main__":
    main()

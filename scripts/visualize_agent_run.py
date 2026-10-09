"""
Draw the path one agent run took (the journey), as a Markdown page with a
Mermaid sequence diagram, the final answer and its citations.

Usage:
  python scripts/visualize_agent_run.py "Which patients have a creatinine above 2?"
  python scripts/visualize_agent_run.py "..." --model openai/gpt-oss-20b
  python scripts/visualize_agent_run.py --from-json data/eval/checkpoint_agent.json --id mh-bisoprolol

The first form runs the agent once (several Groq calls). --from-json makes
no API calls: it reads a stored result (an eval checkpoint row, a list of
them, or a saved ask_agent() dict) that has a "trace", and draws that.

Writes docs/runs/<slug>.md, or --out PATH.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.viz import (  # noqa: E402
    cited_sources,
    events_from_trace,
    record_run,
    slugify,
    to_markdown,
)
from src.tracing import setup_tracing  # noqa: E402

RUNS_DIR = ROOT / "docs" / "runs"


def _pick_row(data, row_id: str | None) -> dict:
    rows = data if isinstance(data, list) else [data]
    if row_id:
        for r in rows:
            if r.get("id") == row_id:
                return r
        raise SystemExit(f"No row with id {row_id!r}.")
    with_trace = [r for r in rows if r.get("trace")]
    if not with_trace:
        raise SystemExit("No row in that file has a trace. Was it written by --system agent?")
    return with_trace[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="question to run through the agent")
    ap.add_argument("--model", help="Groq model for the agent (sets AGENT_MODEL for this run)")
    ap.add_argument("--from-json", type=Path, help="draw a stored result instead of running")
    ap.add_argument("--id", help="with --from-json: which row to draw (default: first with a trace)")
    ap.add_argument("--out", type=Path, help="output file (default docs/runs/<slug>.md)")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if args.from_json:
        row = _pick_row(json.loads(args.from_json.read_text(encoding="utf-8")), args.id)
        question = row.get("question") or row.get("query") or "(unknown question)"
        result = {"answer": row.get("answer", ""), "route": row.get("route", "complex"),
                  # Eval rows keep what the answer step saw, not what it cited.
                  "sources": row.get("sources") or cited_sources(row.get("answer", "")),
                  "model": row.get("model")}
        events = events_from_trace(row)
        origin = (f"Redrawn from `{args.from_json.as_posix()}`"
                  + (f" (row `{row['id']}`)" if row.get("id") else "")
                  + " with no API calls, so there are no timings and each tool call is shown "
                    "as its own agent turn")
        slug = slugify(row.get("id") or question)
    elif args.question:
        if args.model:
            os.environ["AGENT_MODEL"] = args.model
        question = args.question
        setup_tracing()  # also send this run to Phoenix if CLINICALRAG_TRACING=phoenix
        events, result = record_run(question)
        origin = f"Live run, {stamp}"
        slug = slugify(question)
    else:
        ap.error("give a question, or --from-json PATH")

    out = args.out or RUNS_DIR / f"{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(question, events, result, origin=origin), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

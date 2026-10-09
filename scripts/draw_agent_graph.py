"""
Draw the agent's graph (the map: every node and edge, no run involved).

Usage: python scripts/draw_agent_graph.py

Writes docs/agent_graph.mmd (Mermaid text, which GitHub renders inside a
```mermaid block) and prints an ASCII version if grandalf is installed
(requirements-dev.txt). No API calls: the chat models are only called when
a node runs, and nothing runs here.

Not draw_mermaid_png(): by default it renders by sending the graph to the
mermaid.ink web service. The .mmd text renders on GitHub, in VS Code's
Mermaid preview and at mermaid.live without leaving the machine.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.graph import build_graph  # noqa: E402

OUT = ROOT / "docs" / "agent_graph.mmd"


def _never_called(*_):
    raise RuntimeError("draw_agent_graph should not call a model")


def main():
    graph = build_graph(chat=_never_called, simple_answer=_never_called).get_graph()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")

    try:
        print(graph.draw_ascii())
    except ImportError:
        print("(pip install grandalf for an ASCII drawing here)")


if __name__ == "__main__":
    main()

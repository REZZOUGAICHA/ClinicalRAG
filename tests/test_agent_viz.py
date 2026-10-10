"""
Tests for the agent visualisation helpers: node events in stream_agent, and
the run diagrams in src/agent/viz.py. No network, same stubs as test_agent.
"""

import re
import sys

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agent import graph as agent_graph
from src.agent import tools as agent_tools
from src.agent import viz

CHUNKS = [
    {"chunk_id": "r3-med", "source_file": "report_003_acute_mi.pdf", "section": "MEDICATIONS",
     "text": "Bisoprolol 5 mg OD.", "page_number": 1},
]


class StubChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def fixture_chunks(monkeypatch):
    monkeypatch.setattr(agent_tools, "_load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(agent_tools, "_retrieve", lambda query, k: CHUNKS[:k])


def _graph(answer_texts):
    read = AIMessage("", tool_calls=[{"name": "read_section", "id": "c1",
                                      "args": {"source_files": ["report_003"], "section": "MEDICATIONS"}}])
    models = {
        "router": StubChat(messages=iter([AIMessage("complex")])),
        "agent": StubChat(messages=iter([read, AIMessage("DONE")])),
        "answer": StubChat(messages=iter([AIMessage(t) for t in answer_texts])),
    }
    return agent_graph.build_graph(chat=models.__getitem__)


GOOD = "Bisoprolol 5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]"
BAD = "Bisoprolol 5 mg [Source: report_009.pdf | MEDICATIONS]"


def test_stream_reports_each_node_as_it_starts():
    events = list(agent_graph.stream_agent("q", graph=_graph([GOOD])))
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes == ["route", "agent", "tools", "agent", "answer", "ground"]
    # every progress event says which node it came from
    assert all("node" in e for e in events if e["type"] in ("route", "step"))
    assert events[-1]["type"] == "result"


def test_record_run_matches_the_path_taken():
    events, result = viz.record_run("q", graph=_graph([BAD, GOOD]))
    kinds = [e["kind"] for e in events]
    assert kinds == ["route", "plan", "tools", "done", "answer", "ground", "answer", "ground"]
    assert events[5]["ok"] is False and events[7]["ok"] is True
    assert all(e["secs"] >= 0 for e in events if e["kind"] != "note")

    mermaid = viz.to_mermaid(events, result)
    assert "A->>T: read_section(source_files=[report_003], section=\"MEDICATIONS\")" in mermaid
    assert "W->>G: 2nd draft" in mermaid
    assert "participant S" not in mermaid  # simple route wasn't used


def test_events_from_stored_trace_without_api_calls():
    row = {"id": "mh-x", "question": "q", "answer": GOOD, "route": "complex", "trace": [
        {"tool": "list_documents", "args": {}, "step": "Listing all documents", "summary": "12 passage(s)"},
        {"tool": "read_section", "args": {"source_files": ["report_003"], "section": "LABS"},
         "step": "Reading LABS", "summary": "1 passage(s): report_003 LABS"},
        {"tool": None, "args": {}, "step": "Citation check failed, rewriting", "summary": "bad cite"},
    ]}
    kinds = [e["kind"] for e in viz.events_from_trace(row)]
    assert kinds == ["route", "plan", "tools", "plan", "tools", "done",
                     "answer", "ground", "answer", "ground"]


def test_simple_route_from_trace():
    events = viz.events_from_trace({"route": "simple", "trace": []})
    assert [e["kind"] for e in events] == ["route", "simple"]
    assert "R->>S: simple" in viz.to_mermaid(events, {"sources": []})


def test_labels_cannot_break_the_diagram():
    label = viz._label("a; b # c <script>" + "x" * 200)
    assert label.startswith("a#59; b #35; c #lt;script#gt;")
    assert ";" not in re.sub(r"#\w+;", "", label) and "<" not in label
    assert len(label) < 120


def test_tracing_is_off_and_imports_nothing_by_default(monkeypatch):
    from src import tracing

    monkeypatch.delenv("CLINICALRAG_TRACING", raising=False)
    monkeypatch.setattr(tracing, "_enabled", None)
    before = set(sys.modules)
    assert tracing.setup_tracing() is None
    assert not {m for m in set(sys.modules) - before if m.startswith(("phoenix", "openinference"))}

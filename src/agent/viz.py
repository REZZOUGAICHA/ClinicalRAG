"""
Turn one agent run into a Mermaid sequence diagram, for docs/runs/.

Two ways in:
- record_run(question) runs the graph once and notes every node as it
  finishes: what the agent asked for, what each tool returned, how long
  each step took.
- events_from_trace(row) rebuilds the same events from a stored result (an
  eval checkpoint row or an ask_agent() dict), with no API calls. The stored
  trace only keeps tool calls and notes, so this version can't tell which
  calls the agent made in the same turn, and has no timings.

A sequence diagram rather than a flowchart: one run is a single path through
the graph, so a flowchart of it is just a column of boxes. A sequence
diagram keeps who did what (agent, tools, answer, ground) in separate
columns and reads top to bottom in time order, so the agent <-> tools loop
and a citation retry are easy to count.
"""

import re
import time

# Mermaid participant ids and the names shown at the top of each column.
PARTICIPANTS = [
    ("U", "User"),
    ("R", "route"),
    ("S", "simple pipeline"),
    ("A", "agent"),
    ("T", "tools"),
    ("W", "answer"),
    ("G", "ground"),
]

MAX_LABEL = 90
_ENTITIES = {"#": "#35;", ";": "#59;", "<": "#lt;", ">": "#gt;"}


def _label(text: str, limit: int = MAX_LABEL) -> str:
    """Make text safe inside a Mermaid message: one line, no ; or # or <>."""
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    # Mermaid reads ; as a line break and # as the start of an entity code.
    # One pass, so the ; in "#35;" isn't escaped again.
    return re.sub(r"[#;<>]", lambda m: _ENTITIES[m.group()], text)


def _args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, list):
            v = "[" + ", ".join(re.sub(r"^([a-z]+_\d+).*", r"\1", str(x)) for x in v) + "]"
        elif isinstance(v, str):
            v = f'"{v}"'
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def call_text(tool: str, args: dict) -> str:
    return f"{tool}({_args(args)})"


def _secs(event: dict) -> str:
    s = event.get("secs")
    return f" ({s:.1f} s)" if s is not None else ""


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
# A run is a list of small dicts, one per thing that happened:
#   {"kind": "route", "route": "complex"}
#   {"kind": "plan", "calls": [{"tool", "args"}]}     agent asked for tools
#   {"kind": "done"}                                  agent said DONE
#   {"kind": "tools", "results": [{"tool", "args", "summary"}]}
#   {"kind": "answer", "attempt": 1}
#   {"kind": "ground", "ok": False, "problem": "..."}
#   {"kind": "simple"}
#   {"kind": "note", "on": "A", "text": "..."}
# Live events also carry "secs", the time the node took.


def _notes(trace: list[dict], on: str) -> list[dict]:
    return [{"kind": "note", "on": on, "text": f"{t['step']}: {t['summary']}"}
            for t in trace if t.get("tool") is None]


def record_run(question: str, graph=None) -> tuple[list[dict], dict]:
    """Run the agent once. Returns (events, ask_agent-shaped result)."""
    from src.agent.graph import _default_graph, initial_state, recursion_limit, to_result

    graph = graph or _default_graph()
    events, final = [], None
    last = time.perf_counter()
    for mode, data in graph.stream(initial_state(question), {"recursion_limit": recursion_limit()},
                                   stream_mode=["updates", "values"]):
        if mode == "values":
            final = data
            continue
        now = time.perf_counter()
        # Nodes run one at a time, so the gap between two updates is the
        # time the second node took.
        secs, last = now - last, now
        for node, update in data.items():
            events.extend(_events_for(node, update or {}, secs))
    return events, to_result(final)


def _events_for(node: str, update: dict, secs: float) -> list[dict]:
    trace = update.get("trace", [])
    if node == "route":
        return [{"kind": "route", "route": update["route"], "secs": secs}]
    if node == "simple":
        return [{"kind": "simple", "secs": secs}] + _notes(trace, "S")
    if node == "agent":
        msg = update.get("messages", [None])[-1]
        calls = getattr(msg, "tool_calls", None) or []
        head = ({"kind": "plan", "calls": [{"tool": c["name"], "args": c.get("args") or {}}
                                           for c in calls], "secs": secs}
                if calls else {"kind": "done", "secs": secs})
        return [head] + _notes(trace, "A")
    if node == "tools":
        results = [{"tool": t["tool"], "args": t["args"], "summary": t["summary"]}
                   for t in trace if t.get("tool")]
        return [{"kind": "tools", "results": results, "secs": secs}]
    if node == "answer":
        return [{"kind": "answer", "attempt": update.get("attempts", 1), "secs": secs}] + _notes(trace, "W")
    if node == "ground":
        if update.get("feedback"):
            problem = trace[0]["summary"] if trace else ""
            return [{"kind": "ground", "ok": False, "problem": problem, "secs": secs}]
        return [{"kind": "ground", "ok": True, "secs": secs}] + _notes(trace, "G")
    return []


def events_from_trace(row: dict) -> list[dict]:
    """Rebuild events from a stored trace (eval checkpoint row or result dict)."""
    trace = row.get("trace") or []
    route = row.get("route") or ("complex" if any(t.get("tool") for t in trace) else None)
    events = [{"kind": "route", "route": route or "complex"}]
    if route == "simple":
        return events + [{"kind": "simple"}] + _notes(trace, "S")

    state = {"attempt": 0, "drafting": False}

    def draft():
        # The answer node runs once per draft; the agent says DONE only once.
        if state["drafting"]:
            return
        if state["attempt"] == 0:
            events.append({"kind": "done"})
        state["attempt"] += 1
        state["drafting"] = True
        events.append({"kind": "answer", "attempt": state["attempt"]})

    for t in trace:
        step = t.get("step", "")
        if t.get("tool"):
            # One call per turn: the stored trace doesn't keep the grouping.
            call = {"tool": t["tool"], "args": t.get("args") or {}}
            events.append({"kind": "plan", "calls": [call]})
            events.append({"kind": "tools", "results": [{**call, "summary": t.get("summary", "")}]})
        elif step.startswith("Left out passages"):
            draft()
            events += _notes([t], "W")
        elif step.startswith("Citation check failed"):
            draft()
            events.append({"kind": "ground", "ok": False, "problem": t.get("summary", "")})
            state["drafting"] = False
        elif step.startswith("Removed unverifiable"):
            draft()
            return events + [{"kind": "ground", "ok": True}] + _notes([t], "G")
        else:
            events += _notes([t], "A")
    draft()
    return events + [{"kind": "ground", "ok": True}]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def to_mermaid(events: list[dict], result: dict) -> str:
    lines, used = [], set()

    def say(src: str, arrow: str, dst: str, text: str) -> None:
        lines.append(f"{src}{arrow}{dst}: {text}")
        used.update((src, dst))

    n_sources = len(result.get("sources") or [])
    prev = None
    for ev in events:
        kind = ev["kind"]
        if kind == "route":
            say("U", "->>", "R", "question")
            say("R", "->>", "S" if ev["route"] == "simple" else "A", f"{ev['route']}{_secs(ev)}")
        elif kind == "simple":
            say("S", "-->>", "U", f"answer, {n_sources} citation(s){_secs(ev)}")
        elif kind == "plan":
            for i, c in enumerate(ev["calls"]):
                # The time is the planning call's, so it goes on its first arrow.
                t = _secs(ev) if i == 0 else ""
                say("A", "->>", "T", _label(call_text(c["tool"], c["args"])) + t)
        elif kind == "tools":
            for i, r in enumerate(ev["results"]):
                last = i == len(ev["results"]) - 1
                say("T", "-->>", "A", _label(r["summary"]) + (_secs(ev) if last else ""))
        elif kind == "done":
            say("A", "->>", "W", f"DONE{_secs(ev)}")
        elif kind == "answer":
            if prev == "tools":
                # tools went straight to answer: the call budget ran out.
                say("T", "->>", "W", "tool-call limit reached")
            nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(ev["attempt"], f"{ev['attempt']}th")
            say("W", "->>", "G", f"{nth} draft{_secs(ev)}")
        elif kind == "ground":
            if ev["ok"]:
                say("G", "-->>", "U", f"answer, {n_sources} citation(s){_secs(ev)}")
            else:
                say("G", "->>", "W", _label("rewrite: " + ev.get("problem", "")) + _secs(ev))
        elif kind == "note":
            lines.append(f"Note over {ev['on']}: {_label(ev['text'])}")
            used.add(ev["on"])
        if kind != "note":
            prev = kind

    head = ["sequenceDiagram", "    autonumber"]
    head += [f"    {'actor' if pid == 'U' else 'participant'} {pid} as {name}"
             for pid, name in PARTICIPANTS if pid in used]
    return "\n".join(head + ["    " + line for line in lines])


_CITATION = re.compile(r"\[Source:\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]")


def cited_sources(answer: str) -> list[dict]:
    """The [Source: file | SECTION] citations in an answer, in order, once each."""
    pairs = dict.fromkeys(_CITATION.findall(answer or ""))
    return [{"source_file": f, "section": sec} for f, sec in pairs]


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "run"


def to_markdown(question: str, events: list[dict], result: dict, *, origin: str,
                model: str | None = None) -> str:
    tool_calls = sum(len(e["calls"]) for e in events if e["kind"] == "plan")
    timed = [e["secs"] for e in events if e.get("secs") is not None]
    meta = [f"route: **{result.get('route', '?')}**", f"tool calls: **{tool_calls}**"]
    if model or result.get("model"):
        meta.append(f"model: `{model or result.get('model')}`")
    if timed:
        meta.append(f"total: **{sum(timed):.1f} s**")

    steps = []
    for e in events:
        if e["kind"] == "plan":
            steps += [f"- agent asks for `{call_text(c['tool'], c['args'])}`" for c in e["calls"]]
        elif e["kind"] == "tools":
            steps += [f"- `{r['tool']}` returned: {r['summary']}" for r in e["results"]]
        elif e["kind"] == "ground" and not e["ok"]:
            steps.append(f"- citation check failed, rewrite asked: {e.get('problem', '')}")
        elif e["kind"] == "note":
            steps.append(f"- note: {e['text']}")

    sources = result.get("sources") or []
    cites = [f"{i}. `{s['source_file']}` | {s['section']}"
             + (f" (page {s['page_number']})" if s.get("page_number") else "")
             for i, s in enumerate(sources, 1)] or ["(none)"]
    answer = "\n".join("> " + line if line else ">" for line in result.get("answer", "").splitlines())

    out = [
        f"# Agent run: {question}",
        "",
        f"{origin}. " + ", ".join(meta) + ".",
        "",
        "Arrows are numbered in time order. `A->>T` is the agent asking for a tool call, "
        "`T-->>A` is what came back. Times in brackets are how long that step took, "
        "including any wait for a Groq rate limit and, on the first search, loading the "
        "embedding models.",
        "",
        "```mermaid",
        to_mermaid(events, result),
        "```",
        "",
        "## Tool calls in full",
        "",
        *(steps or ["(no tool calls)"]),
        "",
        "## Final answer",
        "",
        answer or "> (empty)",
        "",
        "## Citations",
        "",
        *cites,
        "",
    ]
    return "\n".join(out)

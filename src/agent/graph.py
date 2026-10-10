"""
Agentic mode: a LangGraph state machine for questions one top-3 retrieval
can't answer (comparisons across patients, lists over every report, "find
the patient with X, then look up Y").

    START -> route --simple--> simple (src.pipeline.ask + citation check) -> END
               |
               +--complex--> agent <-> tools      (at most MAX_TOOL_CALLS)
                               |
                               v
                            answer -> ground --bad or missing citations, 1st try--> answer
                                         |
                                         v
                                        END

The agent turn only plans and gathers passages. The answer is written by a
separate call that sees the question and the gathered passages and nothing
else, in the same passage/citation format as the pipeline, so the frontend
parses it the same way and the grounding check has a closed set to test.
The simple route's answer goes through the same citation check.
"""

import logging
import operator
import os
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.grounding import REFUSAL, check_citations, is_refusal
from src.agent.llm import agent_model, get_chat_model, invoke_with_retry
from src.agent.tools import TOOLS, TOOLS_BY_NAME, step_label
from src.generation.generator import _BRACKETS

MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "6"))

# Passage text sent to the answer step. Lab and medication sections are
# 300-550 chars, so this fits ~30 of them while staying well under the
# 8000 tokens/minute free-tier budget together with the planning turns.
ANSWER_CONTEXT_CHARS = 16000

CHUNK_KEYS = ("chunk_id", "text", "source_file", "section", "page_number")

UNVERIFIED_NOTE = ("Note: this answer could not be matched to a cited passage, "
                   "so treat it as unverified.\n\n")
REMOVED_NOTE = ("Note: some citations in this answer did not match a retrieved passage "
                "and were removed. Statements that relied on them are unverified.\n\n")

log = logging.getLogger(__name__)

ROUTER_PROMPT = """Classify a question for a clinical document search system.

simple: asks about one patient or one document, and one search would find the
answer. Greetings and small talk are also simple.
  e.g. "What is the HbA1c of the diabetic patient?"
  e.g. "What medications was the stroke patient discharged with?"
complex: needs several documents or several steps: comparing patients, listing
or counting across patients or documents ("which patients", "every", "all",
"compare"), or identifying a patient from one fact and then looking up another.
  e.g. "Which patients are on bisoprolol, and at what dose?"
  e.g. "The patient whose creatinine rose to 3.2: what dialysis options were discussed?"

Reply with one word: simple or complex."""

AGENT_PROMPT = """You gather evidence for a clinical document assistant. A later
step writes the answer using only the passages your tools return, so anything
you don't retrieve can't be used.

The collection has synthetic patient reports (report_0XX_*.pdf, patient IDs
SYN-2024-0XX, all with the same sections) and MTSamples transcriptions
(mts_0XX_*.pdf) with varied sections. list_documents shows what is indexed.

- For questions about several or all patients, call list_documents first, then
  read_section on the relevant documents. One read_section call can read the
  same section from many documents.
- For specific facts, use search_documents with short queries, one entity per
  search. If a search finds the right document but not the section with the
  fact, call read_section on that document instead of searching again.
- A fact may sit in an unexpected section (a drug in TREATMENT PLAN rather than
  MEDICATIONS); check both when unsure.
- You have at most {max_calls} tool calls in total. Prefer few, broad calls.
- When you have what you need, stop calling tools and reply with one word: DONE."""

ANSWER_PROMPT = """You are a clinical document assistant. Answer the question using ONLY
the passages below.

RULES:
- Do not use outside medical knowledge, only what the passages state.
- Cite each fact right after it in this format: [Source: filename | SECTION]
  using the filename and section from the passage's Source line.
- For comparisons or lists, cover each patient or document the passages
  support, and cite each one separately.
- If the passages answer only part of the question, answer that part, then say
  briefly what is missing and cite the passage you checked for it, e.g.
  "No dose is given [Source: filename | MEDICATIONS]".
- If no passage answers any part of the question, respond with exactly:
  "{refusal}"
- Never guess. Never invent lab values, medications, or diagnoses.

PASSAGES:
{passages}
QUESTION: {question}

ANSWER:"""


def _merge_chunks(existing: list[dict], new: list[dict]) -> list[dict]:
    # "via" is the tool that returned the chunk. One first seen through
    # list_documents and later read on purpose counts as read.
    merged = {c["chunk_id"]: c for c in existing}
    for c in new:
        old = merged.get(c["chunk_id"])
        if old is None:
            merged[c["chunk_id"]] = {**{k: c.get(k) for k in CHUNK_KEYS}, "via": c.get("via")}
        elif old.get("via") == "list_documents" and c.get("via") != "list_documents":
            merged[c["chunk_id"]] = {**old, "via": c.get("via")}
    return list(merged.values())


class AgentState(TypedDict, total=False):
    question: str
    route: str
    messages: Annotated[list[AnyMessage], add_messages]
    chunks: Annotated[list[dict], _merge_chunks]
    trace: Annotated[list[dict], operator.add]
    tool_calls_used: int
    context: list[dict]
    draft: str
    attempts: int
    feedback: str
    answer: str
    sources: list[dict]
    result: dict


def _short(source_file: str) -> str:
    return re.sub(r"^([a-z]+_\d+).*", r"\1", source_file)


def _summarize(content: str, chunks: list[dict]) -> str:
    if not chunks:
        return " ".join(str(content).split())[:120]
    seen = list(dict.fromkeys(f"{_short(c['source_file'])} {c['section']}" for c in chunks))
    more = f" (+{len(seen) - 5} more)" if len(seen) > 5 else ""
    return f"{len(chunks)} passage(s): {', '.join(seen[:5])}{more}"


def _passage(c: dict) -> str:
    # No passage numbers: given "[3] Source: ...", gpt-oss-20b cited
    # "[3 | SECTION]" instead of the filename, which nothing can resolve.
    return f"Source: {c['source_file']} | Section: {c['section']}\n{c['text']}\n"


def _format_passages(chunks: list[dict]) -> str:
    return "\n".join(_passage(c) for c in chunks)


def _select_context(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Pick the passages the answer step sees. Returns (sent, dropped).

    Passages the agent searched for or read go first. list_documents'
    PATIENT INFORMATION/DIAGNOSIS sections fill the space that's left, minus
    any section that was also read explicitly. A passage that doesn't fit is
    skipped rather than ending the list, so one long section can't push out
    every short one after it.
    """
    read = [c for c in chunks if c.get("via") != "list_documents"]
    read_keys = {(c["source_file"], c["section"]) for c in read}
    listed = [c for c in chunks if c.get("via") == "list_documents"
              and (c["source_file"], c["section"]) not in read_keys]
    sent, dropped, used = [], [], 0
    for c in read + listed:
        size = len(_passage(c))
        if used + size > ANSWER_CONTEXT_CHARS:
            dropped.append(c)
            continue
        sent.append(c)
        used += size
    return sent, dropped


def _public(c: dict) -> dict:
    return {k: c.get(k) for k in CHUNK_KEYS}


def _source_entry(c: dict) -> dict:
    return {
        "source_file": c["source_file"],
        "section": c["section"],
        "page_number": c.get("page_number") or 1,
        "text": c["text"],
    }


def _pairs(pairs: list[tuple[str, str]]) -> str:
    return "; ".join(f"{f} | {s}" for f, s in pairs)


def _note(step: str, summary: str) -> dict:
    return {"tool": None, "args": {}, "step": step, "summary": summary}


def build_graph(chat=get_chat_model, simple_answer=None, max_tool_calls: int = MAX_TOOL_CALLS):
    """
    chat(purpose) -> chat model for "router", "agent" or "answer".
    simple_answer(question) -> pipeline-shaped dict; defaults to src.pipeline.ask.
    Both are injectable so tests can run the graph without network access.
    """
    if simple_answer is None:
        def simple_answer(question: str) -> dict:
            from src.pipeline import ask
            return ask(question)

    def route(state: AgentState) -> dict:
        # The API's "auto" mode classifies first (so simple questions can use
        # the streaming pipeline) and passes the decision in; don't ask twice.
        if state.get("route"):
            return {"route": state["route"]}
        return {"route": classify_question(state["question"], chat)}

    def simple(state: AgentState) -> dict:
        result = dict(simple_answer(state["question"]))
        check = check_citations(result["answer"], result.get("chunks", []))
        answer = check["answer"]
        if check["unknown"]:
            # Same notes as the complex route. An answer with no citations at
            # all is left alone: that's how the pipeline answers small talk.
            answer = (REMOVED_NOTE if check["cited"] else UNVERIFIED_NOTE) + answer
        result.update(answer=answer, sources=[_source_entry(c) for c in check["cited"]],
                      chunks_used=len(check["cited"]))
        update = {"result": result}
        if check["unknown"]:
            update["trace"] = [_note("Removed unverifiable citations", _pairs(check["unknown"]))]
        return update

    def agent(state: AgentState) -> dict:
        llm = chat("agent").bind_tools(TOOLS)
        messages = [SystemMessage(AGENT_PROMPT.format(max_calls=max_tool_calls)), *state["messages"]]
        # Groq answers 400 "tool_use_failed" when gpt-oss emits a malformed
        # call (seen: a tool named "read_section<|channel|>commentary"). That's
        # sampling noise, so try once more, then answer from what's been
        # gathered rather than fail the whole question.
        for _ in range(2):
            try:
                return {"messages": [invoke_with_retry(llm, messages)]}
            except Exception as e:
                if getattr(e, "status_code", None) != 400:
                    raise
                log.warning("agent planning call failed: %s", e)
        # The trace is shown in the browser; Groq's error text stays in the log.
        note = _note("Planner error, answering from what was found",
                     "The model sent an invalid tool call twice.")
        return {"messages": [AIMessage("DONE")], "trace": [note]}

    def tools(state: AgentState) -> dict:
        calls = state["messages"][-1].tool_calls
        used = state.get("tool_calls_used", 0)
        budget = max_tool_calls - used
        replies, gathered, trace = [], [], []
        for i, call in enumerate(calls):
            name, args = call["name"], call.get("args") or {}
            if i >= budget:
                # Every tool_call id needs a reply or the next request is invalid.
                replies.append(ToolMessage("Not run: tool-call limit reached.",
                                           tool_call_id=call["id"], name=name))
                continue
            tool = TOOLS_BY_NAME.get(name)
            chunks = []
            if tool is None:
                msg = ToolMessage(f"Unknown tool {name!r}.", tool_call_id=call["id"], name=name)
            else:
                try:
                    msg = tool.invoke({**call, "type": "tool_call"})
                    chunks = msg.artifact or []
                except Exception:
                    log.exception("agent tool %s failed", name)
                    msg = ToolMessage("Tool error, try a different call.",
                                      tool_call_id=call["id"], name=name)
            replies.append(msg)
            gathered.extend({**c, "via": name} for c in chunks)
            trace.append({"tool": name, "args": args, "step": step_label(name, args),
                          "summary": _summarize(msg.content, chunks)})
        return {
            "messages": replies,
            "chunks": gathered,
            "trace": trace,
            "tool_calls_used": used + min(len(calls), budget),
        }

    def answer(state: AgentState) -> dict:
        attempts = state.get("attempts", 0) + 1
        context, dropped = _select_context(state.get("chunks", []))
        if not context:
            # Nothing was retrieved, so there is nothing an answer could cite.
            return {"draft": REFUSAL, "attempts": attempts, "context": []}
        prompt = ANSWER_PROMPT.format(refusal=REFUSAL, passages=_format_passages(context),
                                      question=state["question"])
        if state.get("feedback"):
            prompt += f"\n\n{state['feedback']}\n\nCORRECTED ANSWER:"
        reply = invoke_with_retry(chat("answer"), [HumanMessage(prompt)])
        update = {"draft": str(reply.content).translate(_BRACKETS).strip(), "attempts": attempts,
                  "context": context}
        if dropped and attempts == 1:
            update["trace"] = [_note("Left out passages over the answer size limit",
                                     _summarize("", dropped))]
        return update

    def ground(state: AgentState) -> dict:
        # Citations are checked against what the answer step was shown, not
        # everything gathered: a dropped passage can't back a claim.
        draft, chunks = state["draft"], state.get("context", [])
        check = check_citations(draft, chunks)
        uncited = check["n_cited"] == 0 and not is_refusal(draft)
        if (check["unknown"] or uncited) and state.get("attempts", 0) < 2:
            if check["unknown"]:
                problem = f"These citations match no passage above: {_pairs(check['unknown'])}."
            else:
                problem = "That answer has no citations."
            feedback = (f"Your previous answer was:\n{draft}\n\n{problem} Rewrite it citing only "
                        f"the filename and section of the passages that state each fact.")
            return {"feedback": feedback, "trace": [_note("Citation check failed, rewriting", problem)]}
        final = check["answer"] or REFUSAL
        if not check["cited"] and not is_refusal(final):
            # Out of retries and still nothing verifiable. Keep the text (it may
            # be right) but don't let it pass as a grounded clinical answer.
            final = UNVERIFIED_NOTE + final
        elif check["cited"] and check["unknown"]:
            final = REMOVED_NOTE + final
        update = {"answer": final, "sources": [_source_entry(c) for c in check["cited"]],
                  "feedback": ""}
        if check["unknown"]:
            update["trace"] = [_note("Removed unverifiable citations", _pairs(check["unknown"]))]
        return update

    def after_route(state: AgentState) -> str:
        return "simple" if state["route"] == "simple" else "agent"

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if wants_tools and state.get("tool_calls_used", 0) < max_tool_calls else "answer"

    def after_tools(state: AgentState) -> str:
        # At the cap there's no point asking the model for another call.
        return "answer" if state.get("tool_calls_used", 0) >= max_tool_calls else "agent"

    def after_ground(state: AgentState) -> str:
        return "answer" if state.get("feedback") else END

    g = StateGraph(AgentState)
    for name, fn in [("route", route), ("simple", simple), ("agent", agent), ("tools", tools),
                     ("answer", answer), ("ground", ground)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["simple", "agent"])
    g.add_edge("simple", END)
    g.add_conditional_edges("agent", after_agent, ["tools", "answer"])
    g.add_conditional_edges("tools", after_tools, ["agent", "answer"])
    g.add_edge("answer", "ground")
    g.add_conditional_edges("ground", after_ground, ["answer", END])
    return g.compile()


def recursion_limit(max_tool_calls: int = MAX_TOOL_CALLS) -> int:
    # route + (agent, tools) per call + a final agent turn + two answer/ground
    # rounds, with slack. Hitting it means a bug, not a long question.
    return 2 * max_tool_calls + 10


def classify_question(question: str, chat=get_chat_model) -> str:
    """"simple" (one fact from one document) or "complex", via the router prompt."""
    messages = [SystemMessage(ROUTER_PROMPT), HumanMessage(question)]
    try:
        reply = invoke_with_retry(chat("router"), messages)
        match = re.search(r"\b(simple|complex)\b", str(reply.content).lower())
        return match.group(1) if match else "complex"
    except Exception:
        # A wrong "complex" costs a few extra calls; a wrong "simple" can
        # cost the answer. So any failure here goes the safe way.
        return "complex"


def initial_state(question: str, route: str | None = None) -> AgentState:
    state = {"question": question, "messages": [HumanMessage(question)], "chunks": [],
             "trace": [], "tool_calls_used": 0, "attempts": 0}
    if route:
        state["route"] = route
    return state


def to_result(state: AgentState, model: str | None = None) -> dict:
    """Shape the final state like src.pipeline.ask's return, plus route and trace."""
    if state.get("route") == "simple":
        return {**state["result"], "route": "simple", "trace": state.get("trace", [])}
    sources = state.get("sources", [])
    return {
        "query": state["question"],
        "answer": state.get("answer", REFUSAL),
        "sources": sources,
        # What the answer model was shown, so RAGAS contexts compare fairly
        # with the pipeline's. Everything the tools returned is kept apart.
        "chunks": [_public(c) for c in state.get("context", [])],
        "gathered_chunks": [_public(c) for c in state.get("chunks", [])],
        "model": model or agent_model(),
        "chunks_used": len(sources),
        "route": "complex",
        "trace": state.get("trace", []),
    }


_graph = None


def _default_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask_agent(question: str, graph=None) -> dict:
    graph = graph or _default_graph()
    state = graph.invoke(initial_state(question), {"recursion_limit": recursion_limit()})
    return to_result(state)


def stream_agent(question: str, graph=None, max_tool_calls: int = MAX_TOOL_CALLS, route: str | None = None):
    """
    Yields progress events while the graph runs, then the final result:
      {"type": "node",   "node": "agent"}             (a node is starting)
      {"type": "route",  "route": "simple" | "complex", "node": "route"}
      {"type": "step",   "content": "Searching documents for ...", "node": "tools"}
      {"type": "result", "result": <ask_agent-shaped dict>}
    "node" is the graph node the event came from; the UI uses it to light up
    its copy of the graph.
    """
    graph = graph or _default_graph()
    final = None
    used = 0
    for mode, data in graph.stream(initial_state(question, route), {"recursion_limit": recursion_limit()},
                                   stream_mode=["updates", "values", "tasks"]):
        if mode == "values":
            final = data
            continue
        if mode == "tasks":
            # "tasks" sends one event when a node starts (has "input") and one
            # when it ends; only the start is new information.
            if "input" in data:
                yield {"type": "node", "node": data["name"]}
            continue
        for node, update in data.items():
            update = update or {}
            if node == "route":
                yield {"type": "route", "route": update["route"], "node": node}
            elif node == "tools":
                for t in update.get("trace", []):
                    yield {"type": "step", "content": t["step"], "node": node}
                used = update.get("tool_calls_used", used)
                if used >= max_tool_calls:
                    yield {"type": "step", "content": "Tool-call limit reached, writing the answer",
                           "node": node}
            elif node == "agent":
                for t in update.get("trace", []):
                    yield {"type": "step", "content": t["step"], "node": node}
                last = update.get("messages", [None])[-1]
                if not getattr(last, "tool_calls", None):
                    yield {"type": "step", "content": "Writing the answer", "node": node}
            elif node in ("simple", "answer", "ground"):
                for t in update.get("trace", []):
                    yield {"type": "step", "content": t["step"], "node": node}
    yield {"type": "result", "result": to_result(final)}

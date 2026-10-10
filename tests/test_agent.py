"""
Agent graph tests. No network: chat models are langchain_core's fake model,
the pipeline is a stub, and the tools read a small in-memory chunk list.
"""

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agent import graph as agent_graph
from src.agent import tools as agent_tools
from src.agent.grounding import REFUSAL, check_citations, is_refusal
from src.agent.llm import invoke_with_retry

CHUNKS = [
    {"chunk_id": "r3-med", "source_file": "report_003_acute_mi.pdf", "section": "MEDICATIONS",
     "text": "Bisoprolol 5 mg OD. Atorvastatin 80 mg.", "page_number": 1},
    {"chunk_id": "r6-plan", "source_file": "report_006_stroke.pdf", "section": "TREATMENT PLAN",
     "text": "Bisoprolol 2.5 mg OD for AF rate control.", "page_number": 2},
    {"chunk_id": "r6-info", "source_file": "report_006_stroke.pdf", "section": "PATIENT INFORMATION",
     "text": "Patient ID: SYN-2024-006\nSex: Female | Age: 63", "page_number": 1},
]


class StubChat(GenericFakeChatModel):
    """Fake chat model that also accepts bind_tools, like a real one."""

    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        return super().invoke(input, config, **kwargs)


def _replies(*messages):
    return StubChat(messages=iter(messages))


def _repeat(factory):
    return StubChat(messages=(factory(i) for i in itertools.count()))


@pytest.fixture(autouse=True)
def fixture_chunks(monkeypatch):
    monkeypatch.setattr(agent_tools, "_load_chunks", lambda: CHUNKS)
    # search_documents would load the embedder and ChromaDB; fake it.
    monkeypatch.setattr(agent_tools, "_retrieve", lambda query, k: CHUNKS[:k])


def _run(graph, question="q"):
    state = graph.invoke(agent_graph.initial_state(question),
                         {"recursion_limit": agent_graph.recursion_limit()})
    return agent_graph.to_result(state, model="stub")


def test_simple_route_uses_pipeline_and_nothing_else():
    pipeline_calls = []

    def fake_pipeline(question):
        pipeline_calls.append(question)
        return {"query": question, "answer": "9.2% [Source: report_002.pdf | LABORATORY RESULTS]",
                "sources": [{"source_file": "report_002.pdf", "section": "LABORATORY RESULTS",
                             "page_number": 1}],
                "chunks": [{"chunk_id": "r2-lab", "source_file": "report_002.pdf",
                            "section": "LABORATORY RESULTS", "text": "HbA1c 9.2%"}],
                "model": "pipeline-model", "chunks_used": 1}

    router = _replies(AIMessage("simple"))

    def chat(purpose):
        assert purpose == "router", f"simple route must not use the {purpose} model"
        return router

    result = _run(agent_graph.build_graph(chat=chat, simple_answer=fake_pipeline),
                  "What is the HbA1c of the diabetic patient?")

    assert pipeline_calls == ["What is the HbA1c of the diabetic patient?"]
    assert result["route"] == "simple"
    assert result["answer"].startswith("9.2%")
    assert result["model"] == "pipeline-model"
    assert result["trace"] == []
    assert result["sources"][0]["text"] == "HbA1c 9.2%"


def test_router_failure_falls_back_to_complex():
    class Broken(StubChat):
        def invoke(self, *a, **k):
            raise RuntimeError("boom")

    answer = _replies(AIMessage(REFUSAL))
    models = {"router": Broken(messages=iter([])), "agent": _replies(AIMessage("DONE")),
              "answer": answer}
    result = _run(agent_graph.build_graph(chat=models.__getitem__,
                                          simple_answer=lambda q: pytest.fail("took simple")))
    assert result["route"] == "complex"
    assert result["answer"] == REFUSAL
    assert answer.calls == 0  # nothing gathered, so no answer call is made


def test_tool_call_cap_stops_the_loop():
    def always_read(i):
        return AIMessage("", tool_calls=[{
            "name": "read_section", "id": f"call{i}",
            "args": {"source_files": ["report_003_acute_mi.pdf"], "section": "MEDICATIONS"}}])

    agent = _repeat(always_read)
    answer = _replies(AIMessage("Bisoprolol 5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]"))
    models = {"router": _replies(AIMessage("complex")), "agent": agent, "answer": answer}

    result = _run(agent_graph.build_graph(chat=models.__getitem__, max_tool_calls=3))

    assert agent.calls == 3  # no fourth planning turn once the cap is hit
    assert [t["tool"] for t in result["trace"]] == ["read_section"] * 3
    assert result["route"] == "complex"
    assert [c["chunk_id"] for c in result["chunks"]] == ["r3-med"]  # deduped
    assert result["sources"][0]["section"] == "MEDICATIONS"


def test_parallel_calls_past_the_cap_are_not_run():
    two_calls = AIMessage("", tool_calls=[
        {"name": "search_documents", "id": "a", "args": {"query": "bisoprolol", "k": 3}},
        {"name": "list_documents", "id": "b", "args": {}},
    ])
    models = {"router": _replies(AIMessage("complex")), "agent": _replies(two_calls),
              "answer": _replies(AIMessage(REFUSAL))}
    graph = agent_graph.build_graph(chat=models.__getitem__, max_tool_calls=1)
    state = graph.invoke(agent_graph.initial_state("q"), {"recursion_limit": 20})

    assert state["tool_calls_used"] == 1
    assert [t["tool"] for t in state["trace"]] == ["search_documents"]
    skipped = state["messages"][-1]
    assert skipped.tool_call_id == "b" and "limit" in skipped.content


def test_grounding_strips_citation_to_unretrieved_chunk():
    read = AIMessage("", tool_calls=[{
        "name": "read_section", "id": "c1",
        "args": {"source_files": ["report_003", "report_006"], "section": "MEDICATIONS"}}])
    # report_006 has no MEDICATIONS section, so that citation was never retrieved.
    bad = ("MI patient: bisoprolol 5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]. "
           "Stroke patient: 2.5 mg [Source: report_006_stroke.pdf | MEDICATIONS].")
    answer = _replies(AIMessage(bad), AIMessage(bad))  # ignores the correction too
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(read, AIMessage("DONE")), "answer": answer}

    result = _run(agent_graph.build_graph(chat=models.__getitem__))

    assert answer.calls == 2  # one regeneration, then strip
    assert "report_006_stroke.pdf | MEDICATIONS" not in result["answer"]
    assert "[Source: report_003_acute_mi.pdf | MEDICATIONS]" in result["answer"]
    assert [(s["source_file"], s["section"]) for s in result["sources"]] == [
        ("report_003_acute_mi.pdf", "MEDICATIONS")]
    assert any(t["step"] == "Removed unverifiable citations" for t in result["trace"])


def test_answer_with_no_verifiable_citation_is_flagged():
    read = AIMessage("", tool_calls=[{
        "name": "read_section", "id": "c1",
        "args": {"source_files": ["report_003"], "section": "MEDICATIONS"}}])
    uncited = "The MI patient takes bisoprolol 5 mg."
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(read, AIMessage("DONE")),
              "answer": _replies(AIMessage(uncited), AIMessage(uncited))}

    result = _run(agent_graph.build_graph(chat=models.__getitem__))

    assert result["answer"].startswith(agent_graph.UNVERIFIED_NOTE)
    assert result["sources"] == []


def test_regeneration_that_fixes_citations_is_kept():
    read = AIMessage("", tool_calls=[{
        "name": "search_documents", "id": "c1", "args": {"query": "bisoprolol", "k": 2}}])
    bad = "2.5 mg [Source: report_009_sepsis.pdf | MEDICATIONS]"
    good = "2.5 mg [Source: report_006_stroke.pdf | TREATMENT PLAN]"
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(read, AIMessage("DONE")),
              "answer": _replies(AIMessage(bad), AIMessage(good))}
    result = _run(agent_graph.build_graph(chat=models.__getitem__))
    assert result["answer"] == good
    assert result["sources"][0]["text"].startswith("Bisoprolol 2.5 mg")


def test_check_citations_handles_variants():
    text = ("A 【Source: report_003 | Section: MEDICATIONS】 and "
            "B [Source: report_006_stroke.pdf | treatment plan; report_001.pdf | DIAGNOSIS].")
    check = check_citations(text, CHUNKS)
    assert check["unknown"] == [("report_001.pdf", "DIAGNOSIS")]
    assert check["n_cited"] == 2
    assert check["answer"] == ("A [Source: report_003_acute_mi.pdf | MEDICATIONS] and "
                               "B [Source: report_006_stroke.pdf | TREATMENT PLAN].")


def test_check_citations_accepts_missing_source_label():
    # Verbatim shape from a gpt-oss-20b answer in the API smoke test.
    text = ("started on **bisoprolol 2.5 mg once daily**[report_006_stroke.pdf | TREATMENT PLAN]. "
            "Unrelated brackets [see note] stay.")
    check = check_citations(text, CHUNKS)
    assert check["n_cited"] == 1 and check["unknown"] == []
    assert "**[Source: report_006_stroke.pdf | TREATMENT PLAN]." in check["answer"]
    assert "[see note]" in check["answer"]


def test_read_section_reports_available_sections():
    content, chunks = agent_tools.read_section.func(["report_006_stroke.pdf"], "IMAGING")
    assert chunks == []
    assert "TREATMENT PLAN" in content and "PATIENT INFORMATION" in content


def test_retry_waits_for_retry_after():
    class RateLimited(Exception):
        status_code = 429

        class response:
            headers = {"retry-after": "3"}

    class Flaky:
        n = 0

        def invoke(self, _):
            self.n += 1
            if self.n == 1:
                raise RateLimited("rate limit")
            return "ok"

    waits = []
    assert invoke_with_retry(Flaky(), [], sleep=waits.append) == "ok"
    assert waits == [4.0]


def test_list_documents_passes_on_patient_info_only():
    content, chunks = agent_tools.list_documents.func()
    assert "SYN-2024-006, Female, age 63" in content
    assert [c["chunk_id"] for c in chunks] == ["r6-info"]


def test_malformed_tool_call_is_retried_then_abandoned():
    class BadToolCall(Exception):
        status_code = 400

    class Planner(StubChat):
        def invoke(self, *a, **k):
            self.calls += 1
            raise BadToolCall("tool_use_failed")

    planner = Planner(messages=iter([]))
    models = {"router": _replies(AIMessage("complex")), "agent": planner,
              "answer": _replies(AIMessage(REFUSAL))}
    result = _run(agent_graph.build_graph(chat=models.__getitem__))
    assert planner.calls == 2
    assert result["answer"] == REFUSAL
    assert result["trace"][0]["step"].startswith("Planner error")


def _read_then_answer(*drafts, section="MEDICATIONS", files=("report_003",)):
    read = AIMessage("", tool_calls=[{
        "name": "read_section", "id": "c1",
        "args": {"source_files": list(files), "section": section}}])
    answer = _replies(*(AIMessage(d) for d in drafts))
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(read, AIMessage("DONE")), "answer": answer}
    return agent_graph.build_graph(chat=models.__getitem__), answer


def test_refusal_mixed_with_a_claim_is_not_a_refusal():
    mixed = ("The patient takes warfarin 10 mg daily. The INR is not found in the "
             "available clinical documents.")
    assert not is_refusal(mixed)
    assert is_refusal(REFUSAL)
    assert is_refusal(f'"{REFUSAL}" [Source: report_003_acute_mi.pdf | MEDICATIONS]')

    graph, answer = _read_then_answer(mixed, mixed)
    result = _run(graph)
    assert answer.calls == 2  # the uncited claim triggers the rewrite
    assert result["answer"].startswith(agent_graph.UNVERIFIED_NOTE)
    assert result["sources"] == []


def test_removed_citation_is_flagged_in_the_answer():
    bad = ("MI patient: bisoprolol 5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]. "
           "Stroke patient: 2.5 mg [Source: report_006_stroke.pdf | MEDICATIONS].")
    graph, _ = _read_then_answer(bad, bad, files=("report_003", "report_006"))
    result = _run(graph)
    assert result["answer"].startswith(agent_graph.REMOVED_NOTE)
    assert "[Source: report_003_acute_mi.pdf | MEDICATIONS]" in result["answer"]


def test_clean_answer_has_no_note():
    good = "Bisoprolol 5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]"
    result = _run(_read_then_answer(good)[0])
    assert result["answer"] == good


def test_context_skips_what_does_not_fit_and_puts_reads_first(monkeypatch):
    monkeypatch.setattr(agent_graph, "ANSWER_CONTEXT_CHARS", 250)
    listed = {**CHUNKS[2], "via": "list_documents"}
    big = {"chunk_id": "big", "source_file": "report_009_sepsis.pdf", "section": "IMAGING",
           "text": "x" * 300, "page_number": 1, "via": "read_section"}
    small = {**CHUNKS[0], "via": "read_section"}
    sent, dropped = agent_graph._select_context([listed, big, small])
    assert [c["chunk_id"] for c in sent] == ["r3-med", "r6-info"]
    assert [c["chunk_id"] for c in dropped] == ["big"]


def test_listed_section_that_was_also_read_is_not_sent_twice():
    listed = {**CHUNKS[2], "chunk_id": "r6-info-listed", "via": "list_documents"}
    read = {**CHUNKS[2], "via": "read_section"}
    sent, _ = agent_graph._select_context([listed, read])
    assert [c["chunk_id"] for c in sent] == ["r6-info"]


def test_merge_upgrades_listed_chunk_that_is_later_read():
    merged = agent_graph._merge_chunks([{**CHUNKS[2], "via": "list_documents"}],
                                       [{**CHUNKS[2], "via": "read_section"}])
    assert len(merged) == 1 and merged[0]["via"] == "read_section"


def test_citation_to_passage_left_out_of_the_prompt_is_rejected(monkeypatch):
    # Room for the MEDICATIONS passage only; TREATMENT PLAN is gathered but dropped.
    monkeypatch.setattr(agent_graph, "ANSWER_CONTEXT_CHARS", 110)
    read = AIMessage("", tool_calls=[{
        "name": "read_section", "id": "c1",
        "args": {"source_files": ["report_003"], "section": "MEDICATIONS"}}, {
        "name": "read_section", "id": "c2",
        "args": {"source_files": ["report_006"], "section": "TREATMENT PLAN"}}])
    cites_dropped = ("5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]; "
                     "2.5 mg [Source: report_006_stroke.pdf | TREATMENT PLAN]")
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(read, AIMessage("DONE")),
              "answer": _replies(AIMessage(cites_dropped), AIMessage(cites_dropped))}
    result = _run(agent_graph.build_graph(chat=models.__getitem__))

    assert [c["chunk_id"] for c in result["chunks"]] == ["r3-med"]
    assert {c["chunk_id"] for c in result["gathered_chunks"]} == {"r3-med", "r6-plan"}
    assert "TREATMENT PLAN]" not in result["answer"]
    assert result["answer"].startswith(agent_graph.REMOVED_NOTE)
    steps = [t["step"] for t in result["trace"]]
    assert "Left out passages over the answer size limit" in steps


def test_simple_route_answer_is_citation_checked():
    def fake_pipeline(question):
        return {"query": question,
                "answer": "9.2% [Source: report_002.pdf | LABORATORY RESULTS] and "
                          "metformin [Source: report_002.pdf | MEDICATIONS]",
                "sources": [], "model": "m", "chunks_used": 2,
                "chunks": [{"chunk_id": "r2-lab", "source_file": "report_002.pdf",
                            "section": "LABORATORY RESULTS", "text": "HbA1c 9.2%"}]}

    models = {"router": _replies(AIMessage("simple"))}
    result = _run(agent_graph.build_graph(chat=models.__getitem__, simple_answer=fake_pipeline))
    assert result["answer"].startswith(agent_graph.REMOVED_NOTE)
    assert "MEDICATIONS" not in result["answer"]
    assert [(s["section"], s["text"]) for s in result["sources"]] == [("LABORATORY RESULTS", "HbA1c 9.2%")]
    assert result["trace"][0]["step"] == "Removed unverifiable citations"


def test_tool_errors_do_not_reach_the_trace(monkeypatch):
    def boom(query, k):
        raise RuntimeError("secret org_123 detail")

    monkeypatch.setattr(agent_tools, "_retrieve", boom)
    search = AIMessage("", tool_calls=[{"name": "search_documents", "id": "s1",
                                        "args": {"query": "x"}}])
    models = {"router": _replies(AIMessage("complex")),
              "agent": _replies(search, AIMessage("DONE")), "answer": _replies()}
    result = _run(agent_graph.build_graph(chat=models.__getitem__))
    assert "org_123" not in str(result["trace"])
    assert result["answer"] == REFUSAL


def test_planner_error_text_stays_out_of_the_trace():
    class BadToolCall(Exception):
        status_code = 400

    class Planner(StubChat):
        def invoke(self, *a, **k):
            raise BadToolCall("tool_use_failed org_123")

    models = {"router": _replies(AIMessage("complex")), "agent": Planner(messages=iter([])),
              "answer": _replies()}
    result = _run(agent_graph.build_graph(chat=models.__getitem__))
    assert "org_123" not in str(result["trace"])


def test_preset_route_skips_the_router():
    def chat(purpose):
        assert purpose != "router", "a route passed in must not be classified again"
        return {"agent": _replies(AIMessage("DONE")), "answer": _replies(AIMessage(REFUSAL))}[purpose]

    graph = agent_graph.build_graph(chat=chat, simple_answer=lambda q: pytest.fail("took simple"))
    state = graph.invoke(agent_graph.initial_state("q", route="complex"), {"recursion_limit": 20})
    assert state["route"] == "complex"

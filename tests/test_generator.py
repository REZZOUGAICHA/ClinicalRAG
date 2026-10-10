import json
from types import SimpleNamespace as NS

from src.generation import generator


class FakeClient:
    """Stands in for Groq; records the kwargs and returns a canned response."""

    def __init__(self, response):
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _completion(content, finish_reason="stop"):
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish_reason)])


def _use(monkeypatch, response):
    client = FakeClient(response)
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    return client


CHUNKS = [
    {"source_file": "report_002.pdf", "section": "LABORATORY RESULTS", "text": "HbA1c 9.2%"},
    {"source_file": "report_002.pdf", "section": "MEDICATIONS", "text": "Metformin"},
]


def test_rewrite_falls_back_on_empty_content(monkeypatch):
    # what gpt-oss returned when reasoning used the whole 100-token budget
    _use(monkeypatch, _completion("", finish_reason="length"))
    assert generator.rewrite_query("whats the id of lymphoma patien") == "whats the id of lymphoma patien"


def test_rewrite_falls_back_on_truncated_content(monkeypatch):
    _use(monkeypatch, _completion("What medications is the patient with", finish_reason="length"))
    assert generator.rewrite_query("wat medicatons ...") == "wat medicatons ..."


def test_rewrite_uses_completion_token_cap_and_low_effort(monkeypatch):
    client = _use(monkeypatch, _completion('"What is the patient ID?"'))
    assert generator.rewrite_query("whats the patient id") == "What is the patient ID?"
    kwargs = client.calls[0]
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == generator.REWRITE_MAX_TOKENS
    assert kwargs["reasoning_effort"] == "low"


def test_answer_normalizes_lenticular_brackets(monkeypatch):
    text = "HbA1c is 9.2%\u3010Source: report_002.pdf | LABORATORY RESULTS\u3011"
    _use(monkeypatch, _completion(text))
    result = generator.generate_answer("hba1c?", CHUNKS)
    assert result["answer"].endswith("[Source: report_002.pdf | LABORATORY RESULTS]")
    assert [s["section"] for s in result["sources"]] == ["LABORATORY RESULTS"]


def test_stream_skips_reasoning_and_normalizes_brackets(monkeypatch):
    def chunk(content=None, reasoning=None):
        return NS(choices=[NS(delta=NS(content=content, reasoning=reasoning))])

    events = [
        chunk(reasoning="Look at passage 1..."),
        chunk(content="HbA1c is 9.2%"),
        chunk(content="\u3010Source: report_002.pdf | LABORATORY RESULTS\u3011"),
        NS(choices=[]),  # OpenAI-style usage chunk with no choices
    ]
    _use(monkeypatch, iter(events))

    out = [json.loads(e[len("data: "):]) for e in generator.stream_answer("hba1c?", CHUNKS)]
    tokens = "".join(e["content"] for e in out if e["type"] == "token")
    assert "passage 1" not in tokens
    assert tokens == "HbA1c is 9.2%[Source: report_002.pdf | LABORATORY RESULTS]"
    sources = next(e["sources"] for e in out if e["type"] == "sources")
    assert [s["section"] for s in sources] == ["LABORATORY RESULTS"]
    assert out[-1] == {"type": "done"}

"""API behaviour that doesn't need models: upload filename handling and the
pipeline stream's error path. The app's lifespan (model loading) never runs
because TestClient isn't used as a context manager."""

import json

from fastapi.testclient import TestClient

from src.api import main


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_upload_keeps_only_the_base_name(monkeypatch):
    seen = []
    monkeypatch.setattr(main, "_ingest_upload", lambda name, data: seen.append(name) or {"filename": name})
    client = TestClient(main.app)

    r = client.post("/upload", files={"file": ("../../evil.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    assert seen == ["evil.pdf"]

    r = client.post("/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_pipeline_stream_ends_with_error_and_done_when_groq_fails(monkeypatch):
    class RateLimitError(Exception):
        pass

    def failing_stream(query, chunks):
        yield 'data: {"type": "token", "content": "Bisoprolol"}\n\n'
        raise RateLimitError("429 org_123 secret detail")

    monkeypatch.setattr(main, "rewrite_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve", lambda q, k: [])
    monkeypatch.setattr(main, "stream_answer", failing_stream)

    r = TestClient(main.app).post("/ask", json={"query": "dose?"})
    events = _events(r.text)
    assert [e["type"] for e in events] == ["token", "error", "done"]
    assert "RateLimitError" in events[1]["message"]
    assert "org_123" not in r.text

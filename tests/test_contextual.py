import json

import pytest

from src.config import RetrievalSettings, load_settings
from src.retrieval import contextual
from src.retrieval.contextual import (
    _parse_reply,
    build_prompt,
    cache_entry,
    cached_context,
    indexed_text,
)

CHUNK = {"chunk_id": "r1__PATIENT_INFORMATION__0", "source_file": "r1.pdf",
         "section": "PATIENT INFORMATION", "text": "Patient ID: SYN-2024-001"}


ENV_VARS = ("RAG_EMBEDDER", "RAG_VECTORSTORE_DIR", "RAG_RERANKER", "RAG_RERANK_POOL",
            "RAG_FIRST_STAGE_K", "RAG_RERANK_BATCH", "RAG_CONTEXTUAL", "RAG_RERANK_CONTEXT",
            "RAG_CONTEXTS_PATH")


def test_defaults_are_the_chosen_config(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s == RetrievalSettings()
    # ctx-pool20-b1 in eval/results/retrieval_experiments.md
    assert (s.embedder, s.reranker, s.rerank_pool, s.first_stage_k, s.rerank_batch_size) == (
        "bge-small-en", "ms-marco-minilm", 20, 10, 1)
    assert s.contextual and not s.rerank_with_context


def test_baseline_still_reachable(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_CONTEXTUAL", "0")
    monkeypatch.setenv("RAG_RERANK_POOL", "5")
    monkeypatch.setenv("RAG_RERANK_BATCH", "32")
    s = load_settings()
    assert (s.contextual, s.rerank_pool, s.rerank_batch_size) == (False, 5, 32)


def test_unknown_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDER", "bge-large")
    with pytest.raises(ValueError):
        load_settings()


def test_indexed_text_prepends_context_only_when_present():
    assert indexed_text("chunk", None) == "chunk"
    assert indexed_text("chunk", "ctx") == "ctx\n\nchunk"


def test_stale_context_is_ignored():
    contexts = {CHUNK["chunk_id"]: cache_entry(CHUNK, "Lymphoma report.")}
    assert cached_context(CHUNK, contexts) == "Lymphoma report."
    edited = {**CHUNK, "text": "Patient ID: SYN-2024-009"}
    assert cached_context(edited, contexts) is None


def test_parse_reply_prepends_document_sentence_and_fixes_dashes():
    reply = json.dumps({"document": "Report for SYN‑2024‑001.",
                        "chunks": {"1": "Demographics.", "2": "Labs.", "9": "out of range"}})
    out = _parse_reply(reply, 2)
    assert out == {1: "Report for SYN-2024-001. Demographics.",
                   2: "Report for SYN-2024-001. Labs."}


def test_prompt_numbers_every_chunk():
    prompt = build_prompt("r1.pdf", [CHUNK, {**CHUNK, "section": "LABS"}])
    assert '<chunk number="1" section="PATIENT INFORMATION">' in prompt
    assert '<chunk number="2" section="LABS">' in prompt


def test_upload_falls_back_to_no_context_when_llm_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(contextual, "load_contexts", lambda path=None: {})
    saved = []
    monkeypatch.setattr(contextual, "save_contexts", lambda c, path=None: saved.append(c))

    def boom(*a, **kw):
        raise RuntimeError("no key")

    monkeypatch.setattr(contextual, "generate_document_contexts", boom)
    assert contextual.contextualize_chunks([CHUNK]) == [None]
    assert saved == []  # failures aren't cached


def test_save_contexts_replaces_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "chunk_contexts.json"
    contextual.save_contexts({"a": {"context": "x"}}, path)
    contextual.save_contexts({"b": {"context": "y"}}, path)
    assert contextual.load_contexts(path) == {"b": {"context": "y"}}
    assert [p.name for p in tmp_path.iterdir()] == ["chunk_contexts.json"]

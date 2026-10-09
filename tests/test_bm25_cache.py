"""The BM25 cache under concurrent reads and resets, as the API does them."""

import dataclasses
import json
import threading

from src.retrieval import retriever


def test_index_and_chunks_are_always_returned_together(tmp_path, monkeypatch):
    chunks = [{"chunk_id": f"c{i}", "text": f"bisoprolol dose {i} mg", "source_file": "a.pdf",
               "section": "MEDICATIONS"} for i in range(50)]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    monkeypatch.setattr(retriever, "CHUNKS_PATH", path)
    monkeypatch.setattr(retriever, "settings", dataclasses.replace(retriever.settings, contextual=False))
    retriever.reset_bm25()

    errors = []

    def reader():
        for _ in range(200):
            index, got = retriever._get_bm25()
            if index is None or got is None or index.corpus_size != len(got):
                errors.append((index, got))

    def resetter():
        for _ in range(200):
            retriever.reset_bm25()

    threads = [threading.Thread(target=reader) for _ in range(6)] + [threading.Thread(target=resetter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    retriever.reset_bm25()

    assert errors == []

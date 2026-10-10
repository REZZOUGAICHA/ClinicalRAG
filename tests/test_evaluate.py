"""evaluate.py checkpoint handling, with a fake answer function and no API calls."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_common  # noqa: E402
import evaluate  # noqa: E402

ITEM = {"id": "q1", "type": "lookup", "answerable": True, "question": "Bisoprolol dose?",
        "reference": "5 mg", "key_facts": ["5 mg"], "gold": []}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "load_questions", lambda *a: [ITEM])
    monkeypatch.setattr(evaluate, "append_summary", lambda record: None)
    calls = []

    def answer_fn(question):
        calls.append(question)
        return {"answer": "5 mg [Source: report_003_acute_mi.pdf | MEDICATIONS]", "model": "fake",
                "chunks": [{"source_file": "report_003_acute_mi.pdf", "section": "MEDICATIONS",
                            "text": "Bisoprolol 5 mg"}]}

    monkeypatch.setattr(evaluate, "_load_answer_fn", lambda system: answer_fn)
    args = SimpleNamespace(config="t", system="pipeline", types=None, ids=None, limit=None,
                           ragas=False, pace=0)
    return tmp_path, args, calls


def test_checkpoint_is_kept_and_reused(run):
    tmp_path, args, calls = run
    asyncio.run(evaluate.run_eval(args))
    asyncio.run(evaluate.run_eval(args))
    assert calls == ["Bisoprolol dose?"]  # second run answered from the checkpoint
    rows = json.loads((tmp_path / "checkpoint_t.json").read_text(encoding="utf-8"))
    assert rows[0]["system"] == "pipeline"
    assert rows[0]["retrieval"] == evaluate.settings.describe()


def test_stored_scores_are_recomputed(run):
    tmp_path, args, _ = run
    asyncio.run(evaluate.run_eval(args))
    path = tmp_path / "checkpoint_t.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["scores"] = {"refused": True, "key_fact_recall": 0.0, "missing_facts": ["5 mg"],
                         "false_refusal": True}
    path.write_text(json.dumps(rows), encoding="utf-8")

    asyncio.run(evaluate.run_eval(args))
    result = json.loads(next(tmp_path.glob("eval_t_*.json")).read_text(encoding="utf-8"))
    assert result["aggregate"]["key_fact_recall"] == 1.0


@pytest.mark.parametrize("change", [{"system": "agent"}, {"retrieval": {"rerank_pool": 5}},
                                    {"retrieval": None}])
def test_checkpoint_from_other_settings_is_refused(run, change):
    tmp_path, args, calls = run
    asyncio.run(evaluate.run_eval(args))
    path = tmp_path / "checkpoint_t.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0].update(change)
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="different --system or retrieval settings"):
        asyncio.run(evaluate.run_eval(args))


def test_rescore_row_uses_current_key_facts():
    row = {"id": "q1", "answer": "Bisoprolol 2.5 mg", "scores": {"key_fact_recall": 1.0}}
    item = {**ITEM, "key_facts": ["2.5 mg"]}
    assert eval_common.rescore_row(row, item)["scores"]["key_fact_recall"] == 1.0
    assert eval_common.rescore_row(row, ITEM)["scores"]["key_fact_recall"] == 0.0

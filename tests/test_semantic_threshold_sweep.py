from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scripts import run_semantic_threshold_sweep as sweep


def test_selection_prioritizes_exact_match_then_f1_then_lower_threshold():
    records = [
        {"threshold": 90, "metrics": {"exact_match": .51, "token_f1": .65}},
        {"threshold": 95, "metrics": {"exact_match": .50, "token_f1": .80}},
        {"threshold": 99, "metrics": {"exact_match": .51, "token_f1": .70}},
    ]
    assert sweep.choose(records)["threshold"] == 99
    records[0]["metrics"]["token_f1"] = .70
    assert sweep.choose(records)["threshold"] == 90


def test_provider_failure_is_counted_and_resume_does_not_regenerate(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for qid in ("1", "2"):
        sweep.append_row(dataset / "development_questions.jsonl", {"question_id": qid, "question": qid})
        sweep.append_row(dataset / "development_labels.jsonl", {
            "question_id": qid, "document_id": "d", "reference_answers": ["yes"],
            "references": [{"evidence": {"start_char": 0, "end_char": 3},
                            "short_answers": [{"start_char": 0, "end_char": 3}]}],
        })
    cfg = OmegaConf.create({"dataset_dir": str(dataset),
                           "evaluation": {"split": "development", "seed": 42, "limit": 2},
                           "embedding": {"query_prefix": ""},
                           "retrieval": {"max_context_chars": 10}})
    chunk = {"text": "yes", "metadata": {"document_id": "d", "start_char": 0, "end_char": 3}}
    monkeypatch.setattr(sweep, "retrieve", lambda *args: {"context": "yes", "chunks": [chunk], "retrieval_ms": 1})
    calls = []

    class API:
        def __init__(self, *args):
            pass

        def answer(self, question, context):
            calls.append(question)
            if question == "2":
                raise RuntimeError("OpenRouter API error (code=403): PROHIBITED_CONTENT")
            return {"answer": "yes", "usage": {"cost": .001}}

    monkeypatch.setattr(sweep, "OpenRouter", API)
    embeddings = SimpleNamespace(embed=lambda texts: [[1] for _ in texts])
    args = (cfg, SimpleNamespace(name="collection"), embeddings, tmp_path / "run", {"d": "yes"})
    result = sweep.evaluate(*args)
    assert result["questions"] == 2 and result["generation_failures"] == 1
    assert result["exact_match"] == .5 and result["token_f1"] == .5
    assert len(calls) == 2
    assert sweep.evaluate(*args) == result
    assert len(calls) == 2
    cfg.retrieval.max_context_chars = 9
    with pytest.raises(ValueError, match="Checkpoint identity changed"):
        sweep.evaluate(*args)

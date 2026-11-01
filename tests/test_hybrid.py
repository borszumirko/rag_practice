from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from app.hybrid import HybridRetriever, tokenize


def fixture():
    chunks = [{"chunk_id": cid, "text": text,
               "metadata": {"document_id": cid, "start_char": 10, "end_char": 10 + len(text)}}
              for cid, text in (("a", "alpha evidence"), ("b", "beta evidence"), ("c", "gamma evidence"))]
    cfg = SimpleNamespace(retrieval=SimpleNamespace(k=2, max_context_chars=20))
    retriever = HybridRetriever(chunks, None, None, cfg)
    candidates = {"dense": ["a", "b"], "bm25": ["c", "b"],
                  "dense_distances": {"a": 0.1, "b": 0.2}, "candidate_ms": 1}
    return retriever, candidates


def test_weight_changes_ranking_and_preserves_clipped_offsets():
    retriever, candidates = fixture()
    dense = retriever.retrieve(candidates, 0.75)
    sparse = retriever.retrieve(candidates, 0.25)
    # A chunk supported by both lists outranks single-list candidates.
    assert [c["chunk_id"] for c in dense["chunks"]] == ["b", "a"]
    assert [c["chunk_id"] for c in sparse["chunks"]] == ["b", "c"]
    assert len(dense["context"]) == 20
    assert dense["context"] == "beta evidence\n\nalpha"
    assert dense["chunks"][1]["metadata"]["end_char"] == 15
    assert sparse["chunks"][1]["cosine_distance"] is None


def test_endpoints_do_not_include_other_retriever_and_ids_are_unique():
    retriever, candidates = fixture()
    assert [c["chunk_id"] for c in retriever.retrieve(candidates, 1)["chunks"]] == ["a", "b"]
    assert [c["chunk_id"] for c in retriever.retrieve(candidates, 0)["chunks"]] == ["c", "b"]
    assert len({c["chunk_id"] for c in retriever.retrieve(candidates, 0.5)["chunks"]}) == 2
    with pytest.raises(ValueError):
        retriever.retrieve(candidates, 2)


def test_tokenization_handles_case_punctuation_numbers_and_unicode():
    assert tokenize("Beyoncé's 2016—SHOW!") == ["beyoncé", "s", "2016", "show"]


def test_lexical_candidates_exclude_unmatched_chunks_and_query_instruction():
    retriever, _ = fixture()
    retriever.cfg.embedding = SimpleNamespace(query_prefix="Instruct: retrieval\nQuery: ")
    embedded = []
    retriever.embeddings = SimpleNamespace(embed=lambda texts: embedded.extend(texts) or [[1]])
    retriever.collection = SimpleNamespace(query=lambda **kw: {
        "ids": [["b", "a", "c"]], "distances": [[0.1, 0.2, 0.3]]})
    retriever.depth = 2
    candidates = retriever.candidates("ALPHA!")
    assert candidates["bm25"] == ["a"]
    assert embedded == ["Instruct: retrieval\nQuery: ALPHA!"]
    assert retriever.candidates("unmatched")["bm25"] == []


def test_sweep_resumes_without_repeating_answers_and_counts_failures(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_hybrid_sweep as sweep

    calls = []

    class API:
        def __init__(self, *args):
            pass

        def answer(self, question, context):
            calls.append(question)
            if question == "blocked":
                raise RuntimeError("No answer returned")
            return {"answer": "yes", "usage": {"cost": 0.001}}

    monkeypatch.setattr(sweep, "OpenRouter", API)
    chunk = {"text": "yes", "metadata": {"document_id": "d", "start_char": 0, "end_char": 3}}
    sources = [{"question_id": q, "question": q, "context": "yes", "chunks": [chunk],
                "retrieval_ms": 1} for q in ("ok", "blocked")]
    label = {"document_id": "d", "reference_answers": ["yes"], "references": [
        {"evidence": {"start_char": 0, "end_char": 3},
         "short_answers": [{"start_char": 0, "end_char": 3}]}]}
    labels = {q: label for q in ("ok", "blocked")}
    cfg = OmegaConf.create({"retrieval": {"max_context_chars": 20}})
    args = (cfg, tmp_path / "run", sources, labels, {"d": "yes"})
    _, result = sweep.generate(*args)
    assert result["questions"] == 2 and result["generation_failures"] == 1
    assert result["exact_match"] == 0.5
    assert sweep.generate(*args)[1] == result
    assert calls == ["ok", "blocked"]
    cfg.retrieval.max_context_chars = 19
    with pytest.raises(ValueError, match="Checkpoint identity changed"):
        sweep.generate(*args)


def test_key_limit_is_not_scored_as_a_content_refusal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_hybrid_sweep as sweep

    error = RuntimeError("OpenRouter HTTP 403; stopping")
    assert sweep.terminal_generation_failure(
        RuntimeError("OpenRouter API error (code=403): Gemini blocked the request: PROHIBITED_CONTENT"),
        tmp_path)
    sweep.append_row(tmp_path / "api_errors.jsonl", {
        "body": '{"error":{"message":"Key limit exceeded (total limit)","code":403}}'})
    assert not sweep.terminal_generation_failure(error, tmp_path)
    sweep.append_row(tmp_path / "api_errors.jsonl", {
        "body": '{"error":{"message":"PROHIBITED_CONTENT","code":403}}'})
    assert sweep.terminal_generation_failure(error, tmp_path)

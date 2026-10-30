from types import SimpleNamespace

import pytest

from app.rag import Embeddings, chunk_document, index, open_collection, retrieve, retrieval_scores, span_coverage


def test_recursive_chunks_retain_exact_offsets_and_cover_all_nonspace_text():
    text = "First paragraph has repeated words.\n\nSecond paragraph repeats repeated words.\n\n" + "x" * 90
    doc = {"document_id": "doc", "title": "Title", "url": "https://example.org", "text": text}
    config = {"chunk_size": 35, "chunk_overlap": 8, "separators": ["\n\n", "\n", " ", ""]}
    chunks = list(chunk_document(doc, config))
    covered = set()
    for chunk in chunks:
        start, end = chunk["metadata"]["start_char"], chunk["metadata"]["end_char"]
        assert text[start:end] == chunk["text"]
        assert len(chunk["text"]) <= 35
        covered.update(range(start, end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert chunks == list(chunk_document(doc, config))


@pytest.mark.parametrize("size,overlap", [(0, 0), (20, 20), (20, -1)])
def test_invalid_chunk_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        list(chunk_document({}, {"chunk_size": size, "chunk_overlap": overlap}))


def chunk(doc_id, start, end):
    return {"metadata": {"document_id": doc_id, "start_char": start, "end_char": end}}


def test_coverage_merges_overlaps_and_ignores_other_documents():
    chunks = [chunk("doc", 0, 6), chunk("doc", 4, 8), chunk("other", 0, 10)]
    assert span_coverage({"start_char": 0, "end_char": 10}, chunks, "doc") == 0.8


def test_all_parts_of_one_answer_alternative_are_required():
    label = {"document_id": "doc", "references": [
        {"evidence": {"start_char": 0, "end_char": 20},
         "short_answers": [{"start_char": 0, "end_char": 2}, {"start_char": 18, "end_char": 20}]}]}
    partial = retrieval_scores(label, [chunk("doc", 0, 10)])
    assert partial["reference_page_hit"] == 1
    assert partial["complete_answer_span_hit"] == 0
    assert partial["evidence_coverage"] == 0.5
    assert retrieval_scores(label, [chunk("doc", 0, 10), chunk("doc", 10, 20)])["complete_answer_span_hit"] == 1


def test_local_chroma_index_resumes_and_separates_chunk_variants(tmp_path):
    cfg = SimpleNamespace(chroma_dir=str(tmp_path / "chroma"),
                          embedding=SimpleNamespace(model="test-model", batch_size=2))
    calls = []

    class FakeEmbeddings:
        def embed(self, texts):
            calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

    chunks = [{"chunk_id": "first", "text": "First passage", "metadata": {"document_id": "doc"}}]
    report = {"chunks_sha256": "first-config"}
    assert index(cfg, chunks, report, FakeEmbeddings())["indexed_chunks"] == 1
    assert index(cfg, chunks, report, FakeEmbeddings())["indexed_chunks"] == 1
    assert len(calls) == 1
    collection = open_collection(cfg, chunks, report)
    found = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert found["documents"] == [["First passage"]]
    other = open_collection(cfg, chunks, {"chunks_sha256": "second-config"}, create=True)
    assert other.name != collection.name
    assert other.count() == 0


def test_embedding_cache_deduplicates_content_but_separates_models(tmp_path):
    cfg = SimpleNamespace(embedding_cache=str(tmp_path / "vectors.sqlite3"),
                          embedding=SimpleNamespace(model="model-a", batch_size=4,
                                                    provider_order=["Nebius"], provider_only=["Nebius"]))
    calls = []

    class FakeAPI:
        def call(self, endpoint, payload):
            calls.append(payload)
            return {"data": [{"index": i, "embedding": [float(i + 1), 0.0]}
                             for i in reversed(range(len(payload["input"])))]}

    cache = Embeddings(cfg, FakeAPI())
    try:
        result = cache.embed(["same", "same", "other"])
        assert result[0] == result[1]
        assert len(calls[0]["input"]) == 2
        assert calls[0]["provider"] == {"order": ["Nebius"], "allow_fallbacks": True, "only": ["Nebius"]}
        assert cache.embed(["same"]) == [result[0]]
        assert len(calls) == 1
        cfg.embedding.model = "model-b"
        cache.embed(["same"])
        assert len(calls) == 2
    finally:
        cache.db.close()


def test_query_instruction_and_context_clipping_preserve_source_offsets():
    cfg = SimpleNamespace(embedding=SimpleNamespace(query_prefix="Instruct: task\nQuery:"),
                          retrieval=SimpleNamespace(k=5, max_context_chars=10))

    class FakeEmbeddings:
        def embed(self, texts):
            assert texts == ["Instruct: task\nQuery:question"]
            return [[1.0, 0.0]]

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs["n_results"] == 2
            return {"ids": [["one", "two"]], "documents": [["abcdef", "ghijkl"]],
                    "metadatas": [[{"document_id": "doc", "start_char": 0, "end_char": 6},
                                   {"document_id": "doc", "start_char": 10, "end_char": 16}]],
                    "distances": [[0.1, 0.2]]}

    result = retrieve("question", FakeCollection(), FakeEmbeddings(), cfg)
    assert result["context"] == "abcdef\n\ngh"
    assert result["chunks"][1]["metadata"]["end_char"] == 12


def test_concurrent_embedding_failure_preserves_other_successful_batches(tmp_path):
    cfg = SimpleNamespace(embedding_cache=str(tmp_path / "vectors.sqlite3"),
                          embedding=SimpleNamespace(model="model", batch_size=1, concurrency=3))
    calls = []

    class FakeAPI:
        fail = True

        def call(self, endpoint, payload):
            text = payload["input"][0]
            calls.append(text)
            if text == "bad" and self.fail:
                raise RuntimeError("temporary provider failure")
            return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    api = FakeAPI()
    cache = Embeddings(cfg, api)
    try:
        with pytest.raises(RuntimeError, match="temporary"):
            cache.embed(["good", "bad", "other"])
        assert cache.db.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
        api.fail = False
        assert len(cache.embed(["good", "bad", "other"])) == 3
        assert calls.count("good") == calls.count("other") == 1
        assert calls.count("bad") == 2
    finally:
        cache.db.close()

import json
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from omegaconf import OmegaConf
import pytest

from app.rag import prepare_chunks
from app.semantic import make_splitter, semantic_offsets


CONFIG = dict(method="semantic", buffer_size=0, breakpoint_threshold_type="percentile",
              breakpoint_threshold_amount=50, sentence_split_regex=r"(?<=[.?!])\s+", min_chunk_size=None)


class FakeEmbeddings:
    def embed(self, texts):
        return [[0, 1] if "Switch" in t else [1, 0] for t in texts]


def test_real_langchain_chunker_decides_groups_and_adapter_restores_source_offsets(monkeypatch):
    text = "α Same. \n\tSame.    Switch! \nSwitch?"
    splitter = make_splitter(FakeEmbeddings(), CONFIG)
    assert isinstance(splitter, SemanticChunker)
    original = splitter.split_text
    seen = []

    def observed(source):
        chunks = original(source)
        seen.append(chunks)
        return chunks

    monkeypatch.setattr(splitter, "split_text", observed)
    offsets = semantic_offsets(text, splitter)
    assert seen == [["α Same. Same.", "Switch! Switch?"]]
    assert offsets == [(0, text.index("Switch")), (text.index("Switch"), len(text))]
    assert "".join(text[start:end] for start, end in offsets) == text


def test_short_and_blank_sources_and_trailing_delimiters():
    splitter = make_splitter(FakeEmbeddings(), CONFIG)
    for text in ["One sentence", "Same. \n\n", "α Same. Same. \t"]:
        offsets = semantic_offsets(text, splitter)
        assert "".join(text[a:b] for a, b in offsets) == text
    assert semantic_offsets("", splitter) == []
    assert semantic_offsets(" \n\t", splitter) == []


@pytest.mark.parametrize("change", [{"buffer_size": -1}, {"breakpoint_threshold_amount": 101},
                                  {"sentence_split_regex": r"(\s+)"}, {"min_chunk_size": 0}])
def test_invalid_config_rejected_before_embedding(change):
    cfg = dict(CONFIG, **change)
    with pytest.raises(ValueError):
        make_splitter(FakeEmbeddings(), cfg)


def test_zero_width_delimiters_rejected_before_embedding():
    splitter = make_splitter(FakeEmbeddings(), dict(CONFIG, sentence_split_regex=""))
    with pytest.raises(ValueError, match="consume delimiters"):
        semantic_offsets("Some text", splitter)


def test_prepared_semantic_chunks_preserve_offsets_reuse_artifacts_and_separate_models(tmp_path):
    text = "α Same.\n\nSame. Switch!\n\nSwitch?"
    doc = dict(document_id="d", title="Title", url="https://example.org", text=text)
    (tmp_path / "corpus.jsonl").write_text(json.dumps(doc) + "\n")
    cfg = OmegaConf.create(dict(dataset_dir=str(tmp_path), artifacts_dir=str(tmp_path / "artifacts"),
                               embedding=dict(model="test-a"), chunking=CONFIG))
    rows, report = prepare_chunks(cfg, FakeEmbeddings())
    assert "".join(r["text"] for r in rows) == text
    for row in rows:
        start, end = row["metadata"]["start_char"], row["metadata"]["end_char"]
        assert text[start:end] == row["text"]
    assert report["semantic_splitter_version"]
    assert prepare_chunks(cfg) == (rows, report)
    cfg.embedding.model = "test-b"
    with pytest.raises(ValueError, match="requires embeddings"):
        prepare_chunks(cfg)
    _, other = prepare_chunks(cfg, FakeEmbeddings())
    assert other["fingerprint"] != report["fingerprint"]
    Path(other["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        prepare_chunks(cfg, FakeEmbeddings())

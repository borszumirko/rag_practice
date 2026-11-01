"""LangChain SemanticChunker with an exact-source offset adapter."""

from bisect import bisect_left
import math
import re

from langchain_experimental.text_splitter import SemanticChunker

VERSION = "langchain-semantic-chunker-source-offsets-v2"


class CachedEmbeddingsAdapter:
    """Expose the existing cached embedding client to LangChain."""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        return self.embeddings.embed(texts)


def make_splitter(embeddings, config):
    kind = config["breakpoint_threshold_type"]
    amount = config["breakpoint_threshold_amount"]
    if kind not in ("percentile", "standard_deviation", "interquartile", "gradient"):
        raise ValueError("Unsupported SemanticChunker breakpoint threshold type")
    if (
        not math.isfinite(amount)
        or amount < 0
        or (kind in ("percentile", "gradient") and amount > 100)
    ):
        raise ValueError("Invalid SemanticChunker breakpoint threshold amount")
    if not isinstance(config["buffer_size"], int) or config["buffer_size"] < 0:
        raise ValueError("SemanticChunker buffer_size must be a nonnegative integer")
    if config["min_chunk_size"] is not None and config["min_chunk_size"] < 1:
        raise ValueError("SemanticChunker min_chunk_size must be positive or null")
    pattern = re.compile(config["sentence_split_regex"])
    if pattern.groups:
        raise ValueError("Use a sentence split regex without capture groups")
    return SemanticChunker(
        embeddings=CachedEmbeddingsAdapter(embeddings),
        buffer_size=config["buffer_size"],
        breakpoint_threshold_type=kind,
        breakpoint_threshold_amount=amount,
        sentence_split_regex=config["sentence_split_regex"],
        min_chunk_size=config["min_chunk_size"],
        add_start_index=False,
    )


def semantic_offsets(text, splitter):
    """Map the library's actual chunk boundaries back to untouched source.

    SemanticChunker.split_text makes every grouping decision. It joins original
    sentence strings with single spaces. Track consumed sentence boundaries in
    that representation, then restore exact source slices. Split delimiters
    belong to the preceding chunk. There is no custom size cap or overlap.
    """
    pattern = re.compile(splitter.sentence_split_regex)
    matches = list(pattern.finditer(text))
    if any(m.start() == m.end() for m in matches):
        raise ValueError("Sentence split regex must consume delimiters")
    sentences = pattern.split(text)
    starts = [0] + [m.end() for m in matches]
    ends, length = [], 0
    for sentence in sentences:
        length += len(sentence)
        ends.append(length)
        length += 1
    normalized = " ".join(sentences)
    chunks = splitter.split_text(text)
    if " ".join(chunks) != normalized:
        raise ValueError(
            "SemanticChunker output cannot be mapped losslessly to source sentences"
        )
    offsets, source_start, normalized_start = [], 0, 0
    for chunk in chunks:
        normalized_end = normalized_start + len(chunk)
        last = bisect_left(ends, normalized_end)
        if last == len(ends) or ends[last] != normalized_end:
            raise ValueError("SemanticChunker output ends inside a source sentence")
        source_end = starts[last + 1] if last + 1 < len(starts) else len(text)
        if text[source_start:source_end].strip():
            offsets.append((source_start, source_end))
        source_start, normalized_start = source_end, normalized_end + 1
    return offsets

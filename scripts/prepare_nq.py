"""Prepare a frozen question-only RAG benchmark from pinned NQ page snapshots.

Run: uv run --group data python scripts/prepare_nq.py --download
No model calls, embeddings, or Chroma writes are performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


SOURCES = Path(__file__).with_name("nq_sources.json")
BLOCK_TAG = re.compile(r"</?(?:p|h[1-6]|li|tr|table|ul|ol|dl|div|br)(?:\s|>)", re.I)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def records(struct):
    """HF Parquet represents sequences as either records or parallel lists."""
    if isinstance(struct, list):
        return struct
    if not struct:
        return []
    lengths = {len(v) for v in struct.values()}
    if len(lengths) != 1:
        raise ValueError("Mismatched sequence field lengths")
    return [dict(zip(struct, values)) for values in zip(*struct.values())]


def page_key(url):
    """Group revisions, fragments, and HTTP/HTTPS variants of a Wikipedia page."""
    parsed = urllib.parse.urlsplit(html.unescape(url))
    query = urllib.parse.parse_qs(parsed.query)
    if "title" in query:
        path = "/wiki/" + query["title"][0]
    else:
        path = urllib.parse.unquote(parsed.path)
    return parsed.netloc.lower() + path.replace(" ", "_")


def extract_text(tokens):
    """Keep all visible source tokens and map original indices to character spans.

    Text is deliberately token-spaced, not rewritten prose. Block tags become
    newlines; all other HTML tags are omitted. Offsets use Python characters,
    zero-based and end-exclusive (not original HTML byte offsets).
    """
    pieces, spans, position = [], [], 0
    block_break = False
    for token in tokens:
        if token["is_html"]:
            spans.append(None)
            block_break |= bool(BLOCK_TAG.match(token["token"]))
            continue
        value = html.unescape(token["token"])
        separator = ("\n\n" if block_break else " ") if pieces else ""
        pieces.extend([separator, value])
        position += len(separator)
        spans.append([position, position + len(value)])
        position += len(value)
        block_break = False
    return "".join(pieces), spans


def mapped_span(source, mapping, text):
    start, end = source["start_token"], source["end_token"]
    if not 0 <= start < end <= len(mapping):
        raise ValueError(f"Invalid source span: {start}:{end}")
    visible = [span for span in mapping[start:end] if span is not None]
    if not visible:
        raise ValueError("Evidence has no visible text")
    a, b = visible[0][0], visible[-1][1]
    return {"start_token": start, "end_token": end,
            "start_char": a, "end_char": b, "text": text[a:b]}


def eligible_annotations(row):
    """Require two paragraph/short-answer annotations; preserve all extractive refs."""
    tokens = row["document"]["tokens"]["token"]
    valid, paragraph_votes = [], 0
    annotations = records(row["annotations"])
    # Contradictory boolean answers need a separate answer-type policy.
    if any(a["yes_no_answer"] != -1 for a in annotations):
        return []
    for annotation in annotations:
        long = annotation["long_answer"]
        short = records(annotation["short_answers"])
        if not short or not 0 <= long["start_token"] < long["end_token"] <= len(tokens):
            continue
        if not all(long["start_token"] <= s["start_token"] < s["end_token"] <= long["end_token"]
                   for s in short):
            continue
        valid.append(annotation)
        paragraph_votes += tokens[long["start_token"]].lower() == "<p>"
    return valid if paragraph_votes >= 2 else []


def iter_rows(paths):
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            yield from batch.to_pylist()


def fetch_sources(spec, raw, download):
    paths = []
    for source in spec["files"]:
        path = raw / Path(source["path"]).name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; run with --download")
            url = (f"https://huggingface.co/datasets/{spec['dataset']}/resolve/"
                   f"{spec['revision']}/{source['path']}")
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if temporary.stat().st_size != source["size"] or digest(temporary) != source["sha256"]:
                raise ValueError(f"Download checksum mismatch: {temporary}")
            temporary.replace(path)
        if path.stat().st_size != source["size"] or digest(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        paths.append(path)
    return paths


def validate(corpus, labels, questions, provenance):
    """Validate written artifacts independently of the extraction operation."""
    docs = {row["document_id"]: row for row in corpus}
    maps = {row["document_id"]: row for row in provenance}
    seen_questions, seen_pages = set(), set()
    reference_count, surface_differences = 0, 0
    for split in labels:
        assert len(labels[split]) == len(questions[split])
        for label, question in zip(labels[split], questions[split]):
            assert set(question) == {"question_id", "question"}
            assert question["question_id"] == label["question_id"]
            assert question["question_id"] not in seen_questions
            seen_questions.add(question["question_id"])
            doc = docs[label["document_id"]]
            key = page_key(doc["url"])
            assert key not in seen_pages, "A page occurs more than once across question splits"
            seen_pages.add(key)
            mapping = maps[doc["document_id"]]["source_token_char_spans"]
            assert label["references"]
            for reference in label["references"]:
                reference_count += 1
                # NQ tokens normalize quotes/dashes and omit some citation text.
                # Preserve both surface forms instead of treating these as lost evidence.
                surface_differences += re.sub(r"\s+", "", reference["answer_text"]) != re.sub(
                    r"\s+", "", reference["extracted_answer_text"]
                )
                evidence = reference["evidence"]
                for span in [evidence, *reference["short_answers"]]:
                    assert mapped_span(span, mapping, doc["text"]) == span
                    assert span["text"].strip()
                for answer in reference["short_answers"]:
                    assert evidence["start_char"] <= answer["start_char"] < answer["end_char"] <= evidence["end_char"]
    assert len(docs) == len(corpus) == len(seen_pages)
    return {"questions_checked": len(seen_questions), "documents_checked": len(docs),
            "all_reference_spans_valid": True, "page_overlap_between_splits": 0,
            "reference_annotations_checked": reference_count,
            "reference_surface_form_differences": surface_differences}


def prepare(root, count=500, dev_count=100, seed=42, download=False):
    if not 0 < dev_count < count:
        raise ValueError("Require 0 < dev-count < questions")
    spec = json.loads(SOURCES.read_text())
    raw = root / "raw"
    output = root / "prepared"
    raw.mkdir(parents=True, exist_ok=True)
    paths = fetch_sources(spec, raw, download)
    stats = Counter()
    candidates = []
    for row in iter_rows(paths):
        stats["source_rows"] += 1
        if eligible_annotations(row):
            stats["eligible_rows"] += 1
            candidates.append((str(row["id"]), page_key(row["document"]["url"])))
    # Hash-ranked selection is stable across source iteration order.
    rank = lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    selected, pages = {}, set()
    for qid, key in sorted(candidates, key=lambda item: rank(item[0])):
        if key not in pages and qid not in selected:
            selected[qid] = "development" if len(selected) < dev_count else "held_out"
            pages.add(key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible distinct pages; requested {count}")

    corpus, provenance = [], []
    labels = {"development": [], "held_out": []}
    questions = {"development": [], "held_out": []}
    for row in iter_rows(paths):
        qid = str(row["id"])
        if qid not in selected:
            continue
        doc = row["document"]
        text, mapping = extract_text(records(doc["tokens"]))
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        doc_id = "nq-" + hashlib.sha256((page_key(doc["url"]) + "\0" + text).encode()).hexdigest()[:24]
        corpus.append({"document_id": doc_id, "title": doc["title"], "url": doc["url"],
                       "text": text, "text_sha256": text_sha})
        provenance.append({"document_id": doc_id, "source_example_id": qid,
                           "source_token_char_spans": mapping})
        references = []
        for annotation in eligible_annotations(row):
            short = records(annotation["short_answers"])
            answers = [mapped_span(s, mapping, text) for s in short]
            evidence = mapped_span(annotation["long_answer"], mapping, text)
            references.append({"annotation_id": str(annotation["id"]),
                               "answer_text": " ".join(s["text"] for s in short),
                               "extracted_answer_text": " ".join(s["text"] for s in answers),
                               "short_answers": answers, "evidence": evidence})
        split = selected[qid]
        questions[split].append({"question_id": qid, "question": row["question"]["text"]})
        labels[split].append({"question_id": qid, "document_id": doc_id,
                              "answer_type": "extractive", "references": references,
                              "reference_answers": sorted({a[key] for a in references
                                                           for key in ("answer_text", "extracted_answer_text")}),
                              "source_annotation_count": len(records(row["annotations"]))})

    corpus.sort(key=lambda r: r["document_id"])
    provenance.sort(key=lambda r: r["document_id"])
    for split in questions:
        questions[split].sort(key=lambda r: r["question_id"])
        labels[split].sort(key=lambda r: r["question_id"])
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "provenance.jsonl", provenance)
    for split in questions:
        write_jsonl(output / f"{split}_questions.jsonl", questions[split])
        write_jsonl(output / f"{split}_labels.jsonl", labels[split])
    # Read back to validate serialized offsets, not just in-memory objects.
    read = lambda name: [json.loads(line) for line in (output / name).read_text().splitlines()]
    report = validate(read("corpus.jsonl"),
                      {s: read(f"{s}_labels.jsonl") for s in labels},
                      {s: read(f"{s}_questions.jsonl") for s in questions},
                      read("provenance.jsonl"))
    stats.update({"selected_questions": count, "documents": len(corpus),
                  "development_questions": dev_count, "held_out_questions": count - dev_count,
                  "corpus_characters": sum(len(r["text"]) for r in corpus),
                  "corpus_whitespace_tokens": sum(len(r["text"].split()) for r in corpus)})
    report["statistics"] = dict(stats)
    write_json(output / "validation_report.json", report)
    artifacts = ["corpus.jsonl", "provenance.jsonl", "validation_report.json"] + [
        f"{s}_{kind}.jsonl" for s in questions for kind in ("questions", "labels")]
    write_json(output / "manifest.json", {
        "schema_version": 1, "source": spec, "seed": seed,
        "selection": "SHA256(seed:question_id) rank; >=2 paragraph+short-answer annotators; no boolean annotations; one question per canonical page",
        "reference_policy": "Keep all valid extractive annotations, including non-paragraph alternatives; no-answer annotations are not answer variants.",
        "offsets": "zero-based, end-exclusive Python character offsets; source token offsets refer to pinned NQ snapshots",
        "scope": "custom corpus-wide NQ subset; not official NQ evaluation",
        "statistics": dict(stats),
        "question_ids": {s: [r["question_id"] for r in rows] for s, rows in questions.items()},
        "artifact_sha256": {name: digest(output / name) for name in artifacts},
        "preparation_script_sha256": digest(Path(__file__)),
    })
    print(json.dumps(report, indent=2), flush=True)
    print(f"Prepared files: {output.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/natural_questions"))
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--dev-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.data_dir, args.questions, args.dev_count, args.seed, args.download)

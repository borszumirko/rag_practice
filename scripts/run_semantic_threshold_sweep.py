"""Tune native semantic percentiles on development, freeze selection, then test held-out.

Run from the repository root. Uses the existing Qwen cache and Gemini settings.
Checkpoints each response, reuses the completed 95-percentile development run,
and never selects a threshold using held-out scores. API calls incur charges.
"""

import hashlib
import json
from pathlib import Path
import statistics
import time

from importlib.metadata import version
from omegaconf import OmegaConf

from app.rag import (
    Embeddings, OpenRouter, SYSTEM, answer_scores, append_row, file_hash, index,
    open_collection, prepare_chunks, read_rows, retrieve, retrieval_scores, write_json,
)


ROOT = Path("data/experiments/semantic_threshold_sweep_100")
PREVIOUS = Path("data/experiments/langchain_semantic")
THRESHOLDS = (90, 95, 99)
SELECTION_RULE = "Highest development exact match, then token F1, then lower percentile"


def load(path):
    return json.loads(path.read_text())


def freeze(path, value):
    """Reject mismatched checkpoints instead of silently overwriting them."""
    if path.exists():
        if load(path) != value:
            raise ValueError(f"Checkpoint identity changed: {path}")
    else:
        write_json(path, value)


def choose(records):
    return min(records, key=lambda r: (-r["metrics"]["exact_match"],
                                      -r["metrics"]["token_f1"], r["threshold"]))


def ordered_questions(dataset, split, seed):
    return sorted(read_rows(dataset / f"{split}_questions.jsonl"), key=lambda q:
                  hashlib.sha256(f"{seed}:{q['question_id']}".encode()).hexdigest())


def score(rows, labels):
    """Recompute scores, including explicit zeroes for failed generations."""
    for row in rows:
        row.update(retrieval_scores(labels[row["question_id"]], row["chunks"]))
        row.update(answer_scores(row["answer"], labels[row["question_id"]]["reference_answers"])
                   if "answer" in row else {"exact_match": 0, "token_f1": 0})
    keys = ("exact_match", "token_f1", "reference_page_hit", "evidence_coverage",
            "complete_answer_span_hit", "retrieval_ms", "generation_ms")
    return {
        "questions": len(rows),
        "successful_answers": sum("answer" in r for r in rows),
        "generation_failures": sum("answer" not in r for r in rows),
        **{key: statistics.mean(r[key] for r in rows) for key in keys},
        "mean_context_chars": statistics.mean(len(r["context"]) for r in rows),
        "mean_retained_chunks": statistics.mean(len(r["chunks"]) for r in rows),
        "successful_answer_cost_usd": sum(r.get("usage", {}).get("cost") or 0 for r in rows),
    }


def usage(folder):
    calls = read_rows(folder / "api_calls.jsonl") if (folder / "api_calls.jsonl").exists() else []
    result = {
        "api_calls": len(calls),
        "reported_cost_usd": sum(c["usage"].get("cost") or 0 for c in calls),
        "calls_missing_cost": sum(c["usage"].get("cost") is None for c in calls),
    }
    write_json(folder / "usage.json", result)
    return result


def check_context(row, documents, budget):
    assert len(row["context"]) <= budget
    assert row["context"] == "\n\n".join(c["text"] for c in row["chunks"])
    for chunk in row["chunks"]:
        meta = chunk["metadata"]
        assert documents[meta["document_id"]][meta["start_char"]:meta["end_char"]] == chunk["text"]


def evaluate(cfg, collection, embeddings, folder, documents, reuse=None):
    folder.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg.dataset_dir)
    split = cfg.evaluation.split
    questions = ordered_questions(dataset, split, cfg.evaluation.seed)
    assert len(questions) == cfg.evaluation.limit
    manifest = {
        "question_ids": [q["question_id"] for q in questions],
        "questions_sha256": file_hash(dataset / f"{split}_questions.jsonl"),
        "labels_sha256": file_hash(dataset / f"{split}_labels.jsonl"),
        "collection": collection.name, "system_prompt": SYSTEM,
    }
    freeze(folder / "evaluation_manifest.json", manifest)
    freeze(folder / "config.json", OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, folder / "resolved_config.yaml")
    rows_path = folder / "retrieval.jsonl"
    if rows_path.exists():
        sources = read_rows(rows_path)
    else:
        before = usage(folder)["api_calls"]
        # Prefill avoids measuring cold query embedding calls as retrieval latency.
        embeddings.embed([cfg.embedding.query_prefix + q["question"] for q in questions])
        sources = [{**q, **retrieve(q["question"], collection, embeddings, cfg)} for q in questions]
        temporary = rows_path.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        for row in sources:
            append_row(temporary, row)
        temporary.replace(rows_path)
        print(f"{folder}: retrieved {len(sources)}; query preparation calls={usage(folder)['api_calls'] - before}", flush=True)
    assert [r["question_id"] for r in sources] == manifest["question_ids"]
    for q, row in zip(questions, sources):
        assert q["question"] == row["question"]
        check_context(row, documents, cfg.retrieval.max_context_chars)
    predictions_path = folder / "predictions.jsonl"
    if reuse is not None and not predictions_path.exists():
        previous_manifest = load(PREVIOUS / "semantic_retrieval/evaluation_manifest.json")
        assert previous_manifest == manifest
        previous_cfg = OmegaConf.load(PREVIOUS / "generation/resolved_config.yaml")
        for key in ("chunking", "retrieval", "generation", "evaluation"):
            assert cfg[key] == previous_cfg[key], key
        assert cfg.embedding.model == previous_cfg.embedding.model
        assert cfg.embedding.query_prefix == previous_cfg.embedding.query_prefix
        old_rows = load(reuse)
        assert [r["question_id"] for r in old_rows] == manifest["question_ids"]
        for old, source in zip(old_rows, sources):
            assert old["context"] == source["context"] and old["chunks"] == source["chunks"]
            assert old["question"] == source["question"]
            append_row(predictions_path, {**old, "retrieval_ms": source["retrieval_ms"], "reused_answer": True})
        write_json(folder / "reuse.json", {"path": str(reuse), "sha256": file_hash(reuse)})
    rows = read_rows(predictions_path) if predictions_path.exists() else []
    assert len({r["question_id"] for r in rows}) == len(rows)
    completed = {r["question_id"]: r for r in rows}
    assert set(completed) <= set(manifest["question_ids"])
    api = OpenRouter(cfg, folder)
    # Labels are loaded after retrieval; the index and generator never see them.
    labels = {r["question_id"]: r for r in read_rows(dataset / f"{split}_labels.jsonl")}
    try:
        for source in sources:
            if source["question_id"] in completed:
                old = completed[source["question_id"]]
                assert old["context"] == source["context"] and old["question"] == source["question"]
                continue
            row = dict(source)
            started = time.perf_counter()
            try:
                row.update(api.answer(row["question"], row["context"]))
            except RuntimeError as error:
                # Preserve provider refusals/empty outputs, but do not turn an
                # exhausted quota or transient service outage into quality scores.
                if "403" not in str(error) and "No answer returned" not in str(error):
                    raise
                row.update(generation_failed=True, generation_error=str(error))
            row["generation_ms"] = round((time.perf_counter() - started) * 1000, 2)
            score([row], labels)
            append_row(predictions_path, row)
            completed[row["question_id"]] = row
            write_json(folder / "progress.json", {"attempted": len(completed), "requested": len(questions)})
            print(f"{folder}: generated {len(completed)}/{len(questions)}", flush=True)
    finally:
        usage(folder)
    rows = [completed[qid] for qid in manifest["question_ids"]]
    result = score(rows, labels)
    write_json(folder / "summary.json", result)
    return result


def build(cfg, folder, documents):
    folder.mkdir(parents=True, exist_ok=True)
    freeze(folder / "config.json", OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, folder / "resolved_config.yaml")
    api = OpenRouter(cfg, folder)
    embeddings = Embeddings(cfg, api)
    try:
        chunks, report = prepare_chunks(cfg, embeddings)
        write_json(folder / "chunking.json", report)
        grouped = {}
        for chunk in chunks:
            grouped.setdefault(chunk["metadata"]["document_id"], []).append(chunk)
        assert set(grouped) == set(documents)
        for doc_id, parts in grouped.items():
            cursor = 0
            for chunk in parts:
                m = chunk["metadata"]
                assert m["start_char"] == cursor
                assert documents[doc_id][cursor:m["end_char"]] == chunk["text"]
                cursor = m["end_char"]
            assert cursor == len(documents[doc_id])
        print(f"{folder}: {len(chunks)} chunks; indexing/resuming", flush=True)
        result = index(cfg, chunks, report, embeddings)
        write_json(folder / "result.json", result)
        return open_collection(cfg, chunks, report), report
    finally:
        embeddings.db.close()
        usage(folder)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    base = OmegaConf.load(PREVIOUS / "generation/resolved_config.yaml")
    dataset = Path(base.dataset_dir)
    identity = {
        "thresholds": list(THRESHOLDS), "selection_rule": SELECTION_RULE,
        "base_config": OmegaConf.to_container(base, resolve=True),
        "corpus_sha256": file_hash(dataset / "corpus.jsonl"),
        "development_questions_sha256": file_hash(dataset / "development_questions.jsonl"),
        "development_labels_sha256": file_hash(dataset / "development_labels.jsonl"),
        "source_sha256": {str(p): file_hash(p) for p in
                          (Path(__file__), Path("src/app/rag.py"), Path("src/app/semantic.py"))},
        "library_version": version("langchain-experimental"), "system_prompt": SYSTEM,
        "score_policy": "All questions included; provider-blocked or empty answers score zero",
    }
    freeze(ROOT / "experiment_manifest.json", identity)
    documents = {r["document_id"]: r["text"] for r in read_rows(dataset / "corpus.jsonl")}
    records = []
    for threshold in THRESHOLDS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        cfg.chunking.breakpoint_threshold_amount = threshold
        build_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        build_cfg.embedding.batch_size = 64
        build_cfg.api.interval_seconds = 3
        OmegaConf.update(build_cfg, "embedding.encoding_format", "base64", force_add=True)
        OmegaConf.update(build_cfg, "embedding.max_batch_characters", 80000, force_add=True)
        collection, chunking = build(build_cfg, ROOT / f"threshold_{threshold}/index", documents)
        folder = ROOT / f"threshold_{threshold}/development"
        folder.mkdir(parents=True, exist_ok=True)
        embeddings = Embeddings(cfg, OpenRouter(cfg, folder))
        try:
            metrics = evaluate(cfg, collection, embeddings, folder, documents,
                               PREVIOUS / "predictions_complete.json" if threshold == 95 else None)
        finally:
            embeddings.db.close()
        records.append({"threshold": threshold, "chunks": chunking["chunks"], "metrics": metrics})
        write_json(ROOT / "development_comparison.json", {"selection_rule": SELECTION_RULE, "results": records})
    winner = choose(records)
    # Freeze before opening any held-out labels or generating held-out answers.
    selected = {"threshold": winner["threshold"], "selection_rule": SELECTION_RULE,
                "development_results": records, "held_out_started": False}
    freeze(ROOT / "selection.json", selected)
    print(f"SELECTED percentile {winner['threshold']} from development only", flush=True)
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.chunking.breakpoint_threshold_amount = winner["threshold"]
    cfg.evaluation.split = "held_out"
    cfg.evaluation.limit = 400
    dev_questions = ordered_questions(dataset, "development", cfg.evaluation.seed)
    test_questions = ordered_questions(dataset, "held_out", cfg.evaluation.seed)
    assert {q["question_id"] for q in dev_questions}.isdisjoint(q["question_id"] for q in test_questions)
    report = load(ROOT / f"threshold_{winner['threshold']}/index/chunking.json")
    chunks = read_rows(Path(report["path"]))
    collection = open_collection(cfg, chunks, report)
    folder = ROOT / "held_out"
    folder.mkdir(exist_ok=True)
    embeddings = Embeddings(cfg, OpenRouter(cfg, folder))
    try:
        held_out = evaluate(cfg, collection, embeddings, folder, documents)
    finally:
        embeddings.db.close()
    write_json(ROOT / "result.json", {"selected_threshold": winner["threshold"],
                                     "development": records, "held_out": held_out})
    print(json.dumps(load(ROOT / "result.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()

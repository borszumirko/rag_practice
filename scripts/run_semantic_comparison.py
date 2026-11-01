"""Resume native semantic development generation and compare saved recursive results.

Run from the repository root. Reuses validated saved contexts and answers;
unfinished questions incur Gemini charges. Failed generations count as zero in
all-question answer scores and remain explicitly marked as failures.
"""

from pathlib import Path
import statistics
import time

from omegaconf import OmegaConf

from app.rag import (
    OpenRouter, SYSTEM, answer_scores, append_row, file_hash, read_rows,
    retrieval_scores, write_json,
)


ROOT = Path("data/experiments/langchain_semantic")


def load(path):
    import json
    return json.loads(path.read_text())


def main():
    manifest = load(ROOT / "experiment_manifest.json")
    baseline = Path(manifest["baseline"])
    output = ROOT / "generation_resumed"
    output.mkdir(exist_ok=True)
    cfg = OmegaConf.load(ROOT / "generation/resolved_config.yaml")
    expected = load(ROOT / "semantic_retrieval/evaluation_manifest.json")
    for folder in (baseline, ROOT / "recursive_retrieval", ROOT / "generation"):
        other = load(folder / "evaluation_manifest.json")
        for key in ("question_ids", "questions_sha256", "labels_sha256", "system_prompt"):
            assert other[key] == expected[key], (folder, key)
    assert expected["system_prompt"] == SYSTEM
    dataset = Path(cfg.dataset_dir)
    for kind in ("questions", "labels"):
        assert file_hash(dataset / f"development_{kind}.jsonl") == expected[f"{kind}_sha256"]
    base_cfg = OmegaConf.load(baseline / "resolved_config.yaml")
    for key in ("generation", "retrieval", "evaluation"):
        assert cfg[key] == base_cfg[key], key
    assert cfg.embedding.model == base_cfg.embedding.model
    assert cfg.embedding.query_prefix == base_cfg.embedding.query_prefix
    chunk_report = load(ROOT / "index/chunking.json")
    assert file_hash(dataset / "corpus.jsonl") == chunk_report["corpus_sha256"]
    assert file_hash(Path(chunk_report["path"])) == chunk_report["chunks_sha256"]
    assert dict(cfg.chunking) == chunk_report["chunking"]
    from app.rag import open_collection
    chunks = read_rows(Path(chunk_report["path"]))
    assert open_collection(cfg, chunks, chunk_report).count() == len(chunks)
    documents = {r["document_id"]: r["text"] for r in read_rows(dataset / "corpus.jsonl")}
    for doc_id, text in documents.items():
        parts = sorted((c for c in chunks if c["metadata"]["document_id"] == doc_id),
                       key=lambda c: c["metadata"]["start_char"])
        cursor = 0
        for c in parts:
            m = c["metadata"]
            assert m["start_char"] == cursor
            assert text[cursor:m["end_char"]] == c["text"]
            cursor = m["end_char"]
        assert cursor == len(text)
    semantic = read_rows(ROOT / "semantic_retrieval/predictions.jsonl")
    recursive = read_rows(baseline / "predictions.jsonl")
    control = read_rows(ROOT / "recursive_retrieval/predictions.jsonl")
    for rows in (semantic, recursive, control):
        assert [r["question_id"] for r in rows] == expected["question_ids"]
    assert all(a["context"] == b["context"] and a["chunks"] == b["chunks"]
               for a, b in zip(recursive, control))
    labels = {r["question_id"]: r for r in read_rows(dataset / "development_labels.jsonl")}
    questions = {r["question_id"]: r for r in read_rows(dataset / "development_questions.jsonl")}
    for row in semantic + recursive:
        assert row["question"] == questions[row["question_id"]]["question"]
        assert row["context"] == "\n\n".join(c["text"] for c in row["chunks"])
        for c in row["chunks"]:
            m = c["metadata"]
            assert documents[m["document_id"]][m["start_char"]:m["end_char"]] == c["text"]
        assert all(row[k] == v for k, v in retrieval_scores(labels[row["question_id"]], row["chunks"]).items())
    saved = read_rows(ROOT / "generation/predictions.jsonl")
    if (output / "predictions.jsonl").exists():
        saved += read_rows(output / "predictions.jsonl")
    completed = {r["question_id"]: r for r in saved}
    assert len(completed) == len(saved)
    for row in semantic:
        if row["question_id"] in completed:
            old = completed[row["question_id"]]
            assert old["context"] == row["context"] and old["question"] == row["question"]
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    write_json(output / "evaluation_manifest.json", expected)
    write_json(output / "source_checksums.json", {
        str(p): file_hash(p) for p in (Path(__file__), Path("src/app/rag.py"), Path("src/app/semantic.py"))
    })
    api = OpenRouter(cfg, output)
    for source in semantic:
        if source["question_id"] in completed:
            continue
        row = dict(source)
        started = time.perf_counter()
        try:
            row.update(api.answer(row["question"], row["context"]))
        except RuntimeError as error:
            row.update(generation_failed=True, generation_error=str(error))
        row["generation_ms"] = round((time.perf_counter() - started) * 1000, 2)
        append_row(output / "predictions.jsonl", row)
        completed[row["question_id"]] = row
        print(f"Attempted {len(completed)}/100; answer={not row.get('generation_failed', False)}", flush=True)
    semantic_answers = [completed[qid] for qid in expected["question_ids"]]
    for row in semantic_answers + recursive:
        row.update(answer_scores(row["answer"], labels[row["question_id"]]["reference_answers"])
                   if "answer" in row else {"exact_match": 0, "token_f1": 0})
    write_json(ROOT / "predictions_complete.json", semantic_answers)
    def metrics(rows, retrieval_rows):
        return {
            "questions": len(rows),
            "successful_answers": sum("answer" in r for r in rows),
            "generation_failures": sum("answer" not in r for r in rows),
            **{key: statistics.mean(r[key] for r in rows) for key in
               ("exact_match", "token_f1", "complete_answer_span_hit", "evidence_coverage", "reference_page_hit")},
            "mean_context_chars": statistics.mean(len(r["context"]) for r in rows),
            "mean_retained_chunks": statistics.mean(len(r["chunks"]) for r in rows),
            "mean_cached_retrieval_ms": statistics.mean(r["retrieval_ms"] for r in retrieval_rows),
            "mean_generation_ms": statistics.mean(r["generation_ms"] for r in rows),
            "successful_answer_cost_usd": sum(r.get("usage", {}).get("cost") or 0 for r in rows),
            "failures": [{"question_id": r["question_id"], "error": r.get("generation_error")} for r in rows if "answer" not in r],
        }
    calls = [c for p in ROOT.glob("*/api_calls.jsonl") for c in read_rows(p)]
    assert len({c["id"] for c in calls}) == len(calls)
    usage = {}
    for endpoint in ("embeddings", "chat/completions"):
        entries = [c for c in calls if c["endpoint"] == endpoint]
        usage[endpoint] = {
            "calls": len(entries),
            "reported_cost_usd": sum(c["usage"].get("cost") or 0 for c in entries),
            "calls_missing_cost": sum(c["usage"].get("cost") is None for c in entries),
        }
    report = {
        "split": "development", "held_out_evaluated": False,
        "recursive": metrics(recursive, control),
        "semantic": metrics(semantic_answers, semantic),
        "recursive_chunks": load(baseline / "chunking.json")["chunks"],
        "semantic_chunks": chunk_report["chunks"],
        "semantic_max_chunk_chars": chunk_report["max_chunk_characters"],
        "paired_exact_match_gains": [s["question_id"] for s, r in zip(semantic_answers, recursive) if s["exact_match"] > r["exact_match"]],
        "paired_exact_match_losses": [s["question_id"] for s, r in zip(semantic_answers, recursive) if s["exact_match"] < r["exact_match"]],
        "usage_all_native_semantic_attempts": usage,
        "verification": "Matching question IDs, dataset checksums, generation/retrieval settings, system prompt, recursive control contexts, complete semantic index, exact corpus partition and rescored predictions verified.",
        "answer_score_policy": "All 100 questions; generation failures score zero. Existing successful answers reused; baseline from saved development sweep.",
    }
    write_json(ROOT / "comparison.json", report)
    write_json(output / "summary.json", report["semantic"])
    print(report, flush=True)


if __name__ == "__main__":
    main()

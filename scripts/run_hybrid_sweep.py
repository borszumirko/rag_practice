"""Tune only dense fusion weight on development for two frozen chunkers.

Run from the repository root: uv run python scripts/run_hybrid_sweep.py
Resumes completed answers; never opens held-out questions or labels.
"""

import json
from pathlib import Path
import statistics
import time

from omegaconf import OmegaConf

from app.hybrid import HybridRetriever
from app.rag import (Embeddings, OpenRouter, SYSTEM, append_row, file_hash,
                     open_collection, prepare_chunks, read_rows, retrieve, retrieval_scores, write_json)
from run_semantic_threshold_sweep import (check_context, freeze, ordered_questions,
                                          score, usage)


ROOT = Path("data/experiments/hybrid_weight_sweep_100")
WEIGHTS = (0.25, 0.5, 0.75)
BASELINES = {
    "recursive": Path("data/experiments/chunking_sweep_100/generation_2000_200_attempt1"),
    "semantic": Path("data/experiments/semantic_threshold_sweep_100/threshold_99/development"),
}
RULE = "Highest development exact match, then token F1, then higher dense weight"


def load(path):
    return json.loads(path.read_text())


def paired(rows, baseline):
    old = {r["question_id"]: r for r in baseline}
    return {metric: {
        "gained": sum(r[metric] > old[r["question_id"]][metric] for r in rows),
        "lost": sum(r[metric] < old[r["question_id"]][metric] for r in rows),
    } for metric in ("exact_match", "complete_answer_span_hit")}


def terminal_generation_failure(error, folder):
    """Only score explicit content refusals; quota/auth failures must stop."""
    if "No answer returned" in str(error) or "PROHIBITED_CONTENT" in str(error):
        return True
    errors = folder / "api_errors.jsonl"
    if "403" in str(error) and errors.exists():
        body = read_rows(errors)[-1].get("body", "").lower()
        return "prohibited_content" in body or "content_filter" in body
    return False


def generate(cfg, folder, sources, labels, documents):
    folder.mkdir(parents=True, exist_ok=True)
    freeze(folder / "config.json", OmegaConf.to_container(cfg, resolve=True))
    freeze(folder / "retrieval.json", sources)
    path = folder / "predictions.jsonl"
    saved = read_rows(path) if path.exists() else []
    assert len({r["question_id"] for r in saved}) == len(saved)
    completed = {r["question_id"]: r for r in saved}
    assert set(completed) <= {r["question_id"] for r in sources}
    api = OpenRouter(cfg, folder)
    try:
        for source in sources:
            qid = source["question_id"]
            if qid in completed:
                assert all(completed[qid][k] == source[k] for k in source)
                continue
            row = dict(source)
            started = time.perf_counter()
            try:
                row.update(api.answer(row["question"], row["context"]))
            except RuntimeError as error:
                if not terminal_generation_failure(error, folder):
                    raise
                row.update(generation_failed=True, generation_error=str(error))
            row["generation_ms"] = round((time.perf_counter() - started) * 1000, 2)
            check_context(row, documents, cfg.retrieval.max_context_chars)
            score([row], labels)
            append_row(path, row)
            completed[qid] = row
            print(f"{folder}: generated {len(completed)}/{len(sources)}", flush=True)
    finally:
        usage(folder)
    rows = [completed[r["question_id"]] for r in sources]
    metrics = score(rows, labels)
    write_json(folder / "summary.json", metrics)
    return rows, metrics


def prepare(name, previous):
    cfg = OmegaConf.load(previous / "resolved_config.yaml")
    assert cfg.evaluation.split == "development" and cfg.evaluation.limit == 100
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg.dataset_dir)
    questions = ordered_questions(dataset, "development", cfg.evaluation.seed)
    documents = {r["document_id"]: r["text"] for r in read_rows(dataset / "corpus.jsonl")}
    freeze(folder / "manifest.json", {
        "weights": list(WEIGHTS), "candidate_depth_per_retriever": 50,
        "rrf_constant": 60, "bm25_k1": 1.5, "bm25_b": 0.75,
        "tokenization": "Unicode word tokens, casefold, no stemming or stopword removal",
        "question_ids": [q["question_id"] for q in questions],
        "dataset_sha256": {p: file_hash(dataset / p) for p in
                           ("corpus.jsonl", "development_questions.jsonl", "development_labels.jsonl")},
        "baseline_path": str(previous), "system_prompt": SYSTEM,
        "baseline_config": OmegaConf.to_container(cfg, resolve=True),
        "selection_rule": RULE,
        "source_sha256": {str(p): file_hash(p) for p in
                          (Path(__file__), Path("src/app/hybrid.py"), Path("src/app/rag.py"),
                           Path("scripts/run_semantic_threshold_sweep.py"), Path("src/app/semantic.py"))},
    })
    embeddings = Embeddings(cfg, OpenRouter(cfg, folder))
    try:
        chunks, report = prepare_chunks(cfg)
        collection = open_collection(cfg, chunks, report)
        write_json(folder / "chunking.json", report)
        hybrid = HybridRetriever(chunks, collection, embeddings, cfg)
        candidates_path = folder / "candidates.json"
        dense_path = folder / "dense_control.json"
        if candidates_path.exists():
            candidates = load(candidates_path)
        else:
            embeddings.embed([cfg.embedding.query_prefix + q["question"] for q in questions])
            candidates = [{**q, **hybrid.candidates(q["question"])} for q in questions]
            freeze(candidates_path, candidates)
        assert [r["question_id"] for r in candidates] == [q["question_id"] for q in questions]
        if dense_path.exists():
            dense = load(dense_path)
        else:
            dense = [{**q, **retrieve(q["question"], collection, embeddings, cfg)} for q in questions]
            freeze(dense_path, dense)
        baseline = read_rows(previous / "predictions.jsonl")
        previous_manifest = load(previous / "evaluation_manifest.json")
        assert previous_manifest["system_prompt"] == SYSTEM
        assert previous_manifest["collection"] == collection.name
        for key, filename in (("questions_sha256", "development_questions.jsonl"),
                              ("labels_sha256", "development_labels.jsonl")):
            assert previous_manifest[key] == file_hash(dataset / filename)
        assert [r["question_id"] for r in baseline] == [q["question_id"] for q in questions]
        baseline_folder = folder / "dense_baseline"
        baseline_folder.mkdir(exist_ok=True)
        baseline_path = baseline_folder / "predictions.jsonl"
        changed = []
        reusable = []
        for old, fresh in zip(baseline, dense):
            assert old["question"] == fresh["question"]
            check_context(fresh, documents, cfg.retrieval.max_context_chars)
            if old["context"] == fresh["context"]:
                reusable.append({**old, **fresh, "reused_answer": True})
            else:
                changed.append(fresh["question_id"])
        # ANN search can differ across process/index reopenings. Reuse answers
        # only for byte-identical contexts; generate any changed controls once.
        if not baseline_path.exists():
            for row in reusable:
                append_row(baseline_path, row)
        write_json(folder / "baseline_reuse.json", {
            "source": str(previous), "predictions_sha256": file_hash(previous / "predictions.jsonl"),
            "reused_answers": len(reusable), "changed_context_question_ids": changed})
        labels = {r["question_id"]: r for r in read_rows(dataset / "development_labels.jsonl")}
        baseline, baseline_metrics = generate(cfg, baseline_folder, dense, labels, documents)
        write_json(folder / "baseline_summary.json", baseline_metrics)
        prepared = []
        for weight in WEIGHTS:
            run = folder / f"dense_weight_{weight}"
            run.mkdir(exist_ok=True)
            sources_path = run / "retrieval.json"
            if sources_path.exists():
                sources = load(sources_path)
            else:
                sources = [{"question_id": c["question_id"], "question": c["question"],
                            **hybrid.retrieve(c, weight)} for c in candidates]
                freeze(sources_path, sources)
            for q, source in zip(questions, sources):
                assert q["question_id"] == source["question_id"] and q["question"] == source["question"]
                check_context(source, documents, cfg.retrieval.max_context_chars)
            retrieval_metrics = {
                key: statistics.mean(retrieval_scores(
                    labels[r["question_id"]], r["chunks"])[key] for r in sources)
                for key in ("reference_page_hit", "complete_answer_span_hit", "evidence_coverage")}
            write_json(run / "retrieval_summary.json", retrieval_metrics)
            print(f"{name} weight={weight}: {retrieval_metrics}", flush=True)
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.update(run_cfg, "hybrid", {"dense_weight": weight, "candidate_depth": 50,
                             "rrf_constant": 60, "bm25_k1": 1.5, "bm25_b": 0.75}, force_add=True)
            prepared.append((weight, run_cfg, run, sources))
        return prepared, labels, documents, baseline, baseline_metrics
    finally:
        embeddings.db.close()
        usage(folder)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Finish all retrieval diagnostics before making generation calls.
    prepared = {name: prepare(name, previous) for name, previous in BASELINES.items()}
    results = {}
    for name, (runs, labels, documents, baseline, baseline_metrics) in prepared.items():
        records = []
        for weight, cfg, folder, sources in runs:
            rows, metrics = generate(cfg, folder, sources, labels, documents)
            records.append({"dense_weight": weight, "metrics": metrics,
                            "paired_vs_dense": paired(rows, baseline)})
        winner = max(records, key=lambda r: (r["metrics"]["exact_match"],
                                            r["metrics"]["token_f1"], r["dense_weight"]))
        selection = {"selection_rule": RULE, "selected_dense_weight": winner["dense_weight"],
                     "baseline_metrics": baseline_metrics, "development_results": records,
                     "held_out_evaluated": False}
        freeze(ROOT / name / "selection.json", selection)
        results[name] = selection
        write_json(ROOT / "comparison.json", results)
    calls = [c for p in ROOT.rglob("api_calls.jsonl") for c in read_rows(p)]
    write_json(ROOT / "usage.json", {"api_calls": len(calls),
               "reported_cost_usd": sum(c["usage"].get("cost") or 0 for c in calls),
               "calls_missing_cost": sum(c["usage"].get("cost") is None for c in calls)})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

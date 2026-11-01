"""Evaluate development-selected hybrids on all 400 held-out questions.

Run from the repository root. Exact-input baseline answers may be reused;
missing answers incur API charges. Never selects parameters using test scores.
"""

import json
from pathlib import Path
import statistics

from omegaconf import OmegaConf

from app.hybrid import HybridRetriever
from app.rag import (Embeddings, OpenRouter, SYSTEM, append_row, file_hash,
                     open_collection, prepare_chunks, read_rows, retrieval_scores, write_json)
from run_hybrid_sweep import generate, paired
from run_semantic_threshold_sweep import check_context, freeze, ordered_questions, score, usage


ROOT = Path("data/experiments/hybrid_held_out_400")
DEVELOPMENT = Path("data/experiments/hybrid_weight_sweep_100")
BASELINES = {
    "recursive": Path("data/experiments/held_out_2000_200/generation"),
    "semantic": Path("data/experiments/semantic_threshold_sweep_100/held_out"),
}


def load(path):
    return json.loads(path.read_text())


def prepare(name, previous):
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    selection_path = DEVELOPMENT / name / "selection.json"
    selection = load(selection_path)
    weight = selection["selected_dense_weight"]
    winner = max(selection["development_results"], key=lambda r:
                 (r["metrics"]["exact_match"], r["metrics"]["token_f1"], r["dense_weight"]))
    assert winner["dense_weight"] == weight
    cfg = OmegaConf.create(load(DEVELOPMENT / name / f"dense_weight_{weight}/config.json"))
    assert cfg.hybrid.dense_weight == weight
    assert cfg.hybrid.candidate_depth == 50 and cfg.hybrid.rrf_constant == 60
    cfg.evaluation.split, cfg.evaluation.limit = "held_out", 400
    cfg.command = "evaluate"
    baseline_cfg = OmegaConf.load(previous / "resolved_config.yaml")
    for key in ("chunking", "retrieval", "generation"):
        assert cfg[key] == baseline_cfg[key], key
    for key in ("model", "query_prefix"):
        assert cfg.embedding[key] == baseline_cfg.embedding[key], key
    dataset = Path(cfg.dataset_dir)
    # Freeze the development selection before opening held-out labels.
    freeze(folder / "selection.json", {
        "development_selection_path": str(selection_path),
        "development_selection_sha256": file_hash(selection_path),
        "selected_dense_weight": weight, "selection_rule": selection["selection_rule"],
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "source_sha256": {str(p): file_hash(p) for p in (
            Path(__file__), Path("src/app/hybrid.py"), Path("src/app/rag.py"),
            Path("src/app/semantic.py"), Path("scripts/run_hybrid_sweep.py"),
            Path("scripts/run_semantic_threshold_sweep.py"))},
    })
    questions = ordered_questions(dataset, "held_out", cfg.evaluation.seed)
    development = ordered_questions(dataset, "development", cfg.evaluation.seed)
    assert len(questions) == 400
    assert {q["question_id"] for q in questions}.isdisjoint(q["question_id"] for q in development)
    documents = {r["document_id"]: r["text"] for r in read_rows(dataset / "corpus.jsonl")}
    chunks, report = prepare_chunks(cfg)
    collection = open_collection(cfg, chunks, report)
    write_json(folder / "chunking.json", report)
    manifest = {
        "question_ids": [q["question_id"] for q in questions],
        "questions_sha256": file_hash(dataset / "held_out_questions.jsonl"),
        "labels_sha256": file_hash(dataset / "held_out_labels.jsonl"),
        "corpus_sha256": file_hash(dataset / "corpus.jsonl"),
        "collection": collection.name, "system_prompt": SYSTEM,
    }
    freeze(folder / "evaluation_manifest.json", manifest)
    previous_manifest = load(previous / "evaluation_manifest.json")
    for key in ("question_ids", "questions_sha256", "labels_sha256", "collection", "system_prompt"):
        assert previous_manifest[key] == manifest[key], key
    baseline = read_rows(previous / "predictions.jsonl")
    assert [r["question_id"] for r in baseline] == manifest["question_ids"]
    path = folder / "retrieval.json"
    if path.exists():
        sources = load(path)
    else:
        embeddings = Embeddings(cfg, OpenRouter(cfg, folder))
        try:
            embeddings.embed([cfg.embedding.query_prefix + q["question"] for q in questions])
            hybrid = HybridRetriever(chunks, collection, embeddings, cfg,
                                     cfg.hybrid.candidate_depth, cfg.hybrid.rrf_constant)
            candidates = [{**q, **hybrid.candidates(q["question"])} for q in questions]
            freeze(folder / "candidates.json", candidates)
            sources = [{"question_id": c["question_id"], "question": c["question"],
                        **hybrid.retrieve(c, weight)} for c in candidates]
            freeze(path, sources)
        finally:
            embeddings.db.close()
            usage(folder)
    assert len(sources) == 400
    assert [r["question_id"] for r in sources] == manifest["question_ids"]
    reusable = []
    for q, source, old in zip(questions, sources, baseline):
        assert source["question"] == old["question"] == q["question"]
        check_context(source, documents, cfg.retrieval.max_context_chars)
        check_context(old, documents, cfg.retrieval.max_context_chars)
        if source["context"] == old["context"]:
            reusable.append({**old, **source, "reused_answer": True})
    prediction_path = folder / "predictions.jsonl"
    if not prediction_path.exists():
        for row in reusable:
            append_row(prediction_path, row)
    freeze(folder / "reuse.json", {"baseline": str(previous),
            "predictions_sha256": file_hash(previous / "predictions.jsonl"),
            "reused_answers": len(reusable),
            "question_ids": [r["question_id"] for r in reusable]})
    labels = {r["question_id"]: r for r in read_rows(dataset / "held_out_labels.jsonl")}
    baseline_metrics = score(baseline, labels)
    write_json(folder / "baseline_summary.json", baseline_metrics)
    retrieval = {k: statistics.mean(retrieval_scores(labels[r["question_id"]], r["chunks"])[k]
                 for r in sources) for k in
                 ("reference_page_hit", "complete_answer_span_hit", "evidence_coverage")}
    write_json(folder / "retrieval_summary.json", retrieval)
    print(f"{name}: weight={weight}; reused={len(reusable)}/400; retrieval={retrieval}", flush=True)
    return cfg, folder, sources, labels, documents, baseline, baseline_metrics


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = {name: prepare(name, previous) for name, previous in BASELINES.items()}
    results = {}
    for name, (cfg, folder, sources, labels, documents, baseline, baseline_metrics) in prepared.items():
        rows, metrics = generate(cfg, folder, sources, labels, documents)
        results[name] = {"dense_weight": cfg.hybrid.dense_weight, "hybrid_metrics": metrics,
                         "saved_dense_metrics": baseline_metrics,
                         "paired_vs_saved_dense": paired(rows, baseline),
                         "reused_answers": load(folder / "reuse.json")["reused_answers"]}
        write_json(ROOT / "comparison.json", results)
    calls = [c for p in ROOT.rglob("api_calls.jsonl") for c in read_rows(p)]
    write_json(ROOT / "usage.json", {"api_calls": len(calls),
        "reported_cost_usd": sum(c["usage"].get("cost") or 0 for c in calls),
        "calls_missing_cost": sum(c["usage"].get("cost") is None for c in calls)})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Run the fixed 100-question development sweep, then generate for its top two.

Uses the existing Hydra runner and cache, with one index writer at a time.
Completed steps are reused; interrupted steps get separate attempt directories.
Run from the repository root with the project's Python environment.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


CONFIGURATIONS = [(1000, 200), (1000, 100), (2000, 100), (2000, 200)]


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_step(root, name, size, overlap, command, generate=False):
    for attempt in range(1, 100):
        folder = root / f"{name}_{size}_{overlap}_attempt{attempt}"
        if (folder / "result.json").exists():
            print(f"Reusing {folder}", flush=True)
            return folder
        if not folder.exists():
            break
    else:
        raise RuntimeError("Too many interrupted attempts")
    folder.mkdir(parents=True)
    args = [sys.executable, "-m", "app.rag", f"command={command}",
            f"chunking.chunk_size={size}", f"chunking.chunk_overlap={overlap}",
            "evaluation.split=development", "evaluation.limit=100", "evaluation.seed=42",
            f"evaluation.generate={str(generate).lower()}", f"hydra.run.dir={folder}"]
    if command == "index" and size == 2000:
        # Reduce payloads after observed token-quota errors with larger chunks.
        args.extend(["embedding.batch_size=64", "api.interval_seconds=3"])
    save(folder / "invocation.json", args)
    print(f"Starting {folder}", flush=True)
    started = time.perf_counter()
    with (folder / "console.log").open("w") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
    save(folder / "execution.json", {"wall_seconds": time.perf_counter() - started,
                                    "returncode": result.returncode})
    if result.returncode:
        raise RuntimeError(f"Step failed; inspect {folder / 'console.log'}")
    print(f"Completed {folder}", flush=True)
    return folder


def distribution(values):
    ordered = sorted(values)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": ordered[(95 * len(ordered) + 99) // 100 - 1]}


def index_attempts(root, size, overlap):
    attempts = sorted(root.glob(f"index_{size}_{overlap}_attempt*"))
    # Read durable response logs directly, including attempts stopped before the
    # Hydra runner could write its final usage summary.
    calls = [call for p in attempts if (p / "api_calls.jsonl").exists()
             for call in read_rows(p / "api_calls.jsonl")]
    return {
        "index_attempts": [str(p) for p in attempts],
        "index_usage_this_sweep": {
            "api_calls": len(calls),
            "reported_cost_usd": sum(c["usage"].get("cost") or 0 for c in calls),
            "calls_missing_cost": sum(c["usage"].get("cost") is None for c in calls),
            "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
        },
        "index_wall_seconds": sum(read_json(p / "execution.json")["wall_seconds"]
                                  for p in attempts if (p / "execution.json").exists()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/chunking_sweep_100"))
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_step(root, "index", 1000, 200, "index")
    # Warm all query embeddings before collecting comparable retrieval latency.
    warmup = run_step(root, "warmup", 1000, 200, "evaluate")
    records = []
    expected_manifest = read_json(warmup / "evaluation_manifest.json")
    for size, overlap in CONFIGURATIONS:
        indexed = run_step(root, "index", size, overlap, "index")
        retrieved = run_step(root, "retrieval", size, overlap, "evaluate")
        manifest = read_json(retrieved / "evaluation_manifest.json")
        for key in ("question_ids", "questions_sha256", "labels_sha256", "system_prompt"):
            if manifest[key] != expected_manifest[key]:
                raise ValueError(f"Mismatched {key} in {retrieved}")
        rows = read_rows(retrieved / "predictions.jsonl")
        if len(rows) != 100 or [r["question_id"] for r in rows] != manifest["question_ids"]:
            raise ValueError("Incomplete or reordered retrieval results")
        usage = read_json(retrieved / "usage.json")
        if usage["api_calls"]:
            raise ValueError("Retrieval timing must use cached query embeddings")
        records.append({
            "chunk_size": size, "chunk_overlap": overlap,
            "index_run": str(indexed), "retrieval_run": str(retrieved),
            "chunks": read_json(indexed / "chunking.json")["chunks"],
            **index_attempts(root, size, overlap),
            "retrieval": read_json(retrieved / "summary.json"),
            "cached_retrieval_ms": distribution([r["retrieval_ms"] for r in rows]),
            "context_characters": distribution([len(r["context"]) for r in rows]),
        })
    ranked = sorted(records, key=lambda r: (-r["retrieval"]["complete_answer_span_hit"],
                                           -r["retrieval"]["evidence_coverage"], r["chunks"]))
    report = {"warmup_run": str(warmup), "question_ids": expected_manifest["question_ids"],
              "selection_rule": "Complete answer-span hit descending, evidence coverage descending, chunk count ascending",
              "configurations": ranked, "selected": [[r["chunk_size"], r["chunk_overlap"]] for r in ranked[:2]]}
    save(root / "comparison.json", report)
    if not args.retrieval_only:
        for record in ranked[:2]:
            generated = run_step(root, "generation", record["chunk_size"], record["chunk_overlap"],
                                 "evaluate", generate=True)
            retrieval_manifest = read_json(Path(record["retrieval_run"]) / "evaluation_manifest.json")
            if read_json(generated / "evaluation_manifest.json") != retrieval_manifest:
                raise ValueError("Generation manifest differs from retrieval manifest")
            before = read_rows(Path(record["retrieval_run"]) / "predictions.jsonl")
            after = read_rows(generated / "predictions.jsonl")
            if len(after) != 100 or any(a["question_id"] != b["question_id"] or
                                      a["context"] != b["context"] for a, b in zip(before, after)):
                raise ValueError("Generation questions or contexts differ from retrieval sweep")
            record.update(generation_run=str(generated), generation=read_json(generated / "summary.json"),
                          generation_usage=read_json(generated / "usage.json"))
            save(root / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

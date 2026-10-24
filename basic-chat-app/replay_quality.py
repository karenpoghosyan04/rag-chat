"""Replay the EXISTING benchmark to detect regressions, not tune or claim a new test.

Uses cached local models only, a private in-memory index, and frozen embeddings.
It never opens the app's uploaded-document database or calls a hosted LLM.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import app
import rag_eval
from check_quality import check

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "reports/retrieval-baseline.json"


def assert_no_regression(baseline, current):
    """Gate each question, not just an aggregate that could hide new misses."""
    if [r["question"] for r in baseline["details"]] != [r["question"] for r in current["details"]]:
        raise ValueError("Benchmark questions changed; reports cannot be compared.")
    failures = []
    for old, new in zip(baseline["details"], current["details"]):
        previous = old["first_relevant_rank"]
        now = new["first_relevant_rank"]
        if previous is not None and (now is None or now > previous):
            failures.append(old["question"])
        if old["unanswerable_outcome"] == "correct_rejection" and new["unanswerable_outcome"] != "correct_rejection":
            failures.append(old["question"])
    for name, value in baseline["metrics"].items():
        if name.startswith(("recall@", "precision@", "mrr@")) or name == "no_answer_accuracy":
            if value is not None and (current["metrics"].get(name) is None or current["metrics"][name] + 1e-9 < value):
                failures.append(name)
    if failures:
        raise ValueError("Retrieval regression: " + "; ".join(failures))


def run(capture=False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["LOCAL_EMBEDDING_MODEL"] = "google/embeddinggemma-2"
    check()
    chunks = [app.DocumentChunk(**c) for c in json.loads((ROOT / "embedding-eval/frozen-corpus.json").read_text())]
    vectors = np.load(ROOT / "embedding-eval/new-model.index.npy", allow_pickle=False)
    cases = rag_eval.load_cases(ROOT / "eval_questions.jsonl")
    index = app.DocumentIndex()
    index.replace("frozen-corpus", chunks, embeddings=vectors.tolist(), embedding_model="google/embeddinggemma-2")
    # Frozen chunks/vectors must be replayed with their recorded configuration.
    # Newly tuned defaults are evaluated against freshly chunked PDFs separately.
    settings = app.RagSettings.from_dict(json.loads(BASELINE.read_text())["settings"]) if not capture else app.DEFAULT_SETTINGS
    report = rag_eval.evaluate_cases(index, cases, settings, [1, 3, 5], embed_query=lambda q: app.embed_texts([q], kind="query")[0])
    report["scope"] = "Regression replay of previously evaluated data; not a fresh final test or answer-generation evaluation."
    report["settings"] = settings.to_dict()
    # The evaluator's historical production_gate name is too strong for one document.
    report["benchmark_thresholds"] = report.pop("production_gate")
    destination = BASELINE if capture else ROOT / "reports/retrieval-current.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if not capture:
        assert_no_regression(json.loads(BASELINE.read_text()), report)
        print("PASS: no measured retrieval regression on the existing 41 questions.")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Explicitly capture/replace the measured baseline.")
    run(parser.parse_args().capture)

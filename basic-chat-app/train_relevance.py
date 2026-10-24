"""Train two offline passage classifiers without changing live retrieval.

Labels mark exact known evidence, not exhaustively adjudicated relevance.
Unmatched passages are weak negatives. Results are development diagnostics only.
The nine previously held-out questions are excluded, never scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app
import rag_eval
from src.relevance import PairFeatures

ROOT = Path(__file__).resolve().parent


def question_split(cases, excluded, seed=42):
    """Split questions before expanding them into passage pairs."""
    development = [i for i in range(len(cases)) if i + 1 not in excluded]
    train, validation = train_test_split(
        development, test_size=0.25, random_state=seed,
        stratify=[bool(cases[i].relevant) for i in development],
    )
    return sorted(train), sorted(validation)


def make_pairs(cases, chunks, ids):
    pairs, labels, groups = [], [], []
    for i in ids:
        for chunk in chunks:
            pairs.append((cases[i].question, chunk.text))
            labels.append(int(any(rag_eval.evidence_matches(chunk, e) for e in cases[i].relevant)))
            groups.append(i)
    return pairs, np.asarray(labels), np.asarray(groups)


def metrics(labels, scores, groups):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, scores >= 0.5, average="binary", zero_division=0,
    )
    hit = []
    for group in sorted(set(groups)):
        mask = groups == group
        if labels[mask].sum():
            ranked = np.argsort(-scores[mask], kind="stable")[:5]
            hit.append(float(labels[mask][ranked].any()))
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "precision_at_0.5": float(precision), "recall_at_0.5": float(recall),
        "f1_at_0.5": float(f1),
        "answerable_question_hit_at_5": float(np.mean(hit)) if hit else None,
    }


def run(output, tracking_uri=None, report_path=None):
    output.mkdir(parents=True, exist_ok=True)
    question_path = ROOT / "eval_questions.jsonl"
    corpus_path = ROOT / "embedding-eval/frozen-corpus.json"
    calibration_path = ROOT / "embedding-eval/calibration.json"
    cases = rag_eval.load_cases(question_path)
    chunks = [app.DocumentChunk(**c) for c in json.loads(corpus_path.read_text())]
    excluded = set(json.loads(calibration_path.read_text())["dataset"]["held_out_case_numbers"])
    train_ids, val_ids = question_split(cases, excluded)
    X, y, groups = make_pairs(cases, chunks, train_ids)
    X_val, y_val, val_groups = make_pairs(cases, chunks, val_ids)
    specifications = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1),
            {"model__max_depth": [2, 4, None], "model__min_samples_leaf": [2, 8]},
        ),
    }
    baseline = PairFeatures().transform(X_val)[:, 0]
    report = {
        "purpose": "Offline supervised prototype; never used by the chat app.",
        "label_policy": "Exact evidence matches are positive; other pairs are weak negatives, not verified irrelevant.",
        "split_policy": "Question-disjoint train/validation from prior calibration questions; same source document.",
        "limitations": [
            "One document and 41 questions cannot establish generalization.",
            "Validation is used to choose a model, so its score is not a final test score.",
            "The prior holdout has already been reported; collect new documents and labels for a fresh final test.",
            "Lexical classifiers may miss paraphrases. No model is promoted to live retrieval.",
        ],
        "train_case_numbers": [i + 1 for i in train_ids],
        "validation_case_numbers": [i + 1 for i in val_ids],
        "excluded_prior_holdout_case_numbers": sorted(excluded),
        "pair_counts": {"train": len(y), "validation": len(y_val), "train_positive": int(y.sum()), "validation_positive": int(y_val.sum())},
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (question_path, corpus_path, calibration_path)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "baseline_query_coverage": metrics(y_val, baseline, val_groups), "models": {},
        "live_retrieval_changed": False, "final_test_evaluated": False,
    }
    tracker = None
    if tracking_uri:
        import mlflow
        tracker = mlflow
        tracker.set_tracking_uri(tracking_uri)
        tracker.set_experiment("pdf-rag-offline-relevance")
    for name, (model, grid) in specifications.items():
        pipeline = Pipeline([("features", PairFeatures()), ("scale", StandardScaler()), ("model", model)])
        search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=GroupKFold(3), n_jobs=1, error_score="raise")
        search.fit(X, y, groups=groups)
        scores = search.predict_proba(X_val)[:, 1]
        result = {
            "best_params": search.best_params_, "cv_average_precision": float(search.best_score_),
            "validation": metrics(y_val, scores, val_groups),
            "search_runs": [
                {"params": params, "mean_cv_average_precision": float(mean), "std_cv_average_precision": float(std)}
                for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])
            ],
        }
        report["models"][name] = result
        artifact = output / f"{name}.joblib"
        joblib.dump(search.best_estimator_, artifact)
        if tracker:
            with tracker.start_run(run_name=name):
                tracker.log_params(result["best_params"])
                tracker.log_metrics({f"validation_{k}": v for k, v in result["validation"].items() if v is not None})
                tracker.log_metric("cv_average_precision", result["cv_average_precision"])
                tracker.log_artifact(str(artifact))
                tracker.log_dict(result["search_runs"], "cv_search.json")
    selected = max(report["models"], key=lambda name: report["models"][name]["validation"]["average_precision"])
    report["selected_offline_model"] = selected
    report["selection_reason"] = "Highest validation average precision; deployment deliberately requires separate end-to-end evidence."
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_offline_model": selected, "models": report["models"], "live_retrieval_changed": False}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/relevance")
    parser.add_argument("--mlflow-uri", help="Optional tracking URI; install requirements-tracking.txt first.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/relevance-training.json")
    args = parser.parse_args()
    run(args.output, args.mlflow_uri, args.report)

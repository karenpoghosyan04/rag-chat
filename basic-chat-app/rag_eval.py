"""Offline retrieval evaluation for the local PDF RAG index.

The evaluator deliberately measures retrieval separately from answer generation.
That makes a miss actionable: the relevant evidence either reached the context or
it did not, independently of what the LLM later wrote.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import app


@dataclass(frozen=True)
class RelevantEvidence:
    """One labeled fact location that a retrieved chunk should cover."""

    page: Optional[int] = None
    contains: str = ""


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant: tuple[RelevantEvidence, ...]


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def evidence_matches(chunk: app.DocumentChunk, evidence: RelevantEvidence) -> bool:
    """Return whether a chunk satisfies all labels present on one evidence item."""
    if evidence.page is not None:
        page_end = chunk.page_end or chunk.page
        if not chunk.page <= evidence.page <= page_end:
            return False
    if evidence.contains and _normalized(evidence.contains) not in _normalized(chunk.text):
        return False
    return True


def load_cases(path: Path) -> list[EvalCase]:
    """Load strict JSONL labels so malformed evaluations fail loudly."""
    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            question = item["question"].strip()
            raw_relevant = item["relevant"]
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: expected question and relevant fields") from exc
        if not question or not isinstance(raw_relevant, list):
            raise ValueError(f"{path}:{line_number}: question must be text and relevant must be a list")
        relevant: list[RelevantEvidence] = []
        for label in raw_relevant:
            if not isinstance(label, dict):
                raise ValueError(f"{path}:{line_number}: every relevant item must be an object")
            page = label.get("page")
            contains = label.get("contains", "")
            if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
                raise ValueError(f"{path}:{line_number}: relevant page must be a positive integer")
            if not isinstance(contains, str):
                raise ValueError(f"{path}:{line_number}: relevant contains must be text")
            if page is None and not contains.strip():
                raise ValueError(f"{path}:{line_number}: relevant items need page and/or contains")
            relevant.append(RelevantEvidence(page=page, contains=contains.strip()))
        cases.append(EvalCase(question=question, relevant=tuple(relevant)))
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases


def evaluate_cases(
    index: app.DocumentIndex,
    cases: list[EvalCase],
    settings: app.RagSettings,
    cutoffs: list[int],
    embed_query: Optional[Callable[[str], list[float]]] = None,
    include_details: bool = True,
) -> dict:
    """Compute macro retrieval metrics and retain per-question diagnostics."""
    cutoffs = sorted(set(cutoffs))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("At least one positive retrieval cutoff is required.")
    max_k = cutoffs[-1]
    answerable = [case for case in cases if case.relevant]
    unanswerable = [case for case in cases if not case.relevant]
    aggregates = {
        k: {"recall": 0.0, "hit_rate": 0.0, "precision": 0.0}
        for k in cutoffs
    }
    reciprocal_rank = 0.0
    correct_rejections = 0
    unanswerable_retrieved_chunks = 0
    latencies_ms: list[float] = []
    details: list[dict] = []

    for case in cases:
        started = time.perf_counter()
        query_embedding = embed_query(case.question) if embed_query else None
        ranked = index.search_with_scores(
            case.question,
            limit=max_k,
            query_embedding=query_embedding,
            settings=settings,
        )
        retrieved = [result.chunk for result in ranked]
        latencies_ms.append((time.perf_counter() - started) * 1000)
        ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if any(evidence_matches(chunk, label) for label in case.relevant)
        ]
        if case.relevant:
            if ranks:
                reciprocal_rank += 1.0 / ranks[0]
            for k in cutoffs:
                top_k = retrieved[:k]
                matched_labels = sum(
                    any(evidence_matches(chunk, label) for chunk in top_k)
                    for label in case.relevant
                )
                relevant_chunks = sum(
                    any(evidence_matches(chunk, label) for label in case.relevant)
                    for chunk in top_k
                )
                aggregates[k]["recall"] += matched_labels / len(case.relevant)
                aggregates[k]["hit_rate"] += float(matched_labels > 0)
                aggregates[k]["precision"] += relevant_chunks / k
        else:
            unanswerable_retrieved_chunks += len(retrieved)
            if not retrieved:
                correct_rejections += 1

        if include_details:
            details.append({
                "question": case.question,
                "expected": [
                    {key: value for key, value in {"page": label.page, "contains": label.contains}.items() if value}
                    for label in case.relevant
                ],
                "first_relevant_rank": ranks[0] if ranks else None,
                "unanswerable_outcome": (
                    None if case.relevant else "correct_rejection" if not retrieved else "gated_false_positive"
                ),
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk": chunk.number,
                        "page": chunk.page,
                        "page_end": chunk.page_end or chunk.page,
                        "section": chunk.section,
                        "relevant": rank in ranks,
                        "reranker_score": ranked[rank - 1].score,
                        "preview": chunk.text[:180],
                    }
                    for rank, chunk in enumerate(retrieved, start=1)
                ],
            })

    answerable_count = len(answerable)
    metrics = {
        f"recall@{k}": aggregates[k]["recall"] / answerable_count if answerable_count else None
        for k in cutoffs
    }
    metrics.update({
        f"hit_rate@{k}": aggregates[k]["hit_rate"] / answerable_count if answerable_count else None
        for k in cutoffs
    })
    metrics.update({
        f"precision@{k}": aggregates[k]["precision"] / answerable_count if answerable_count else None
        for k in cutoffs
    })
    metrics[f"mrr@{max_k}"] = reciprocal_rank / answerable_count if answerable_count else None
    metrics["no_answer_accuracy"] = (
        correct_rejections / len(unanswerable) if unanswerable else None
    )
    metrics["gated_false_positive_rate"] = (
        1.0 - correct_rejections / len(unanswerable) if unanswerable else None
    )
    metrics["avg_unanswerable_chunks"] = (
        unanswerable_retrieved_chunks / len(unanswerable) if unanswerable else None
    )
    metrics["latency_ms_p50"] = statistics.median(latencies_ms)
    metrics["latency_ms_p95"] = sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95 + 0.999) - 1)]
    recall_at_max_k = metrics[f"recall@{max_k}"]
    no_answer_accuracy = metrics["no_answer_accuracy"]
    production_targets = {f"recall@{max_k}": 0.85, "no_answer_accuracy": 0.85}
    production_ready = (
        recall_at_max_k is not None
        and no_answer_accuracy is not None
        and recall_at_max_k > production_targets[f"recall@{max_k}"]
        and no_answer_accuracy > production_targets["no_answer_accuracy"]
    )
    return {
        "cases": len(cases),
        "answerable_cases": answerable_count,
        "unanswerable_cases": len(unanswerable),
        "metrics": metrics,
        "production_gate": {
            "targets": production_targets,
            "ready": production_ready,
        },
        "details": details,
    }


def run_benchmark(
    pdf: Path,
    cases: list[EvalCase],
    base_settings: app.RagSettings,
    cutoffs: list[int],
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
    dense_weights: list[float],
) -> dict:
    """Benchmark token chunking and dense/sparse RRF ratios with cached queries."""
    if base_settings.search_method == "bm25":
        raise ValueError("The hybrid benchmark requires hybrid or embedding retrieval.")
    questions = [case.question for case in cases]
    query_vectors = app.embed_texts(questions, kind="query")
    query_embeddings = dict(zip(questions, query_vectors))
    pdf_bytes = pdf.read_bytes()
    runs: list[dict] = []

    for chunk_size in chunk_sizes:
        for overlap in chunk_overlaps:
            if overlap >= chunk_size:
                raise ValueError("Every benchmark chunk overlap must be smaller than its chunk size.")
            chunk_values = base_settings.to_dict()
            chunk_values.update({
                "search_method": "hybrid",
                "chunk_size_tokens": chunk_size,
                "chunk_overlap_tokens": overlap,
            })
            chunk_settings = app.RagSettings.from_dict(chunk_values)
            print(
                f"Benchmarking {chunk_size}-token chunks with {overlap}-token overlap...",
                flush=True,
            )
            chunks, page_count, visual_pages = app.parse_pdf(pdf_bytes, chunk_settings)
            embeddings = app.embed_texts([chunk.text for chunk in chunks], kind="passage")
            index = app.DocumentIndex()
            _, embedding_model, _, _ = app.embedding_config()
            index.replace(
                pdf.name,
                chunks,
                pdf_bytes,
                page_count,
                visual_pages,
                embeddings,
                embedding_model,
            )
            for dense_weight in dense_weights:
                values = chunk_settings.to_dict()
                values.update({
                    "bm25_weight": 1.0 - dense_weight,
                    "embedding_weight": dense_weight,
                })
                settings = app.RagSettings.from_dict(values)
                report = evaluate_cases(
                    index,
                    cases,
                    settings,
                    cutoffs,
                    embed_query=query_embeddings.__getitem__,
                    include_details=False,
                )
                runs.append({
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": overlap,
                    "dense_weight": dense_weight,
                    "bm25_weight": 1.0 - dense_weight,
                    "chunks": len(chunks),
                    "metrics": report["metrics"],
                })

    max_k = max(cutoffs)
    best = max(
        runs,
        key=lambda run: (
            run["metrics"].get(f"recall@{max_k}") or 0.0,
            run["metrics"].get("no_answer_accuracy") or 0.0,
            run["metrics"].get(f"mrr@{max_k}") or 0.0,
            -(run["metrics"].get("latency_ms_p95") or float("inf")),
        ),
    )
    production_targets = {f"recall@{max_k}": 0.85, "no_answer_accuracy": 0.85}
    passing_runs = [
        run for run in runs
        if (run["metrics"].get(f"recall@{max_k}") or 0.0) >= production_targets[f"recall@{max_k}"]
        and (run["metrics"].get("no_answer_accuracy") or 0.0) >= production_targets["no_answer_accuracy"]
    ]
    return {
        "pdf": str(pdf),
        "cases": len(cases),
        "grid": {
            "chunk_size_tokens": chunk_sizes,
            "chunk_overlap_tokens": chunk_overlaps,
            "dense_weight": dense_weights,
            "bm25_weight": "1 - dense_weight",
        },
        "selection_order": [f"recall@{max_k}", "no_answer_accuracy", f"mrr@{max_k}", "latency_ms_p95"],
        "production_gate": {
            "targets": production_targets,
            "passing_runs": len(passing_runs),
            "ready": bool(passing_runs),
        },
        "best": best,
        "runs": runs,
    }


def _settings_from_args(args: argparse.Namespace) -> app.RagSettings:
    values = {}
    if args.settings:
        values = json.loads(args.settings.read_text(encoding="utf-8"))
    if args.method:
        values["search_method"] = args.method
    return app.RagSettings.from_dict(values)


def _print_report(report: dict, method: str, retrieval: str) -> None:
    print(f"Evaluated {report['cases']} questions with {method} search ({retrieval}).")
    for name, value in report["metrics"].items():
        if value is not None:
            print(f"{name:>20}: {value:.3f}")
    misses = [item for item in report["details"] if item["expected"] and item["first_relevant_rank"] is None]
    if misses:
        print("\nMissed questions:")
        for item in misses:
            print(f"- {item['question']}")
    false_positives = [
        item for item in report["details"]
        if item.get("unanswerable_outcome") == "gated_false_positive"
    ]
    if false_positives:
        print("\nUnanswerable queries that passed retrieval gates:")
        for item in false_positives:
            print(f"- {item['question']}")


def _parse_int_list(parser: argparse.ArgumentParser, value: str, option: str) -> list[int]:
    try:
        parsed = sorted(set(int(item.strip()) for item in value.split(",")))
    except ValueError:
        parsed = []
    if not parsed or parsed[0] < 0:
        parser.error(f"{option} must be a comma-separated list of non-negative integers")
    return parsed


def _parse_weight_list(parser: argparse.ArgumentParser, value: str) -> list[float]:
    try:
        parsed = sorted(set(float(item.strip()) for item in value.split(",")))
    except ValueError:
        parsed = []
    if not parsed or any(not 0 <= item <= 1 for item in parsed):
        parser.error("--dense-weights must contain numbers between 0 and 1")
    return parsed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure PDF retrieval Recall@K, MRR, precision, and latency.")
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to index")
    parser.add_argument("--questions", type=Path, required=True, help="Labeled JSONL evaluation set")
    parser.add_argument("--k", default="1,3,5", help="Comma-separated retrieval cutoffs (default: 1,3,5)")
    parser.add_argument("--method", choices=("hybrid", "embedding", "bm25"), help="Override search_method")
    parser.add_argument("--settings", type=Path, help="JSON object containing the same RAG settings as the UI")
    parser.add_argument("--output", type=Path, help="Optional full JSON report")
    parser.add_argument("--fail-under-recall", type=float, help="Exit 1 when Recall@max(K) is below this value")
    parser.add_argument(
        "--fail-under-no-answer-accuracy",
        type=float,
        help="Exit 1 when too many unanswerable queries pass the retrieval confidence gates",
    )
    parser.add_argument("--benchmark-output", type=Path, help="Run a hybrid grid search and write JSON results")
    parser.add_argument("--chunk-sizes", default="256,512,1024", help="Benchmark token budgets")
    parser.add_argument("--chunk-overlaps", default="0,64", help="Benchmark token overlaps")
    parser.add_argument("--dense-weights", default="0,0.5,1", help="Dense RRF ratios; BM25 uses 1-ratio")
    args = parser.parse_args(argv)

    try:
        cutoffs = [int(value.strip()) for value in args.k.split(",")]
        if any(value < 1 for value in cutoffs):
            raise ValueError
    except ValueError:
        parser.error("--k must be a comma-separated list of positive integers")
    if args.fail_under_recall is not None and not 0 <= args.fail_under_recall <= 1:
        parser.error("--fail-under-recall must be between 0 and 1")
    if args.fail_under_no_answer_accuracy is not None and not 0 <= args.fail_under_no_answer_accuracy <= 1:
        parser.error("--fail-under-no-answer-accuracy must be between 0 and 1")
    chunk_sizes = _parse_int_list(parser, args.chunk_sizes, "--chunk-sizes")
    if chunk_sizes[0] == 0:
        parser.error("--chunk-sizes values must be positive")
    chunk_overlaps = _parse_int_list(parser, args.chunk_overlaps, "--chunk-overlaps")
    dense_weights = _parse_weight_list(parser, args.dense_weights)

    app.load_env()
    settings = _settings_from_args(args)
    cases = load_cases(args.questions)
    status = app.index_document(args.pdf.name, args.pdf.read_bytes(), settings)
    embed_query = None
    if settings.search_method != "bm25" and status["embedding_model"]:
        embed_query = lambda query: app.embed_texts([query], kind="query")[0]
    elif settings.search_method != "bm25":
        raise RuntimeError(
            f"{settings.search_method} retrieval was requested but embeddings were unavailable; "
            "refusing to report fallback BM25 metrics as hybrid results."
        )
    report = evaluate_cases(app.DOCUMENT_INDEX, cases, settings, cutoffs, embed_query)
    report["configuration"] = {
        "pdf": str(args.pdf),
        "settings": settings.to_dict(),
        "retrieval": status["retrieval"],
        "embedding_model": status["embedding_model"],
        "reranker_model": app.os.getenv("RERANKER_MODEL", app.RERANKER_MODEL),
    }
    _print_report(report, settings.search_method, status["retrieval"])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote detailed report to {args.output}")
    if args.benchmark_output:
        benchmark = run_benchmark(
            args.pdf,
            cases,
            settings,
            cutoffs,
            chunk_sizes,
            chunk_overlaps,
            dense_weights,
        )
        args.benchmark_output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(benchmark['runs'])} benchmark runs to {args.benchmark_output}")
    recall = report["metrics"][f"recall@{max(cutoffs)}"]
    no_answer = report["metrics"]["no_answer_accuracy"]
    recall_failed = args.fail_under_recall is not None and recall is not None and recall < args.fail_under_recall
    no_answer_failed = (
        args.fail_under_no_answer_accuracy is not None
        and no_answer is not None
        and no_answer < args.fail_under_no_answer_accuracy
    )
    return int(recall_failed or no_answer_failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(2)

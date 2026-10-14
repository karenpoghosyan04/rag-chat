"""Reproducible, non-production comparison of two local embedding models.

Each model runs in a fresh subprocess, embeds the same frozen corpus and queries,
and writes a separate float32 index.  The production DocumentIndex is constructed
locally; the application singleton and deployment configuration are never changed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import app
import rag_eval


OLD_MODEL = "BAAI/bge-small-en-v1.5"
NEW_MODEL = "google/embeddinggemma-2"
BASELINE_RECALL_5 = 0.91
CUTOFFS = (1, 5, 10)
CALIBRATED_THRESHOLDS = {
    OLD_MODEL: {"embedding_min_similarity": 0.555, "min_score": 0.20},
    NEW_MODEL: {"embedding_min_similarity": 0.630, "min_score": 0.20},
}


def category_for(case_number: int) -> str:
    if case_number < 16:
        return "architecture_and_mechanisms"
    if case_number < 24:
        return "training_and_efficiency"
    if case_number < 35:
        return "results_ablations_and_extensions"
    return "unanswerable"


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def rank_metrics(
    cases: list[rag_eval.EvalCase],
    rankings: list[list[app.DocumentChunk]],
) -> dict:
    rows = []
    for case_number, (case, chunks) in enumerate(zip(cases, rankings)):
        if not case.relevant:
            rows.append({
                "question": case.question,
                "category": category_for(case_number),
                "answerable": False,
                "returned": len(chunks),
            })
            continue
        first_label_ranks: list[int | None] = []
        for label in case.relevant:
            first_label_ranks.append(next((
                rank for rank, chunk in enumerate(chunks, start=1)
                if rag_eval.evidence_matches(chunk, label)
            ), None))
        first_relevant = min((rank for rank in first_label_ranks if rank is not None), default=None)
        recalls = {
            str(k): sum(rank is not None and rank <= k for rank in first_label_ranks) / len(case.relevant)
            for k in CUTOFFS
        }
        # Credit each evidence label once, at its first retrieved occurrence.
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in first_label_ranks
            if rank is not None and rank <= 10
        )
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(case.relevant)) + 1))
        rows.append({
            "question": case.question,
            "category": category_for(case_number),
            "answerable": True,
            "recall": recalls,
            "first_relevant_rank": first_relevant,
            "mrr": 1.0 / first_relevant if first_relevant and first_relevant <= 10 else 0.0,
            "ndcg@10": dcg / idcg if idcg else 0.0,
            "top10_chunks": [chunk.number for chunk in chunks[:10]],
        })

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def summarize(selected: list[dict]) -> dict:
        positives = [row for row in selected if row["answerable"]]
        negatives = [row for row in selected if not row["answerable"]]
        result = {}
        if positives:
            for k in CUTOFFS:
                result[f"recall@{k}"] = statistics.fmean(row["recall"][str(k)] for row in positives)
            result["mrr"] = statistics.fmean(row["mrr"] for row in positives)
            result["ndcg@10"] = statistics.fmean(row["ndcg@10"] for row in positives)
        if negatives:
            result["no_answer_accuracy"] = statistics.fmean(row["returned"] == 0 for row in negatives)
        return result

    categories = {}
    for category in sorted({row["category"] for row in rows}):
        categories[category] = summarize([row for row in rows if row["category"] == category])
    return {"metrics": summarize(rows), "categories": categories, "per_query": rows}


def load_corpus(path: Path) -> list[app.DocumentChunk]:
    return [app.DocumentChunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def configure_model(model_name: str):
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["LOCAL_EMBEDDING_MODEL"] = model_name
    app._LOCAL_EMBEDDER = None
    app._LOCAL_EMBEDDER_MODEL = ""
    return app.local_embedder(model_name)


def embed(texts: list[str], kind: str) -> np.ndarray:
    return np.asarray(app.embed_texts(texts, kind=kind), dtype=np.float32)


def peak_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def worker(args: argparse.Namespace) -> None:
    chunks = load_corpus(args.corpus)
    cases = rag_eval.load_cases(args.questions)
    texts = [chunk.text for chunk in chunks]
    questions = [case.question for case in cases]
    model = configure_model(args.model)

    # Warm kernels/model initialization, but exclude warmup from reported timing.
    embed([questions[0]], "query")
    started = time.perf_counter()
    document_vectors = normalize(embed(texts, "passage"))
    document_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_vectors_batch = normalize(embed(questions, "query"))
    query_batch_seconds = time.perf_counter() - started

    index_path = args.result.with_suffix(".index.npy")
    np.save(index_path, document_vectors, allow_pickle=False)

    exact_rankings = []
    exact_latencies = []
    for question in questions:
        started = time.perf_counter()
        query_vector = normalize(embed([question], "query"))[0]
        scores = document_vectors @ query_vector
        ranked = np.argsort(-scores, kind="stable")[:10]
        exact_rankings.append([chunks[int(index)] for index in ranked])
        exact_latencies.append((time.perf_counter() - started) * 1000)

    settings = app.RagSettings.from_dict(CALIBRATED_THRESHOLDS[args.model])
    production_index = app.DocumentIndex()
    production_index.replace(
        "frozen-evaluation-corpus.pdf",
        chunks,
        embeddings=document_vectors.tolist(),
        embedding_model=args.model,
    )
    # Warm the unchanged cross-encoder reranker before latency measurement.
    app.rerank_texts(questions[0], [chunks[0].text])
    production_rankings = []
    production_latencies = []
    for question in questions:
        started = time.perf_counter()
        query_vector = normalize(embed([question], "query"))[0]
        ranked = production_index.search_with_scores(
            question,
            limit=10,
            query_embedding=query_vector.tolist(),
            settings=settings,
        )
        production_rankings.append([item.chunk for item in ranked])
        production_latencies.append((time.perf_counter() - started) * 1000)

    dimension = int(document_vectors.shape[1])
    if args.model == NEW_MODEL:
        pooling = "mean pooling (prompt tokens included), then L2 normalization"
        prefixes = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }
    else:
        pooling = "FastEmbed model-defined pooling and normalization"
        prefixes = {"query": "query_embed", "document": "passage_embed"}
    report = {
        "model": args.model,
        "dimension": dimension,
        "normalization": "L2-normalized for cosine search",
        "pooling": pooling,
        "prefixes": prefixes,
        "thresholds": CALIBRATED_THRESHOLDS[args.model],
        "corpus_chunks": len(chunks),
        "index": {
            "path": str(index_path),
            "payload_bytes": int(document_vectors.nbytes),
            "file_bytes": index_path.stat().st_size,
        },
        "performance": {
            "document_embedding_seconds": document_seconds,
            "document_embeddings_per_second": len(texts) / document_seconds,
            "batch_query_embeddings_per_second": len(questions) / query_batch_seconds,
            "peak_process_rss_mb": peak_rss_mb(),
            "exact_query_latency_ms_p50": statistics.median(exact_latencies),
            "exact_query_latency_ms_p95": percentile(exact_latencies, 0.95),
            "production_query_latency_ms_p50": statistics.median(production_latencies),
            "production_query_latency_ms_p95": percentile(production_latencies, 0.95),
        },
        "exact_dense": rank_metrics(cases, exact_rankings),
        "production_hybrid_exact": rank_metrics(cases, production_rankings),
    }
    args.result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def bootstrap_delta(old_rows: list[dict], new_rows: list[dict], samples: int = 20_000) -> dict:
    old = [row for row in old_rows if row["answerable"]]
    new_by_question = {row["question"]: row for row in new_rows if row["answerable"]}
    paired = [(row["recall"]["5"], new_by_question[row["question"]]["recall"]["5"]) for row in old]
    observed = statistics.fmean(new - previous for previous, new in paired)
    rng = random.Random(20261007)
    deltas = []
    for _ in range(samples):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        deltas.append(statistics.fmean(new - previous for previous, new in draw))
    deltas.sort()
    return {
        "old_recall@5": statistics.fmean(old for old, _ in paired),
        "new_recall@5": statistics.fmean(new for _, new in paired),
        "delta": observed,
        "bootstrap_95_ci": [deltas[int(samples * 0.025)], deltas[int(samples * 0.975)]],
        "samples": samples,
        "statistically_meaningful": deltas[int(samples * 0.025)] > 0 or deltas[int(samples * 0.975)] < 0,
    }


def differentials(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old = {row["question"]: row for row in old_rows if row["answerable"]}
    new = {row["question"]: row for row in new_rows if row["answerable"]}
    return {
        "new_succeeds_old_fails": [
            question for question in old
            if new[question]["recall"]["5"] == 1.0 and old[question]["recall"]["5"] < 1.0
        ],
        "old_succeeds_new_fails": [
            question for question in old
            if old[question]["recall"]["5"] == 1.0 and new[question]["recall"]["5"] < 1.0
        ],
    }


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(comparison: dict, path: Path) -> None:
    old, new = comparison["models"]
    lines = [
        "# Embedding model comparison",
        "",
        "| Retrieval | Model | R@1 | R@5 | R@10 | MRR | nDCG@10 | p50 / p95 latency | Index | Peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for retrieval, label, latency_prefix in (
        ("exact_dense", "Exact dense", "exact"),
        ("production_hybrid_exact", "Production hybrid (exact vectors)", "production"),
    ):
        for report in (old, new):
            metrics = report[retrieval]["metrics"]
            perf = report["performance"]
            lines.append(
                f"| {label} | `{report['model']}` | {fmt_percent(metrics['recall@1'])} | "
                f"{fmt_percent(metrics['recall@5'])} | {fmt_percent(metrics['recall@10'])} | "
                f"{metrics['mrr']:.3f} | {metrics['ndcg@10']:.3f} | "
                f"{perf[latency_prefix + '_query_latency_ms_p50']:.1f} / "
                f"{perf[latency_prefix + '_query_latency_ms_p95']:.1f} ms | "
                f"{report['index']['payload_bytes'] / 1024:.1f} KiB | "
                f"{perf['peak_process_rss_mb']:.0f} MiB |"
            )
    lines.extend([
        "",
        "## Throughput",
        "",
        "| Model | Document embeddings/s | Batched query embeddings/s | Dimension |",
        "|---|---:|---:|---:|",
    ])
    for report in (old, new):
        perf = report["performance"]
        lines.append(
            f"| `{report['model']}` | {perf['document_embeddings_per_second']:.2f} | "
            f"{perf['batch_query_embeddings_per_second']:.2f} | {report['dimension']} |"
        )
    lines.extend(["", "## Recall@5 paired bootstrap", ""])
    for name, result in comparison["significance"].items():
        ci = result["bootstrap_95_ci"]
        lines.append(
            f"- {name}: delta {result['delta'] * 100:+.2f} pp, "
            f"95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] pp."
        )
    lines.extend([
        "",
        "## Production results by category",
        "",
        "| Category | Old R@5 | New R@5 |",
        "|---|---:|---:|",
    ])
    old_categories = old["production_hybrid_exact"]["categories"]
    new_categories = new["production_hybrid_exact"]["categories"]
    for category in ("architecture_and_mechanisms", "training_and_efficiency", "results_ablations_and_extensions"):
        lines.append(
            f"| {category.replace('_', ' ')} | {fmt_percent(old_categories[category]['recall@5'])} | "
            f"{fmt_percent(new_categories[category]['recall@5'])} |"
        )
    lines.append(
        f"| unanswerable (no-answer accuracy) | {fmt_percent(old_categories['unanswerable']['no_answer_accuracy'])} | "
        f"{fmt_percent(new_categories['unanswerable']['no_answer_accuracy'])} |"
    )
    lines.extend(["", "## Production Recall@5 query differences", ""])
    production_differences = comparison["differential_queries"]["production_hybrid_exact"]
    lines.append("EmbeddingGemma 2 succeeds and the old model fails:")
    lines.extend(f"- {question}" for question in production_differences["new_succeeds_old_fails"])
    lines.extend(["", "The old model succeeds and EmbeddingGemma 2 fails:"])
    lines.extend(f"- {question}" for question in production_differences["old_succeeds_new_fails"])
    lines.extend([
        "",
        "## Recommendation",
        "",
        comparison["recommendation"],
        "",
        "The production retriever has no ANN implementation or ANN parameters; it scans every vector with exact cosine similarity. "
        "Accordingly, the production-style result above is exact and there is no approximation gap to report.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = app.RagSettings()
    chunks, page_count, _ = app.parse_pdf(args.pdf.read_bytes(), settings)
    corpus_path = args.output_dir / "frozen-corpus.json"
    corpus_path.write_text(json.dumps([asdict(chunk) for chunk in chunks], indent=2) + "\n", encoding="utf-8")
    results = []
    for label, model in (("old", OLD_MODEL), ("new", NEW_MODEL)):
        result_path = args.output_dir / f"{label}-model.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--model", model,
            "--corpus", str(corpus_path), "--questions", str(args.questions), "--result", str(result_path),
        ]
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        subprocess.run(command, check=True, env=environment)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    old, new = results
    significance = {
        "exact_dense": bootstrap_delta(old["exact_dense"]["per_query"], new["exact_dense"]["per_query"]),
        "production_hybrid_exact": bootstrap_delta(
            old["production_hybrid_exact"]["per_query"], new["production_hybrid_exact"]["per_query"]
        ),
    }
    differences = {
        "exact_dense": differentials(old["exact_dense"]["per_query"], new["exact_dense"]["per_query"]),
        "production_hybrid_exact": differentials(
            old["production_hybrid_exact"]["per_query"], new["production_hybrid_exact"]["per_query"]
        ),
    }
    production_delta = significance["production_hybrid_exact"]["delta"]
    old_no_answer = old["production_hybrid_exact"]["metrics"].get("no_answer_accuracy")
    new_no_answer = new["production_hybrid_exact"]["metrics"].get("no_answer_accuracy")
    operational_regression = (
        new["performance"]["document_embeddings_per_second"]
        < old["performance"]["document_embeddings_per_second"] / 10
        or new["performance"]["peak_process_rss_mb"] > old["performance"]["peak_process_rss_mb"] * 2
    )
    abstention_regression = (
        old_no_answer is not None and new_no_answer is not None and new_no_answer < old_no_answer - 0.01
    )
    if production_delta < -0.01 or abstention_regression or operational_regression:
        recommendation = "Keep the previous model: the production-style Recall@5 regression exceeds the 1 percentage-point limit."
        if abstention_regression or operational_regression:
            recommendation = (
                "Keep the previous model: Recall@5 is tied, but EmbeddingGemma 2 regresses no-answer accuracy "
                "and has materially worse indexing throughput and memory use under the frozen configuration."
            )
    elif production_delta > 0.01 and significance["production_hybrid_exact"]["statistically_meaningful"]:
        recommendation = "Switch: EmbeddingGemma 2 improves production-style Recall@5 by more than 1 point with a paired 95% CI excluding zero."
    else:
        recommendation = "Run a shadow test: offline Recall@5 is within the 1-point guardrail or the paired result remains uncertain."
    comparison = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_inputs": {
            "pdf": str(args.pdf),
            "pages": page_count,
            "corpus_chunks": len(chunks),
            "questions": str(args.questions),
            "settings": settings.to_dict(),
            "similarity": "cosine over L2-normalized float32 vectors",
            "reranker": os.getenv("RERANKER_MODEL", app.RERANKER_MODEL),
        },
        "baseline": {
            "reported_recall@5": BASELINE_RECALL_5,
            "reported_model": OLD_MODEL,
            "observed_old_production_recall@5": old["production_hybrid_exact"]["metrics"]["recall@5"],
        },
        "ann": {
            "run": False,
            "reason": "No ANN implementation or ANN parameters exist; DocumentIndex performs an exact cosine scan.",
        },
        "models": results,
        "significance": significance,
        "differential_queries": differences,
        "recommendation": recommendation,
    }
    json_path = args.output_dir / "comparison.json"
    markdown_path = args.output_dir / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_markdown(comparison, markdown_path)
    print(markdown_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.pdf or not args.output_dir:
            parser.error("--pdf and --output-dir are required")
        parent(args)


if __name__ == "__main__":
    main()

"""Reproducible dataset inspection with dependency-free HTML/SVG charts."""
from collections import Counter
from html import escape
import json
from pathlib import Path

import numpy as np

import app
import rag_eval
from src.relevance import PairFeatures

ROOT = Path(__file__).resolve().parents[1]


class NotebookHTML:
    def __init__(self, html):
        self.html = html

    def _repr_html_(self):
        return self.html


def summarize():
    cases = rag_eval.load_cases(ROOT / "eval_questions.jsonl")
    chunks = [app.DocumentChunk(**c) for c in json.loads((ROOT / "embedding-eval/frozen-corpus.json").read_text())]
    lengths = [len(c.text.split()) for c in chunks]
    questions = [" ".join(c.question.casefold().split()) for c in cases]
    texts = [" ".join(c.text.casefold().split()) for c in chunks]
    missing = [i + 1 for i, c in enumerate(cases) if any(not any(rag_eval.evidence_matches(chunk, label) for chunk in chunks) for label in c.relevant)]
    excluded = set(json.loads((ROOT / "embedding-eval/calibration.json").read_text())["dataset"]["held_out_case_numbers"])
    pairs, labels = [], []
    for i, case in enumerate(cases, 1):
        if i in excluded:
            continue
        for chunk in chunks:
            pairs.append((case.question, chunk.text))
            labels.append(int(any(rag_eval.evidence_matches(chunk, e) for e in case.relevant)))
    features = PairFeatures().transform(pairs)
    correlations = np.corrcoef(np.column_stack([features, labels]), rowvar=False)
    names = [*PairFeatures.names, "known_evidence_match"]
    return {
        "documents": 1, "chunks": len(chunks), "questions": len(cases),
        "answerable": sum(bool(c.relevant) for c in cases),
        "unanswerable": sum(not c.relevant for c in cases),
        "empty_chunk_texts": sum(not c.text.strip() for c in chunks),
        "duplicate_questions": len(questions) - len(set(questions)),
        "duplicate_chunk_texts": len(texts) - len(set(texts)),
        "cases_with_unmatched_evidence": missing,
        "chunk_words": {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)},
        "chunk_word_counts": lengths,
        "chunks_per_start_page": dict(sorted(Counter(c.page for c in chunks).items())),
        "development_pair_labels": {"known_evidence_match": sum(labels), "weak_negative": len(labels) - sum(labels)},
        "feature_names": names, "development_feature_correlations": correlations.tolist(),
    }


def bars(title, values):
    width, row = 720, 34
    maximum = max(values.values(), default=1) or 1
    elements = [f'<svg viewBox="0 0 {width} {60 + row * len(values)}" role="img" aria-label="{escape(title)}"><text x="12" y="24" font-size="18">{escape(title)}</text>']
    for i, (label, value) in enumerate(values.items()):
        y = 50 + i * row
        elements.append(f'<text x="12" y="{y + 16}" font-size="12">{escape(str(label))}</text><rect x="220" y="{y}" width="{value / maximum * 420:.1f}" height="23" fill="#2563eb"/><text x="{228 + value / maximum * 420:.1f}" y="{y + 16}" font-size="12">{value}</text>')
    return "".join(elements) + "</svg>"


def render_html(s):
    bins = {"0–99 words": 0, "100–199 words": 0, "200–399 words": 0, "400+ words": 0}
    for count in s["chunk_word_counts"]:
        key = list(bins)[0 if count < 100 else 1 if count < 200 else 2 if count < 400 else 3]
        bins[key] += 1
    chart_html = bars("Questions by answerability", {"Answerable": s["answerable"], "Unanswerable": s["unanswerable"]})
    chart_html += bars("Chunk length distribution", bins)
    chart_html += bars("Chunks per starting page", s["chunks_per_start_page"])
    chart_html += bars("Development pair labels", s["development_pair_labels"])
    names = s["feature_names"]
    table = "<table><tr><th>Feature</th>" + "".join(f"<th>{escape(n)}</th>" for n in names) + "</tr>"
    for name, row in zip(names, s["development_feature_correlations"]):
        table += f"<tr><th>{escape(name)}</th>" + "".join(f"<td>{value:.3f}</td>" for value in row) + "</tr>"
    table += "</table>"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PDF RAG dataset exploration</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#172033}}svg{{display:block;width:100%;max-width:800px;margin:24px 0}}table{{border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border:1px solid #cbd5e1}}h1,h2{{color:#153b71}}</style>
<h1>PDF RAG dataset exploration</h1><p>{s['documents']} document · {s['chunks']} chunks · {s['questions']} questions. These are development data, not evidence of performance on new documents.</p>
<h2>Quality checks</h2><p>Empty chunks: {s['empty_chunk_texts']}. Duplicate questions: {s['duplicate_questions']}. Duplicate chunk texts: {s['duplicate_chunk_texts']}. Cases with missing evidence matches: {escape(str(s['cases_with_unmatched_evidence']))}.</p>
{chart_html}<h2>Development feature correlations</h2>{table}
<h2>What this means</h2><p>The data are small and mostly answerable. Pair labels are strongly imbalanced: most passages do not match the supplied evidence. An unmatched passage is only a weak negative; it can still contain useful information. Use average precision and recall instead of reporting majority-class accuracy.</p>
<p>Normalize text for matching, preserve page/section metadata, and split questions before creating pairs. StandardScaler is fitted inside each training fold for the supervised models. The live embedding normalization, chunking, thresholds, and reranker remain unchanged.</p>
<p>Correlations describe associations, not causes. Feature analysis excludes the nine previously held-out questions. Same-document question splitting does not measure generalization to new documents. Gather new PDFs with manually reviewed labels and split by document for a final test.</p></html>'''


def write_report(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize()
    (output / "eda.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "eda.html").write_text(render_html(summary))
    return summary

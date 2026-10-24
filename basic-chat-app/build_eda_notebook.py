"""Build and execute our authored EDA cells using Python; no Jupyter required.

Captures stdout and the last expression's text/HTML representation. The saved
notebook can also be rerun normally in Jupyter after installing experiment deps.
"""
import ast
import contextlib
import io
import json
import sys

from src.eda import ROOT


def main():
    cells = []
    namespace = {}

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        tree = ast.parse(text)
        tail = tree.body.pop() if isinstance(tree.body[-1], ast.Expr) else None
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(tree, "eda-cell", "exec"), namespace)
            value = eval(compile(ast.Expression(tail.value), "eda-cell", "eval"), namespace) if tail else None
        outputs = []
        if stream.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(keepends=True)})
        if value is not None:
            data = {"text/plain": repr(value)}
            if hasattr(value, "_repr_html_"):
                data["text/html"] = value._repr_html_()
            outputs.append({"output_type": "execute_result", "execution_count": 1 + sum(c["cell_type"] == "code" for c in cells), "metadata": {}, "data": data})
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": 1 + sum(c["cell_type"] == "code" for c in cells), "outputs": outputs})

    markdown("# PDF question answering: exploratory data analysis\n\nGoal: find reliable evidence in a PDF and avoid unsupported answers. We inspect the frozen development corpus before comparing supervised relevance models. This notebook does not change the running app or evaluate a new final test.")
    code('from pathlib import Path\nimport sys, json\nroot = Path.cwd() if (Path.cwd() / "app.py").exists() else Path.cwd().parent\nif not (root / "app.py").exists():\n    raise RuntimeError("Run from basic-chat-app or its notebooks folder")\nsys.path.insert(0, str(root))\nfrom src.eda import write_report, render_html, NotebookHTML\nsummary = write_report(root / "reports")\nprint(json.dumps({k:v for k,v in summary.items() if k not in {"chunk_word_counts", "development_feature_correlations"}}, indent=2))')
    markdown("## Distributions, missing data, and correlations\n\nExact duplicate questions and chunks are counted. Every known evidence label is checked against the frozen chunks. Charts show length, page coverage, answerability, and the weak pair labels. Correlations use development questions only.")
    code('NotebookHTML(render_html(summary))')
    markdown("## Engineering decisions\n\n- Preserve the existing parser, embedding normalization, thresholds, and reranker.\n- Keep pages and source text with every chunk for evidence matching and citations.\n- Do not randomly split passage rows: one question would leak across splits. Split questions first.\n- Scale the four lexical features inside each training fold.\n- Treat unmatched passages as weak negatives, not verified irrelevant passages.\n- Exclude the old nine-question holdout from training and validation. It is already reported, so a final test needs new documents and labels.")
    code('path = root / "reports/relevance-training.json"\nif path.exists():\n    training = json.loads(path.read_text())\n    print(json.dumps({"baseline": training["baseline_query_coverage"], "models": {name: item["validation"] for name,item in training["models"].items()}, "live_retrieval_changed": training["live_retrieval_changed"]}, indent=2))\nelse:\n    print("Run train_relevance.py first.")')
    markdown("## Interpretation and next experiment\n\nThe supervised models are prototypes. Compare average precision against query coverage; top-five hit rate alone can hide weak precision. A classifier must not replace the current retriever just because it was trained. Retain the live pipeline until end-to-end evaluation on a larger reviewed dataset supports a change.\n\nFor a defensible final evaluation: collect several new PDFs, manually review answerable and unanswerable questions, group by document, tune on validation documents, then evaluate a locked test set once. Existing benchmark replays are regression checks, not fresh test results.")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": sys.version.split()[0]}}, "cells": cells}
    for i, cell in enumerate(cells):
        cell["id"] = f"eda-{i:02d}"
    destination = ROOT / "notebooks/01_eda.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2) + "\n")
    print(f"Executed {sum(c['cell_type'] == 'code' for c in cells)} cells and wrote {destination}")


if __name__ == "__main__":
    main()

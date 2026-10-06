"""Generate a verified, paraphrase-heavy retrieval evaluation set for testrag.pdf.

The questions are deliberately curated transformations of facts found while parsing
the document. Evidence labels are not trusted blindly: every phrase/page pair must
match a chunk produced by the application before the JSONL file is written.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import app


CASES: tuple[tuple[str, tuple[tuple[int, str], ...]], ...] = (
    (
        "Which design removes both recurrence and convolution from the sequence model?",
        ((1, "based solely on attention mechanisms"),),
    ),
    (
        "What training advantage follows from replacing step-by-step state propagation?",
        ((2, "allows for significantly more parallelization"),),
    ),
    (
        "How does the proposed system change the operation count needed to connect distant positions?",
        ((2, "reduced to a constant number of operations"),),
    ),
    (
        "How deep is the repeated block on the input-processing side?",
        ((3, "stack of N = 6 identical layers"),),
    ),
    (
        "What extra module lets the output stack consult the input stack?",
        ((3, "performs multi-head attention over the output of the encoder stack"),),
    ),
    (
        "What mechanism stops a generated position from peeking at future outputs?",
        ((3, "prevent positions from attending to subsequent positions"),),
    ),
    (
        "How are raw query-key products adjusted before normalization?",
        ((4, "divide each by [√] dk"),),
    ),
    (
        "Why is an inverse square-root factor applied when key vectors are wide?",
        ((4, "dot products grow large in magnitude"),),
    ),
    (
        "What capability is lost when all feature views are averaged into one focus operation?",
        ((5, "jointly attend to information from different representation subspaces"),),
    ),
    (
        "How many simultaneous focus branches are used in the reported configuration?",
        ((5, "h = 8 parallel attention layers"),),
    ),
    (
        "In how many distinct roles is the parallel focus mechanism deployed?",
        ((5, "multi-head attention in three different ways"),),
    ),
    (
        "What is the hidden width of the position-local nonlinear subnetwork?",
        ((5, "inner-layer has dimensionality dff = 2048"),),
    ),
    (
        "Which parameters are tied across token input, token output, and final prediction?",
        ((5, "share the same weight matrix"),),
    ),
    (
        "How does the model represent ordering despite lacking sequential or spatial filters?",
        ((6, "inject some information about the relative or absolute position"),),
    ),
    (
        "Why did the authors favor fixed trigonometric location signals over learned ones?",
        ((6, "extrapolate to sequence lengths longer than the ones encountered during training"),),
    ),
    (
        "Which layer family offers direct all-to-all communication without serial steps?",
        ((6, "connects all positions with a constant number of sequentially executed operations"),),
    ),
    (
        "Under what size relationship is global token interaction cheaper than recurrence?",
        ((7, "length n is smaller than the representation dimensionality d"),),
    ),
    (
        "What connectivity penalty comes from limiting each token to a local neighborhood?",
        ((7, "increase the maximum path length to O (n/r)"),),
    ),
    (
        "What qualitative linguistic patterns emerged when inspecting the learned focus maps?",
        ((7, "behavior related to the syntactic and semantic structure"),),
    ),
    (
        "Contrast the amount of bilingual training material used for the two translation directions.",
        ((7, "about 4.5 million sentence pairs"), (7, "consisting of 36M sentences")),
    ),
    (
        "Roughly how many source and target units were packed into one optimization batch?",
        ((7, "approximately 25000 source tokens and 25000 target tokens"),),
    ),
    (
        "Compare the wall-clock schedules for the smaller and larger configurations.",
        ((7, "100,000 steps or 12 hours"), (7, "300,000 steps (3.5 days)")),
    ),
    (
        "Which optimizer coefficients were selected for first moment, second moment, and numerical stability?",
        ((7, "β 1 = 0.9, β 2 = 0.98 and ϵ = 10 [−][9]"),),
    ),
    (
        "When does the learning-rate ramp stop and inverse-root decay begin?",
        ((7, "warmup steps_ = 4000"),),
    ),
    (
        "For the larger translation configuration, report German and French quality together with compute cost.",
        (
            (8, "state-of-the-art BLEU score of 28.4"),
            (8, "big model achieves a BLEU score of 41.0"),
            (8, "Training took 3.5 days on 8 P100 GPUs"),
        ),
    ),
    (
        "By what margin did the strongest German system beat previously published alternatives?",
        ((8, "outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU"),),
    ),
    (
        "How did checkpoint aggregation differ between the compact and expanded systems?",
        ((8, "averaging the last 5 checkpoints"), (8, "averaged the last 20 checkpoints")),
    ),
    (
        "What decoding breadth and length adjustment were used for translation?",
        ((8, "beam size of 4 and length penalty α = 0.6"),),
    ),
    (
        "What happened to translation quality when the architecture used only one focus branch?",
        ((9, "single-head attention is 0.9 BLEU worse than the best setting"),),
    ),
    (
        "What do the ablations say about capacity and regularization?",
        ((9, "bigger models are better, and dropout is very helpful in avoiding over-fitting"),),
    ),
    (
        "Did trainable location vectors materially outperform the fixed periodic scheme?",
        ((9, "observe nearly identical results to the base model"),),
    ),
    (
        "How much labeled and automatically labeled data supported the grammar experiments?",
        ((9, "about 40K training sentences"), (9, "approximately 17M sentences")),
    ),
    (
        "What score did the semi-supervised four-layer system obtain on the treebank test split?",
        ((10, "|Transformer(4 layers)|semi-supervised|92.7|"),),
    ),
    (
        "How was search configured when producing parse trees?",
        ((10, "beam size of 21 and α = 0.3"),),
    ),
    (
        "Which non-text input and output domains are identified as future extensions?",
        ((10, "images, audio and video"),),
    ),
    ("What dosage of aspirin was administered in the clinical trial?", ()),
    ("Which Martian crater hosted the rover's soil experiment?", ()),
    ("How did the quarterly tax rate affect the company's dividend?", ()),
    ("What recipe temperature produced the crispest sourdough crust?", ()),
    ("Which encryption cipher protected the blockchain wallet?", ()),
    ("How many goals were scored in the championship football match?", ()),
)


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _trigrams(words: list[str]) -> set[tuple[str, str, str]]:
    return {tuple(words[index:index + 3]) for index in range(len(words) - 2)}


def validate_cases(chunks: Iterable[app.DocumentChunk]) -> None:
    chunks = list(chunks)
    positive_count = sum(bool(evidence) for _, evidence in CASES)
    negative_count = len(CASES) - positive_count
    if positive_count < 20 or negative_count < 5:
        raise ValueError("The evaluation set must contain at least 20 positive and 5 negative cases.")
    for question, evidence_items in CASES:
        question_trigrams = _trigrams(_normalized_words(question))
        for page, phrase in evidence_items:
            matching_chunks = [
                chunk for chunk in chunks
                if chunk.page <= page <= (chunk.page_end or chunk.page)
                and " ".join(phrase.casefold().split()) in " ".join(chunk.text.casefold().split())
            ]
            if not matching_chunks:
                raise ValueError(f"Evidence not found on page {page}: {phrase!r}")
            if question_trigrams & _trigrams(_normalized_words(phrase)):
                raise ValueError(f"Question copies a three-word evidence phrase: {question!r}")


def generate(pdf: Path, output: Path) -> int:
    chunks, _, _ = app.parse_pdf(pdf.read_bytes())
    validate_cases(chunks)
    lines = []
    for question, evidence_items in CASES:
        lines.append(json.dumps({
            "question": question,
            "relevant": [
                {"contains": phrase, "page": page}
                for page, phrase in evidence_items
            ],
        }, ensure_ascii=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verified paraphrased RAG evaluation cases.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval_questions.jsonl"))
    args = parser.parse_args()
    count = generate(args.pdf, args.output)
    print(f"Wrote {count} verified evaluation cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

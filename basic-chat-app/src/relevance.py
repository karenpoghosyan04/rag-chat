"""Small, interpretable supervised relevance experiment, isolated from chat."""
from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PairFeatures(BaseEstimator, TransformerMixin):
    """Extract lexical features without learning from validation/test labels.

    Inputs are (question, passage) tuples. Scaling is fitted inside the training
    pipeline, separately in every cross-validation fold.
    """

    names = ("query_coverage", "jaccard", "length_ratio", "log_passage_words")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for question, passage in X:
            q = set(re.findall(r"[a-z0-9]+", question.casefold()))
            p = set(re.findall(r"[a-z0-9]+", passage.casefold()))
            rows.append([
                len(q & p) / max(1, len(q)),
                len(q & p) / max(1, len(q | p)),
                len(q) / max(1, len(p)),
                np.log1p(len(passage.split())),
            ])
        result = np.asarray(rows, dtype=np.float64).reshape(-1, len(self.names))
        if not np.isfinite(result).all():
            raise ValueError("Pair features must be finite.")
        return result

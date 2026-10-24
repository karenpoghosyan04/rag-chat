"""Leakage and quality-gate checks for the new experiment workflow."""
import copy
import json
import unittest

import numpy as np

import rag_eval
from check_quality import ROOT, check
from src.relevance import PairFeatures
from train_relevance import question_split


class CapstoneTests(unittest.TestCase):
    def test_training_never_uses_prior_holdout_questions(self):
        cases = rag_eval.load_cases(ROOT / "eval_questions.jsonl")
        excluded = set(json.loads((ROOT / "embedding-eval/calibration.json").read_text())["dataset"]["held_out_case_numbers"])
        train, validation = question_split(cases, excluded)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), {i for i in range(len(cases)) if i + 1 not in excluded})
        for ids in (train, validation):
            self.assertEqual({bool(cases[i].relevant) for i in ids}, {True, False})

    def test_empty_and_disjoint_pairs_have_finite_features(self):
        features = PairFeatures().transform([("", ""), ("attention", "trees grow"), ("attention", "attention works")])
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[1, 0], 0)
        self.assertEqual(features[2, 0], 1)

    def test_live_logic_is_protected(self):
        check()

    def test_gate_detects_a_per_question_loss_even_when_average_is_unchanged(self):
        from replay_quality import assert_no_regression
        old = {"metrics": {"recall@5": 1.0}, "details": [{"question": "q", "first_relevant_rank": 1, "unanswerable_outcome": None}]}
        new = copy.deepcopy(old)
        new["details"][0]["first_relevant_rank"] = 2
        with self.assertRaisesRegex(ValueError, "regression"):
            assert_no_regression(old, new)

    def test_gate_detects_lost_abstention(self):
        from replay_quality import assert_no_regression
        old = {"metrics": {}, "details": [{"question": "q", "first_relevant_rank": None, "unanswerable_outcome": "correct_rejection"}]}
        new = copy.deepcopy(old)
        new["details"][0]["unanswerable_outcome"] = "gated_false_positive"
        with self.assertRaisesRegex(ValueError, "regression"):
            assert_no_regression(old, new)

"""Selection and cache behavior for reproducible retrieval tuning."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app
import rag_eval


class TuningTests(unittest.TestCase):
    def run_config(self, recall1=.6, recall3=.92, precision1=.6, precision3=.4, rejection=1.):
        return {'settings': app.DEFAULT_SETTINGS.to_dict(),
                'metrics': {'recall@1': recall1, 'recall@3': recall3,
                            'precision@1': precision1, 'precision@3': precision3,
                            'no_answer_accuracy': rejection, 'mrr@3': .8}}

    def test_precision_cannot_win_by_losing_recall_or_refusals(self):
        safe = self.run_config()
        low_recall = self.run_config(recall3=.7, precision3=.6)
        unsafe = self.run_config(precision3=.8, rejection=.5)
        best = rag_eval.select_best_settings([low_recall, unsafe, safe], [1, 3], .9, .85)
        self.assertTrue(best['eligible'])
        self.assertEqual(best['k'], 3)
        self.assertEqual(best['metrics'], safe['metrics'])
        self.assertEqual(best['settings']['max_context_chunks'], 3)

    def test_selection_compares_context_count_and_accepts_exact_floor(self):
        best = rag_eval.select_best_settings([self.run_config(recall1=.9, precision1=.9)], [1, 3], .9, 1.)
        self.assertEqual(best['k'], 1)
        self.assertTrue(best['eligible'])

    def test_missing_negative_labels_prevent_eligible_export(self):
        best = rag_eval.select_best_settings([self.run_config(rejection=None)], [1, 3], .85, .85)
        self.assertFalse(best['eligible'])

    def test_impossible_target_is_explicit(self):
        best = rag_eval.select_best_settings([self.run_config()], [1, 3], 1., 1.)
        self.assertFalse(best['eligible'])

    def test_benchmark_reuses_pairs_without_changing_live_index(self):
        chunks = [app.DocumentChunk('Cats sleep in homes.', 1, 1), app.DocumentChunk('Rockets orbit planets.', 2, 2)]
        cases = [rag_eval.EvalCase('Where do cats sleep?', (rag_eval.RelevantEvidence(page=1),))]
        settings = app.RagSettings.from_dict({'hybrid_candidates': 4, 'embedding_min_similarity': 0.})
        before = app.DOCUMENT_INDEX.status()
        with TemporaryDirectory() as temp:
            pdf = Path(temp) / 'test.pdf'
            pdf.write_bytes(b'fixture')
            with patch.object(app, 'parse_pdf', return_value=(chunks, 2, set())), \
                 patch.object(app, 'embed_texts', side_effect=lambda texts, **kw: [[1., 0.] for _ in texts]), \
                 patch.object(app, 'rerank_texts', side_effect=lambda q, docs: [.9 for _ in docs]) as ranker:
                report = rag_eval.run_benchmark(pdf, cases, settings, [1, 3], [256], [0], [.25, .5, .75])
        self.assertEqual(len(report['runs']), 3)
        self.assertEqual(ranker.call_count, 1)
        self.assertEqual(app.DOCUMENT_INDEX.status(), before)
        self.assertIn('not end-to-end', report['timing_scope'])


if __name__ == '__main__':
    unittest.main()

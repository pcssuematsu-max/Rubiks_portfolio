import unittest

import numpy as np

from managers.debug_analysis import DebugAnalysisManager


class _NormalizeAI:
    def __init__(self):
        self.params = {
            'W1': np.asarray(((-1.0, 1.0), (-2.0, 2.0)), dtype = 'f'),
            'B1': np.asarray((2.0, 4.0), dtype = 'f'),
            'WQ1': np.asarray(((-3.0, 3.0), (0.0, 0.0)), dtype = 'f'),
            'WK1': np.asarray(((-4.0, 4.0),), dtype = 'f'),
            'WV1': np.asarray(((-5.0, 5.0),), dtype = 'f'),
            'BNg1': np.asarray((3.0, 4.0), dtype = 'f'),
            'BNb1': np.asarray((3.0, 4.0), dtype = 'f'),
        }
        self.v = {
            key: np.full_like(value, 7.0)
            for key,value in self.params.items()
        }
        self.dirty_calls = 0
        self.perfect_calls = 0

    def mark_params_dirty(self):
        self.dirty_calls += 1

    def set_perfect_val(self):
        self.perfect_calls += 1


class _NormalizeFrame:
    def __init__(self, ai):
        self.AIs = [ai]
        self.grad_index = 0
        self.grad_mode = 'Grad'
        self.grad_layer = 'WO_V'


class DebugNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.ai = _NormalizeAI()
        self.manager = DebugAnalysisManager(_NormalizeFrame(self.ai))

    def test_normalize_includes_transformer_qkv_weights(self):
        summary = self.manager.normalize(0)

        self.assertEqual(summary['normalized_rows']['W1'], 2)
        self.assertEqual(summary['normalized_rows']['WQ1'], 1)
        self.assertEqual(summary['normalized_rows']['WK1'], 1)
        self.assertEqual(summary['normalized_rows']['WV1'], 1)
        self.assertEqual(summary['skipped_rows']['WQ1'], 1)
        self.assertTrue(np.all(np.isfinite(self.ai.params['WQ1'])))
        self.assertTrue(np.allclose(self.ai.params['WQ1'][1], (0.0, 0.0)))
        self.assertTrue(np.allclose(self.ai.v['WQ1'][1], (7.0, 7.0)))
        self.assertTrue(np.allclose(self.ai.v['WQ1'][0], (0.0, 0.0)))

    def test_normalize_rescales_matching_bias_and_resets_optimizer_state(self):
        self.manager.normalize(0)

        self.assertTrue(np.allclose(self.ai.params['B1'], (2.0, 2.0)))
        self.assertTrue(np.all(self.ai.v['W1'] == 0.0))
        self.assertTrue(np.all(self.ai.v['B1'] == 0.0))
        self.assertTrue(np.all(self.ai.params['BNg1'] == 1.0))
        self.assertTrue(np.all(self.ai.params['BNb1'] == 0.0))
        self.assertEqual(self.ai.dirty_calls, 1)
        self.assertEqual(self.ai.perfect_calls, 1)

    def test_normalization_summary_reports_zero_variance_rows(self):
        summary = self.manager.normalize(0)

        text = self.manager.normalization_summary_text(summary)

        self.assertIn('skipped zero/non-finite rows=WQ1:1', text)


if __name__ == '__main__':
    unittest.main()

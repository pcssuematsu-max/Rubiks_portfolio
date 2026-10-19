import numpy as np
import unittest

from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config


class TransformerWeightDecayTests(unittest.TestCase):
    def test_transformer_decay_rates_match_linear_search_modes(self):
        config = build_experiment_frame_config()
        for index, mode in enumerate(config.ai_search_modes):
            expected = 1.0e-7 if mode == 'search2' else 1.0e-5
            self.assertEqual(config.wdlrs[index], expected)


    def test_torch_update_does_not_decay_policy_or_value_heads(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.params = {
            key: np.ones((1, 1), dtype='f')
            for key in ('WO_P', 'WM_P', 'WO_V', 'WM_V', 'W1')
        }
        ai.affines = ['Aff1']
        ai.attentions = []
        ai.transformer_linear_skip = False
        ai.weight_decay = True
        ai.wdlr = 0.1
        ai.v = {}
        ai.h = {}

        ai._apply_torch_grad_updates({})

        self.assertEqual(ai.params['WO_P'][0, 0], 1.0)
        self.assertEqual(ai.params['WM_P'][0, 0], 1.0)
        self.assertEqual(ai.params['WO_V'][0, 0], 1.0)
        self.assertEqual(ai.params['WM_V'][0, 0], 1.0)
        self.assertTrue(np.isclose(ai.params['W1'][0, 0], 0.9))

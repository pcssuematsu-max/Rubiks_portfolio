import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from cube.search3_engine import Search3Engine
from ai.transformer_variance import (
    RESIDUAL_BRANCH_TARGET_VARIANCE,
    TRANSFORMER_INPUT_TARGET_VARIANCE,
    TRANSFORMER_QUERY_KEY_TARGET_VARIANCE,
    TRANSFORMER_VALUE_TARGET_VARIANCE,
)


class TransformerInitializationTests(unittest.TestCase):
    def test_wo_v_initial_variance_is_configurable_for_both_model_types(self):
        random_state = np.random.get_state()
        try:
            for transformer in (False, True):
                common = dict(
                    cube_size=3, residual=True,
                    use_transformer_attention=transformer,
                    transformer_attention_token_mode='piece',
                )
                np.random.seed(9876)
                baseline = Rubiks_3_AI(
                    [64, 64], search_mode='search2',
                    wo_v_initial_variance=2.0, **common,
                )
                np.random.seed(9876)
                search3 = Rubiks_3_AI(
                    [64, 64], search_mode='search3',
                    wo_v_initial_variance=2.0, **common,
                )
                np.random.seed(9876)
                smaller = Rubiks_3_AI(
                    [64, 64], search_mode='search3',
                    wo_v_initial_variance=0.5, **common,
                )
                np.testing.assert_array_equal(search3.params['WO_V'], baseline.params['WO_V'])
                np.testing.assert_allclose(smaller.params['WO_V'], baseline.params['WO_V'] * 0.5)
        finally:
            np.random.set_state(random_state)

    def test_model_type_follows_transformer_attention_configuration(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.use_transformer_attention = False
        self.assertFalse(ai.is_transformer)
        ai.use_transformer_attention = True
        self.assertTrue(ai.is_transformer)

    def test_transformer_budget_result_uses_value_bounded_prefix(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.use_transformer_attention = True
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = ai
        engine._value_trace_for_moves = lambda root, moves: [root, 0.1, 0.4, 0.3]
        moves = ('R', 'U', 'F')

        self.assertEqual(engine._bounded_budget_moves(0.2, moves), ('R', 'U'))
        ai.use_transformer_attention = False
        self.assertEqual(engine._bounded_budget_moves(0.2, moves), moves)

    def test_piece_transformer_initialization_matches_normalization_targets(self):
        random_state = np.random.get_state()
        np.random.seed(12345)
        try:
            ai = Rubiks_3_AI(
                [64, 64],
                cube_size = 3,
                residual = True,
                use_transformer_attention = True,
                transformer_attention_dim = 64,
                transformer_attention_token_mode = 'piece',
            )
        finally:
            np.random.set_state(random_state)

        expected = {
            'W1': TRANSFORMER_INPUT_TARGET_VARIANCE,
            'W2': RESIDUAL_BRANCH_TARGET_VARIANCE,
            'WQ1': TRANSFORMER_QUERY_KEY_TARGET_VARIANCE,
            'WK1': TRANSFORMER_QUERY_KEY_TARGET_VARIANCE,
            'WV1': TRANSFORMER_VALUE_TARGET_VARIANCE,
        }
        for key,target in expected.items():
            weights = ai.params[key]
            observed = float(np.var(weights) * weights.shape[1])
            self.assertAlmostEqual(observed, target, delta = target * 0.15)


if __name__ == '__main__':
    unittest.main()

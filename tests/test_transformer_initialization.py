import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from ai.transformer_variance import (
    RESIDUAL_BRANCH_TARGET_VARIANCE,
    TRANSFORMER_INPUT_TARGET_VARIANCE,
    TRANSFORMER_QUERY_KEY_TARGET_VARIANCE,
    TRANSFORMER_VALUE_TARGET_VARIANCE,
)


class TransformerInitializationTests(unittest.TestCase):
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

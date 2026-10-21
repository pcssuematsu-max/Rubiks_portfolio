import unittest
from types import SimpleNamespace

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from ui.frame import Frame
from ui.frame_config import FrameConfig


class SiLUBetaInitialTests(unittest.TestCase):
    def test_configured_beta_initializes_trunk_and_heads(self):
        config = FrameConfig(ai_search_modes=('search2',), silu_beta_initials=(0.75,))
        model = Rubiks_3_AI([16, 16], silu_beta_initial=config.silu_beta_initials[0])

        self.assertEqual(len(model.activation_betas), 4)
        for key, layer in model.activation_betas:
            self.assertAlmostEqual(float(layer.beta[0]), 0.75)
            self.assertIs(model.params[key], layer.beta)

    def test_initialize_command_resets_beta_and_optimizer_state(self):
        model = Rubiks_3_AI([16, 16], silu_beta_initial=0.25)
        for key, _ in model.activation_betas:
            model.v[key][:] = 3
            model.h[key][:] = 4
        frame = SimpleNamespace(
            AIs=[model],
            _ai_silu_beta_initial=lambda index: 0.25,
        )

        Frame.initialize(frame, 0, beta=1.5)
        for key, layer in model.activation_betas:
            self.assertAlmostEqual(float(layer.beta[0]), 1.5)
            np.testing.assert_array_equal(model.v[key], [0])
            np.testing.assert_array_equal(model.h[key], [0])

        Frame.initialize(frame, 0)
        self.assertTrue(all(float(layer.beta[0]) == 0.25 for _, layer in model.activation_betas))

    def test_non_finite_beta_is_rejected(self):
        with self.assertRaises(ValueError):
            FrameConfig(ai_search_modes=('search2',), silu_beta_initials=(float('nan'),))
        with self.assertRaises(ValueError):
            Rubiks_3_AI([16, 16], silu_beta_initial=float('inf'))


if __name__ == '__main__':
    unittest.main()

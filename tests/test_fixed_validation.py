import unittest

import numpy as np

from ai.fixed_validation import FIXTURE_ID, evaluate_fixed_validation
from ai.rubiks_ai import Rubiks_3_AI


class FixedValidationTests(unittest.TestCase):
    def test_is_deterministic_and_restores_the_cube_state(self):
        ai = Rubiks_3_AI([8], cube_size = 3, search_mode = 'search2')
        ai.cube.scramble(3)
        state_before = ai.cube.state.copy()

        first = evaluate_fixed_validation(ai)
        second = evaluate_fixed_validation(ai)

        self.assertTrue(np.array_equal(ai.cube.state, state_before))
        self.assertEqual(first, second)
        self.assertEqual(first['fixtureId'], FIXTURE_ID)
        self.assertEqual(first['fixtureLengths'], [12, 24, 36])
        self.assertEqual(first['stateCount'], 216)
        self.assertIn('valuePathCrossEntropy', first)
        self.assertNotIn('valueMae', first)
        self.assertGreaterEqual(first['policyTop1Accuracy'], 0.0)
        self.assertLessEqual(first['policyTop1Accuracy'], 1.0)

    def test_search3_uses_probability_calibration_metrics(self):
        ai = Rubiks_3_AI([8], cube_size = 3, search_mode = 'search3')

        metrics = evaluate_fixed_validation(ai)

        self.assertIn('valueMae', metrics)
        self.assertIn('valueBce', metrics)
        self.assertNotIn('valuePathCrossEntropy', metrics)
        self.assertGreaterEqual(metrics['valueMae'], 0.0)


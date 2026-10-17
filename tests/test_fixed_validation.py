import unittest

import numpy as np

from ai.fixed_validation import FIXTURE_ID, _effect_decision_metrics, evaluate_fixed_validation
from ai.rubiks_ai import Rubiks_3_AI


class FixedValidationTests(unittest.TestCase):
    def test_effect_decision_metrics_exclude_terminal_value_jump(self):
        metrics = _effect_decision_metrics(
            np.asarray([[2.0,0.0,2.0],[0.0,2.0,0.0]],dtype='f'),
            np.asarray([0.0,1.0,0.5],dtype='f'),
            np.asarray([0,1,0]),
            (slice(0,3),),
            np.asarray([8,8,8]),
        )['8-13']
        self.assertEqual(metrics['stateCount'],3)
        self.assertEqual(metrics['valueStepCount'],2)
        self.assertEqual(metrics['policyTargetRankMean'],1.0)
        self.assertEqual(metrics['valueImprovementRate'],0.5)
        self.assertEqual(metrics['valueDeltaMean'],0.25)

    def test_is_deterministic_and_restores_the_cube_state(self):
        ai = Rubiks_3_AI([8], cube_size = 3, search_mode = 'search2')
        ai.cube.scramble(3)
        state_before = ai.cube.state.copy()

        first = evaluate_fixed_validation(ai)
        second = evaluate_fixed_validation(ai)

        self.assertTrue(np.array_equal(ai.cube.state, state_before))
        self.assertEqual(first, second)
        self.assertEqual(first['fixtureId'], FIXTURE_ID)
        self.assertEqual(first['fixtureLengths'], [12, 24, 36, 48, 60])
        self.assertEqual(first['stateCount'], 540)
        self.assertEqual(first['byLength']['12']['stateCount'], 36)
        self.assertEqual(first['byLength']['60']['stateCount'], 180)
        self.assertEqual(first['long40Plus']['fixtureCount'], 6)
        self.assertEqual(first['long40Plus']['stateCount'], 324)
        self.assertIn('policyTop1Accuracy', first['byLength']['48'])
        self.assertIn('valuePathCrossEntropy', first)
        self.assertNotIn('valueMae', first)
        self.assertGreaterEqual(first['policyTop1Accuracy'], 0.0)
        self.assertLessEqual(first['policyTop1Accuracy'], 1.0)
        self.assertEqual(sum(band['stateCount'] for band in first['byEffect'].values()), 540)
        self.assertEqual(sum(band['valueStepCount'] for band in first['byEffect'].values()), 525)
        self.assertTrue(all(0.0 <= band['valueImprovementRate'] <= 1.0 for band in first['byEffect'].values()))
        self.assertTrue(all(band['policyTargetRankMean'] >= 1.0 for band in first['byEffect'].values()))
        self.assertTrue(first['byEffectChoices'])
        self.assertTrue(all(1 <= band['sampleCount'] <= 4 for band in first['byEffectChoices'].values()))
        self.assertTrue(all(band['valueTargetRankMean'] >= 1.0 for band in first['byEffectChoices'].values()))
        self.assertTrue(all(0.0 <= band['valueTop1Accuracy'] <= 1.0 for band in first['byEffectChoices'].values()))

    def test_search3_uses_probability_calibration_metrics(self):
        ai = Rubiks_3_AI([8], cube_size = 3, search_mode = 'search3')

        metrics = evaluate_fixed_validation(ai)

        self.assertIn('valueMae', metrics)
        self.assertIn('valueBce', metrics)
        self.assertNotIn('valuePathCrossEntropy', metrics)
        self.assertGreaterEqual(metrics['valueMae'], 0.0)

    def test_steps_to_goal_reports_near_mid_and_far_remaining_step_bands(self):
        ai = Rubiks_3_AI([8], cube_size = 3, search_mode = 'search2')
        ai.set_search2_value_loss_type('steps_to_goal')

        metrics = evaluate_fixed_validation(ai)
        bands = metrics['stepsToGoalByRemainingSteps']
        exact = metrics['stepsToGoalExactDistance']

        self.assertGreater(exact['stateCount'], 1)
        self.assertGreaterEqual(exact['goalValueMae'], 0.0)
        self.assertGreaterEqual(exact['oneMoveValueMae'], 0.0)

        self.assertEqual(set(bands), {'0-10','11-30','31+'})
        self.assertEqual(sum(band['stateCount'] for band in bands.values()), metrics['stateCount'])
        self.assertTrue(all('valueMae' in band for band in bands.values()))
        self.assertTrue(all('valueSignedError' in band for band in bands.values()))
        self.assertTrue(all('valueRmse' in band for band in bands.values()))

    def test_transformer_branch_probe_is_read_only_and_keeps_full_prediction(self):
        from ai.rubiks_ai import torch
        if torch is None:
            self.skipTest('PyTorch is unavailable')
        ai = Rubiks_3_AI(
            [16,16],cube_size=3,search_mode='search3',residual=True,
            use_transformer_attention=True,transformer_attention_token_mode='piece',
            transformer_linear_skip=True,
        )
        state_before = ai.cube.state.copy()
        weights_before = {key:value.copy() for key,value in ai.params.items()}
        prediction_before = ai.predict(ai.cube.makedata().reshape(-1,1))

        metrics = evaluate_fixed_validation(ai)

        self.assertEqual(metrics['transformerBranches']['sampleCount'], 45)
        self.assertIn('withoutAttention',metrics['transformerBranches'])
        self.assertIn('withoutLinear',metrics['transformerBranches'])
        np.testing.assert_array_equal(ai.cube.state,state_before)
        for key,value in weights_before.items():
            np.testing.assert_array_equal(ai.params[key],value)
        np.testing.assert_allclose(ai.predict(ai.cube.makedata().reshape(-1,1)),prediction_before)

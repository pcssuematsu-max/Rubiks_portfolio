import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from core.learning_history import LearningHistoryStore, completed_learning_record


class _FakeAI:
    search_mode = 'search3'
    lr = 0.001
    lr_C = 0.25
    lr_v = 0.9
    lr_h = 0.8
    update_scale_shared = 1.0
    update_scale_policy = 0.5
    update_scale_value = 2.0
    search2_value_loss_type = 'myloss2'
    last_training_metrics = {
        'policyLoss': 0.12,
        'valueLoss': 0.34,
        'policyCePerState': 1.23,
        'policyEffectiveStateCount': 160.0,
        'valueBcePerState': 0.56,
        'valueMae': 0.12,
        'valueStartToEndDelta': 0.31,
        'valueTargetStartToEndDelta': 0.45,
        'valueEffectiveStateCount': 240.5,
        'valueSequenceCount': 80,
        'updatesDuringSolve': 8,
        'trainingDataCount': 100,
        'retainedDataCount': 60,
    }


class LearningHistoryTests(unittest.TestCase):
    def test_writes_a_learning_record_with_loss_settings_and_update_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            record = completed_learning_record(3, _FakeAI(), 1.23456789)
            LearningHistoryStore(path).append(record)

            payload = json.loads(path.read_text(encoding = 'utf-8'))
            saved = payload['records'][0]
            self.assertEqual(payload['schemaVersion'], 1)
            self.assertEqual(saved['aiIndex'], 3)
            self.assertEqual(saved['searchMode'], 'search3')
            self.assertEqual(saved['policyLoss'], 0.12)
            self.assertEqual(saved['policyCePerState'], 1.23)
            self.assertEqual(saved['valueBcePerState'], 0.56)
            self.assertEqual(saved['valueMae'], 0.12)
            self.assertEqual(saved['valueStartToEndDelta'], 0.31)
            self.assertEqual(saved['valueSequenceCount'], 80)
            self.assertEqual(saved['updatesDuringSolve'], 8)
            self.assertEqual(saved['learningRate']['base'], 0.001)
            self.assertEqual(saved['updateScales']['value'], 2.0)

    def test_writes_fixed_validation_metrics_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.last_training_metrics = dict(ai.last_training_metrics)
            ai.last_training_metrics['fixedValidation'] = {
                'fixtureId': 'canonical-inverse-v1',
                'fixtureLengths': [12, 24, 36],
                'fixtureCount': 9,
                'stateCount': 216,
                'policyTop1Accuracy': 0.2,
                'policyTop3Accuracy': 0.5,
                'policyTargetProbability': 0.1,
                'policyCrossEntropy': 2.3,
                'valueRankCorrelation': 0.4,
                'valuePearsonCorrelation': 0.5,
                'valuePredictionMean': 0.6,
                'valuePredictionStd': 0.2,
                'valueTargetMean': 0.3,
                'valueTargetStd': 0.1,
                'valueMae': 0.2,
                'valueBce': 0.7,
                'valuePathCrossEntropy': None,
            }
            record = completed_learning_record(3, ai, 1.0)
            LearningHistoryStore(path).append(record)

            saved = LearningHistoryStore(path).records()[0]

            self.assertEqual(saved['fixedValidation']['fixtureId'], 'canonical-inverse-v1')
            self.assertEqual(saved['fixedValidation']['stateCount'], 216)
            self.assertAlmostEqual(saved['fixedValidation']['valueRankCorrelation'], 0.4)

    def test_ai_training_metrics_uses_processed_batch_count_as_update_count(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(2.0, 4.0, 30, 18, 4)

        self.assertEqual(ai.last_training_metrics['policyLoss'], 0.5)
        self.assertEqual(ai.last_training_metrics['valueLoss'], 1.0)
        self.assertEqual(ai.last_training_metrics['updatesDuringSolve'], 4)
        self.assertEqual(ai.last_training_metrics['trainingDataCount'], 30)
        self.assertEqual(ai.last_training_metrics['retainedDataCount'], 18)
        self.assertIsNone(ai.last_training_metrics['valueBcePerState'])

    def test_search3_quality_metrics_are_weighted_by_state_and_keep_sequence_deltas(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        metrics = ai._search3_quality_metrics(
            # The final row is the value logit.  Every policy action and
            # value logit is zero, so both CE values have closed-form checks.
            np.zeros((3, 3), dtype = 'f'),
            {
                'policy_targets': np.array([
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                ], dtype = 'f'),
                'policy_weights': np.array([[1.0, 2.0, 0.0]], dtype = 'f'),
                'value_targets': np.array([[0.0, 1.0, 0.5]], dtype = 'f'),
                'sample_weights': np.array([[1.0, 2.0, 1.0]], dtype = 'f'),
                'value_indices': [0, 2, 3],
            },
        )
        ai._record_training_metrics(2.0, 4.0, 30, 18, 4, metrics)

        self.assertAlmostEqual(ai.last_training_metrics['policyCePerState'], 0.693147, places = 5)
        self.assertEqual(ai.last_training_metrics['policyEffectiveStateCount'], 3.0)
        self.assertAlmostEqual(ai.last_training_metrics['valueBcePerState'], 0.693147, places = 5)
        self.assertAlmostEqual(ai.last_training_metrics['valueMae'], 0.375, places = 6)
        self.assertAlmostEqual(ai.last_training_metrics['valueStartToEndDelta'], 0.0, places = 6)
        self.assertAlmostEqual(ai.last_training_metrics['valueTargetStartToEndDelta'], 0.5, places = 6)
        self.assertEqual(ai.last_training_metrics['valueSequenceCount'], 2)

    def test_legacy_history_record_remains_readable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            legacy_record = completed_learning_record(3, _FakeAI(), 1.0)
            for field in (
                'policyCePerState', 'policyEffectiveStateCount',
                'valueBcePerState', 'valueMae', 'valueStartToEndDelta',
                'valueTargetStartToEndDelta', 'valueEffectiveStateCount',
                'valueSequenceCount',
            ):
                del legacy_record[field]
            path.write_text(json.dumps({
                'schemaVersion': 1,
                'updatedAt': legacy_record['timestamp'],
                'records': [legacy_record],
            }), encoding = 'utf-8')

            records = LearningHistoryStore(path).records()

            self.assertEqual(len(records), 1)
            self.assertNotIn('valueBcePerState', records[0])

    def test_ai_training_metrics_marks_loss_as_unavailable_when_no_update_ran(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(0.0, 0.0, 5, 5, 0)

        self.assertIsNone(ai.last_training_metrics['policyLoss'])
        self.assertIsNone(ai.last_training_metrics['valueLoss'])

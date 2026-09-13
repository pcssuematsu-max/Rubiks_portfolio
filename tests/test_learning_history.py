import json
from pathlib import Path
import tempfile
import unittest

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
            self.assertEqual(saved['updatesDuringSolve'], 8)
            self.assertEqual(saved['learningRate']['base'], 0.001)
            self.assertEqual(saved['updateScales']['value'], 2.0)

    def test_ai_training_metrics_uses_processed_batch_count_as_update_count(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(2.0, 4.0, 30, 18, 4)

        self.assertEqual(ai.last_training_metrics['policyLoss'], 0.5)
        self.assertEqual(ai.last_training_metrics['valueLoss'], 1.0)
        self.assertEqual(ai.last_training_metrics['updatesDuringSolve'], 4)
        self.assertEqual(ai.last_training_metrics['trainingDataCount'], 30)
        self.assertEqual(ai.last_training_metrics['retainedDataCount'], 18)

    def test_ai_training_metrics_marks_loss_as_unavailable_when_no_update_ran(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(0.0, 0.0, 5, 5, 0)

        self.assertIsNone(ai.last_training_metrics['policyLoss'])
        self.assertIsNone(ai.last_training_metrics['valueLoss'])

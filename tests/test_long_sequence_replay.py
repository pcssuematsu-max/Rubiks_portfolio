import random
from types import SimpleNamespace
import unittest

from ai.rubiks_ai import Rubiks_3_AI


class LongSequenceReplayTests(unittest.TestCase):
    def test_reserved_batches_include_old_long_trajectories(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 4
        ai.train_recent_ratio = 0.25
        ai.train_long_sequence_min_steps = 20
        ai.train_long_sequence_ratio = 0.5
        data = [
            SimpleNamespace(moves = tuple(range(length)))
            for length in (5, 8, 20, 21, 30, 35, 6, 7)
        ]

        random.seed(7)
        batches, remainder = ai._build_sampled_training_batches(
            list(range(len(data))),
            data,
            batch_size = 1,
            state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        selected = [index for batch in batches for index in batch]
        summary = ai._last_training_sample_summary
        self.assertEqual(len(selected), 4)
        self.assertEqual(summary['recent_batches'], 1)
        self.assertEqual(summary['long_batches'], 2)
        self.assertEqual(summary['long_min_steps'], 20)
        self.assertEqual(summary['long_eligible_items'], 4)
        self.assertGreaterEqual(sum(len(data[index].moves) >= 20 for index in selected), 2)
        self.assertEqual(set(selected) | set(remainder), set(range(len(data))))
        self.assertEqual(set(selected) & set(remainder), set())

    def test_missing_long_data_falls_back_to_random_batches(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 4
        ai.train_recent_ratio = 0.5
        ai.train_long_sequence_min_steps = 20
        ai.train_long_sequence_ratio = 0.25
        data = [SimpleNamespace(moves = tuple(range(5))) for _ in range(8)]

        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))),
            data,
            batch_size = 1,
            state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        summary = ai._last_training_sample_summary
        self.assertEqual(len(batches), 4)
        self.assertEqual(summary['long_batches'], 0)
        self.assertEqual(summary['random_batches'], 2)

    def test_replay_uses_remaining_solution_length_for_short_segments(self):
        item = SimpleNamespace(moves = (0, 1), steps_to_goal = 42)

        self.assertEqual(Rubiks_3_AI._replay_sequence_steps(item), 42)

    def test_replay_keeps_search2_distance_tuples_as_local_move_lengths(self):
        item = SimpleNamespace(moves = (0, 1, 2), steps_to_goal = (3, 2, 1, 0))

        self.assertEqual(Rubiks_3_AI._replay_sequence_steps(item), 3)

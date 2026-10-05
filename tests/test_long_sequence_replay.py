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
        self.assertEqual(summary['long_reserved_items'], 2)
        self.assertGreaterEqual(summary['long_selected_items'], 2)
        self.assertGreaterEqual(summary['long_selected_step_mean'], 20)
        self.assertGreaterEqual(sum(len(data[index].moves) >= 20 for index in selected), 2)
        self.assertEqual(set(selected) | set(remainder), set(range(len(data))))
        self.assertEqual(set(selected) & set(remainder), set())
        self.assertEqual(
            ai._training_sample_history_metrics()['longReservedItemCount'],
            2,
        )

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

    def test_medium_reservation_and_long_cap_balance_the_effective_mix(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 10
        ai.train_recent_ratio = 0.5
        ai.train_short_sequence_max_steps = 19
        ai.train_short_sequence_min_ratio = 0.2
        ai.train_medium_sequence_min_steps = 20
        ai.train_medium_sequence_max_steps = 29
        ai.train_medium_sequence_ratio = 0.2
        ai.train_long_sequence_min_steps = 30
        ai.train_long_sequence_ratio = 0.2
        ai.train_long_sequence_max_ratio = 0.6
        data = [
            SimpleNamespace(moves = tuple(range(length)))
            for length in ([8] * 10 + [24] * 10 + [40] * 10)
        ]

        random.seed(3)
        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))),
            data,
            batch_size = 1,
            state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        selected = [index for batch in batches for index in batch]
        summary = ai._last_training_sample_summary
        self.assertEqual(len(selected), 10)
        self.assertEqual(summary['medium_reserved_items'], 2)
        self.assertGreaterEqual(summary['medium_selected_items'], 2)
        self.assertGreaterEqual(summary['short_selected_items'], 2)
        self.assertGreaterEqual(summary['short_selected_ratio'], 0.2)
        self.assertLessEqual(summary['long_selected_items'], 6)
        self.assertLessEqual(summary['long_selected_ratio'], 0.6)

    def test_short_minimum_replaces_unreserved_hard_items(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_short_sequence_min_ratio = 0.4
        data = [
            SimpleNamespace(moves = tuple(range(length)))
            for length in ([8] * 4 + [40] * 6)
        ]
        batches = [[4], [5], [6], [7], [8]]

        random.seed(4)
        selected = ai._enforce_short_replay_minimum(
            batches,
            list(range(len(data))),
            data,
            protected_items = [],
            short_max_steps = 19,
        )

        selected_indices = [index for batch in selected for index in batch]
        self.assertGreaterEqual(sum(index < 4 for index in selected_indices), 2)

    def test_replay_uses_remaining_solution_length_for_short_segments(self):
        item = SimpleNamespace(moves = (0, 1), steps_to_goal = 42)

        self.assertEqual(Rubiks_3_AI._replay_sequence_steps(item), 42)

    def test_replay_summary_records_direct_and_fallback_trajectory_counts(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 4
        ai.train_recent_ratio = 1.0
        data = [
            SimpleNamespace(moves = (0,), trajectory_source = 'direct-search'),
            SimpleNamespace(moves = (0,), trajectory_source = 'search3-fallback-prefix'),
            SimpleNamespace(moves = (0,), trajectory_source = 'greedy-fallback'),
            SimpleNamespace(moves = (0,), trajectory_source = 'bootstrap'),
        ]

        ai._build_sampled_training_batches(
            list(range(len(data))),
            data,
            batch_size = 1,
            state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        summary = ai._last_training_sample_summary
        self.assertEqual(summary['direct_search_selected_items'], 1)
        self.assertEqual(summary['fallback_selected_items'], 2)
        history = ai._training_sample_history_metrics()
        self.assertEqual(history['directSearchSelectedItemCount'], 1)
        self.assertEqual(history['fallbackSelectedItemCount'], 2)

    def test_replay_keeps_search2_distance_tuples_as_local_move_lengths(self):
        item = SimpleNamespace(moves = (0, 1, 2), steps_to_goal = (3, 2, 1, 0))

        self.assertEqual(Rubiks_3_AI._replay_sequence_steps(item), 3)

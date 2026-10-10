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

    def test_pairwise_replay_caps_fallback_origin_when_direct_data_exists(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'myloss2_pairwise'
        ai.pairwise_fallback_max_ratio = 0.25
        data = [
            SimpleNamespace(moves = (0,), trajectory_origin = 'fallback')
            for _ in range(3)
        ] + [
            SimpleNamespace(moves = (0,), trajectory_origin = 'direct_search')
            for _ in range(4)
        ]

        batches = ai._cap_pairwise_fallback_replay_selection(
            [[0], [1], [2], [3]], list(range(len(data))), data,
        )

        selected = [index for batch in batches for index in batch]
        self.assertLessEqual(
            sum(data[index].trajectory_origin == 'fallback' for index in selected),
            1,
        )

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

    def test_replay_summary_prefers_current_trajectory_origin_metadata(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 3
        ai.train_recent_ratio = 1.0
        data = [
            SimpleNamespace(moves = (0,), trajectory_origin = 'direct_search'),
            SimpleNamespace(moves = (0,), trajectory_origin = 'fallback'),
            SimpleNamespace(moves = (0,), trajectory_origin = 'search3-fallback-prefix'),
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

    def test_replay_keeps_search2_distance_tuples_as_local_move_lengths(self):
        item = SimpleNamespace(moves = (0, 1, 2), steps_to_goal = (3, 2, 1, 0))

        self.assertEqual(Rubiks_3_AI._replay_sequence_steps(item), 3)

    def test_steps_to_goal_stratified_replay_interleaves_distance_and_efficiency(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 6
        ai.train_recent_ratio = 0.0
        ai.train_short_sequence_max_steps = 0
        ai.train_short_sequence_min_ratio = 0.0
        ai.train_medium_sequence_min_steps = 0
        ai.train_medium_sequence_max_steps = 0
        ai.train_medium_sequence_ratio = 0.0
        ai.train_long_sequence_min_steps = 0
        ai.train_long_sequence_ratio = 0.0
        ai.train_long_sequence_max_ratio = 0.0
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_replay_stratified_ratio = 1.0
        data = [
            SimpleNamespace(scramble = tuple(range(10)), moves = tuple(range(length)))
            for length in (8, 15, 35, 16, 25, 45)
        ]

        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))), data, batch_size = 1, state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        selected = [index for batch in batches for index in batch]
        summary = ai._last_training_sample_summary
        self.assertEqual(len(selected), 6)
        self.assertEqual(summary['steps_to_goal_stratified_batches'], 6)
        self.assertEqual(summary['steps_to_goal_stratified_items'], 6)
        self.assertEqual(summary['random_batches'], 0)

    def test_steps_to_goal_replay_diagnostics_detect_conflicting_start_targets(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_replay_stratified_ratio = 0.5
        ai._last_training_sample_summary = {
            'steps_to_goal_stratified_batches': 1,
            'steps_to_goal_stratified_items': 2,
        }
        data = [
            SimpleNamespace(scramble = ('R',), moves = ('A', 'B')),
            SimpleNamespace(scramble = ('R',), moves = ('A', 'B', 'C', 'D')),
            SimpleNamespace(scramble = ('U', 'F'), moves = ('A', 'B')),
        ]

        replay = ai._steps_to_goal_replay_history_metrics(
            available_indices = [0,1,2], selected_indices = [0,1], data_source = data,
        )

        self.assertEqual(replay['selected']['duplicateStartStateCount'], 1)
        self.assertEqual(replay['selected']['conflictingStartStateCount'], 1)
        self.assertGreater(replay['selected']['duplicateTargetStdMean'], 0.0)
        self.assertEqual(replay['selected']['shortestTargetRouteCount'], 1)
        self.assertEqual(replay['selected']['longerDuplicateRouteCount'], 1)

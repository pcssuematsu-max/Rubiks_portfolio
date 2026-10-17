import random
from types import SimpleNamespace
import unittest

from ai.rubiks_ai import Rubiks_3_AI


class LongSequenceReplayTests(unittest.TestCase):
    def test_mixed_batches_preserve_items_and_put_both_ages_in_every_batch(self):
        batches = [[8, 9, 10, 11], [12, 13, 14, 15], [0, 1, 2, 3], [4, 5, 6, 7]]
        mixed = Rubiks_3_AI._mix_recent_old_batches(batches, recent_window_floor=8)
        self.assertEqual([len(batch) for batch in mixed], [4] * 4)
        self.assertEqual(sorted(index for batch in mixed for index in batch), list(range(16)))
        self.assertTrue(all(sum(index >= 8 for index in batch) == 2 for batch in mixed))

    def test_recent_and_old_mixing_is_logged_after_sampling(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 4
        ai.train_recent_ratio = 0.5
        ai.train_batch_mix_mode = 'mixed'
        data = [SimpleNamespace(moves = ('R',)) for _ in range(12)]
        random.seed(9)
        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))),data,
            batch_size=2,state_batch_size=0,
            state_count_fn=lambda item: len(item.moves) + 1,
        )
        ordered_ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ordered_ai.train_max_batches = 4
        ordered_ai.train_recent_ratio = 0.5
        ordered_ai.train_batch_mix_mode = 'ordered'
        random.seed(9)
        ordered_batches, _ = ordered_ai._build_sampled_training_batches(
            list(range(len(data))),data,
            batch_size=2,state_batch_size=0,
            state_count_fn=lambda item: len(item.moves) + 1,
        )
        self.assertEqual(
            sorted(index for batch in batches for index in batch),
            sorted(index for batch in ordered_batches for index in batch),
        )
        summary = ai._last_training_sample_summary
        self.assertEqual(len(batches), 4)
        self.assertEqual(summary['mixed_batches'], 4)
        self.assertEqual(summary['recent_window_selected_items'], 4)
        self.assertEqual(summary['selected_unique_items'], 8)
        history = ai._training_sample_history_metrics()
        self.assertEqual(history['batchMixMode'], 'mixed')
        self.assertEqual(history['recentReplayRatio'], 0.5)
        self.assertEqual(history['mixedBatchCount'], 4)

    def test_reservations_never_exceed_the_batch_cap(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 4
        ai.train_recent_ratio = 0.75
        ai.train_medium_sequence_min_steps = 20
        ai.train_medium_sequence_max_steps = 29
        ai.train_medium_sequence_ratio = 0.25
        ai.train_long_sequence_min_steps = 30
        ai.train_long_sequence_ratio = 0.25
        data = [SimpleNamespace(moves=tuple(range(length))) for length in ([24] * 8 + [40] * 8)]
        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))),data,
            batch_size=1,state_batch_size=0,
            state_count_fn=lambda item: len(item.moves) + 1,
        )
        self.assertLessEqual(len(batches), 4)

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

    def test_steps_to_goal_long_route_minimum_oversamples_direct_canonical_routes(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.train_max_batches = 10
        ai.train_recent_ratio = 1.0
        ai.train_short_sequence_max_steps = 0
        ai.train_short_sequence_min_ratio = 0.0
        ai.train_medium_sequence_min_steps = 0
        ai.train_medium_sequence_max_steps = 0
        ai.train_medium_sequence_ratio = 0.0
        ai.train_long_sequence_min_steps = 0
        ai.train_long_sequence_ratio = 0.0
        ai.train_long_sequence_max_ratio = 0.0
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_replay_stratified_ratio = 0.0
        ai.steps_to_goal_long_route_min_ratio = 0.5
        data = [
            SimpleNamespace(
                scramble = ('short',index), moves = tuple(range(8)),
                trajectory_origin = 'direct_search',
            )
            for index in range(8)
        ] + [
            SimpleNamespace(
                scramble = ('long',index), moves = tuple(range(35)),
                trajectory_origin = 'direct_search',
            )
            for index in range(2)
        ]

        batches, _ = ai._build_sampled_training_batches(
            list(range(len(data))), data, batch_size = 1, state_batch_size = 0,
            state_count_fn = lambda item: len(item.moves) + 1,
        )

        selected = [index for batch in batches for index in batch]
        summary = ai._last_training_sample_summary
        self.assertEqual(len(selected), 10)
        self.assertEqual(summary['steps_to_goal_long_route_eligible_items'], 2)
        self.assertEqual(summary['steps_to_goal_long_route_required_items'], 5)
        self.assertEqual(summary['steps_to_goal_long_route_selected_items'], 5)
        self.assertEqual(summary['steps_to_goal_long_route_oversampled_items'], 3)
        self.assertEqual(sum(index >= 8 for index in selected), 5)
        history = ai._training_sample_history_metrics()
        self.assertEqual(history['stepsToGoalLongRouteMinRatio'], 0.5)
        self.assertEqual(history['stepsToGoalLongRouteSelectedItemCount'], 5)

    def test_original_transformer_reserves_quality_long_direct_routes_with_repeat_cap(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'myloss'
        ai.use_transformer_attention = True
        ai.original_long_direct_min_ratio = 0.5
        data = [
            SimpleNamespace(scramble = tuple(range(20)), moves = tuple(range(8)), trajectory_origin = 'direct_search')
            for _ in range(7)
        ] + [
            SimpleNamespace(scramble = tuple(range(20)), moves = tuple(range(35)), trajectory_origin = 'direct_search'),
            SimpleNamespace(scramble = tuple(range(10)), moves = tuple(range(35)), trajectory_origin = 'direct_search'),
            SimpleNamespace(scramble = tuple(range(20)), moves = tuple(range(35)), trajectory_origin = 'fallback'),
        ]
        batches = [[index] for index in range(len(data))]

        summary = ai._enforce_original_long_direct_minimum(batches,list(range(len(data))),data)

        selected = [index for batch in batches for index in batch]
        self.assertEqual(summary['eligible_item_count'],1)
        self.assertEqual(summary['required_item_count'],5)
        self.assertEqual(summary['selected_item_count'],2)
        self.assertEqual(selected.count(7),2)
        self.assertEqual(selected.count(8),1)
        self.assertEqual(selected.count(9),1)

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

from types import SimpleNamespace
import unittest

import numpy as np

from ai.losses import Huber
from ai.rubiks_ai import Rubiks_3_AI
from managers.search_data import SearchDataManager
from managers.solve_session import SolveSessionManager


class StepsToGoalValueTests(unittest.TestCase):
    def test_huber_is_quadratic_near_target_and_clipped_for_large_errors(self):
        loss = Huber()
        prediction = np.asarray([[-3.0, 2.0]], dtype = 'f')
        target = np.asarray([[-3.0, -1.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target), 2.5)
        np.testing.assert_allclose(loss.backward(), [[0.0, 1.0]])

    def test_huber_mean_reduction_keeps_longer_routes_from_scaling_one_update(self):
        loss = Huber(reduction = 'mean')
        prediction = np.asarray([[-3.0, 2.0]], dtype = 'f')
        target = np.asarray([[-3.0, -1.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target), 1.25)
        np.testing.assert_allclose(loss.backward(), [[0.0, 0.5]])

    def test_huber_trajectory_sum_normalizes_each_route_not_whole_batch(self):
        loss = Huber(reduction = 'trajectory_sum')
        prediction = np.asarray([[2.0, 2.0, 2.0]], dtype = 'f')
        target = np.asarray([[0.0, 0.0, 0.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target, [0, 1, 3]), 3.0)
        np.testing.assert_allclose(loss.backward(), [[1.0, 0.5, 0.5]])

    def test_steps_to_goal_value_selection_balances_remaining_step_bands(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_states_per_band = 2
        item = SimpleNamespace(steps_to_goal = tuple(range(40,-1,-1)))

        columns = ai._steps_to_goal_value_columns_for_route(item,10,51)
        selected_steps = [item.steps_to_goal[column - 10] for column in columns]

        self.assertLessEqual(sum(step <= 10 for step in selected_steps), 2)
        self.assertLessEqual(sum(11 <= step <= 30 for step in selected_steps), 2)
        self.assertLessEqual(sum(step >= 31 for step in selected_steps), 2)
        self.assertEqual(len(columns), 6)

    def test_connected_search2_route_has_distances_to_the_final_goal(self):
        shared_data = []
        source_ai = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'myloss',
            datas = shared_data,
            indices = [],
        )
        regression_one = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'steps_to_goal',
            datas = shared_data,
            indices = [],
        )
        regression_two = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'steps_to_goal',
            datas = shared_data,
            indices = [],
        )
        cube = SimpleNamespace(
            simplify = lambda moves: tuple(moves),
            make_transformations = lambda scramble, moves: ((tuple(scramble),), (tuple(moves),)),
        )
        state = SimpleNamespace(
            s = ('scramble',),
            move_lis = [('A', 'B'), ('C', 'D', 'E')],
            search_TF = True,
            last_perfect_key = '',
            last_top_group = None,
        )
        frame = SimpleNamespace(
            solve_state = state,
            cube = cube,
            AIs = [source_ai, regression_one, regression_two],
            AI_idx = 0,
            AInum = 3,
        )

        stored = SolveSessionManager(frame)._store_connected_steps_to_goal_training_sample()

        self.assertTrue(stored)
        self.assertEqual(len(shared_data), 1)
        sample = shared_data[0]
        self.assertEqual(sample.moves, ('A', 'B', 'C', 'D', 'E'))
        self.assertEqual(sample.steps_to_goal, (5, 4, 3, 2, 1, 0))
        self.assertEqual(sample.source_search2_value_loss_type, 'steps_to_goal_regression')
        self.assertEqual(sample.trajectory_origin, 'direct_search')
        self.assertEqual(source_ai.indices, [])
        self.assertEqual(regression_one.indices, [0])
        self.assertEqual(regression_two.indices, [0])

    def test_direct_steps_to_goal_route_creates_one_connected_sample_per_target_kind(self):
        source = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'steps_to_goal')
        original = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'myloss')
        search3 = SimpleNamespace(search_mode = 'search3', search2_value_loss_type = 'myloss')
        state = SimpleNamespace(search_TF = True, fallback_used = False, move_lis = [('A',), ('B',)])
        calls = []
        frame = SimpleNamespace(
            solve_state = state,
            AIs = [source, original, search3],
            AI_idx = 0,
            AInum = 3,
            search_data_manager = SimpleNamespace(
                store_connected_search3_data = lambda ai: calls.append(('search3_connected', ai)),
            ),
        )
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda: calls.append('steps_connected') or True
        manager._store_connected_original_search2_training_sample = lambda ai: calls.append(('original_connected', ai)) or True
        manager._store_search2_segment_training_samples = lambda: calls.append('segments') or True

        self.assertTrue(manager._store_completed_training_samples())
        self.assertEqual(
            calls,
            ['steps_connected', ('original_connected', source), ('search3_connected', source)],
        )

    def test_fallback_steps_to_goal_route_stays_in_steps_to_goal_only(self):
        source = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'steps_to_goal')
        state = SimpleNamespace(search_TF = False, fallback_used = True, move_lis = [('A',)])
        calls = []
        frame = SimpleNamespace(
            solve_state = state,
            AIs = [source],
            AI_idx = 0,
            AInum = 1,
            search_data_manager = SimpleNamespace(
                store_connected_search3_data = lambda ai: calls.append(('search3_connected', ai)),
            ),
        )
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda: calls.append('steps_connected') or True
        manager._store_connected_original_search2_training_sample = lambda ai: calls.append(('original_connected', ai)) or True

        self.assertFalse(manager._store_completed_training_samples())
        self.assertEqual(calls, ['steps_connected'])

    def test_direct_search3_route_also_creates_a_connected_steps_to_goal_sample(self):
        source = SimpleNamespace(search_mode = 'search3', search2_value_loss_type = 'myloss')
        state = SimpleNamespace(search_TF = True, fallback_used = False, move_lis = [('A',), ('B',)])
        calls = []
        frame = SimpleNamespace(solve_state = state, AIs = [source], AI_idx = 0, AInum = 1)
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda: calls.append('steps_connected') or True
        manager._store_connected_myloss_training_sample = lambda: calls.append('pairwise_connected') or True
        manager._store_search2_segment_training_samples = lambda: calls.append('original_segments') or True

        self.assertFalse(manager._store_completed_training_samples())
        self.assertEqual(calls, ['steps_connected', 'pairwise_connected', 'original_segments'])

    def test_connected_steps_route_adds_one_search3_sample_per_search3_ai(self):
        source = SimpleNamespace(search_mode = 'search2')
        original = SimpleNamespace(search_mode = 'search2', datas_search3 = [], indices_search3 = [])
        search3_one = SimpleNamespace(search_mode = 'search3', datas_search3 = [], indices_search3 = [])
        search3_two = SimpleNamespace(search_mode = 'transformer', datas_search3 = [], indices_search3 = [])
        frame = SimpleNamespace(
            solve_state = SimpleNamespace(s = ('scramble',), move_lis = [('A',), ('B',)]),
            cube = SimpleNamespace(simplify = lambda moves: tuple(moves)),
            AIs = [original, search3_one, search3_two],
        )
        manager = SearchDataManager(frame)
        built = []
        manager.build_connected_search3_training_sample = lambda scramble, moves, ai: built.append(
            (scramble, moves, ai)
        ) or SimpleNamespace()

        self.assertEqual(manager.store_connected_search3_data(source), 2)
        self.assertEqual(built, [(('scramble',), ('A', 'B'), source)] * 2)
        self.assertEqual(original.datas_search3, [])
        self.assertEqual(search3_one.indices_search3, [0])
        self.assertEqual(search3_two.indices_search3, [0])


if __name__ == '__main__':
    unittest.main()

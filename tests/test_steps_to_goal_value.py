from types import SimpleNamespace
import unittest

import numpy as np

from ai.losses import Huber
from ai.rubiks_ai import Rubiks_3_AI
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


if __name__ == '__main__':
    unittest.main()

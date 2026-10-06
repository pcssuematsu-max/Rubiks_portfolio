from types import SimpleNamespace
import unittest

import numpy as np

from ai.losses import Huber
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
        self.assertEqual(source_ai.indices, [])
        self.assertEqual(regression_one.indices, [0])
        self.assertEqual(regression_two.indices, [0])


if __name__ == '__main__':
    unittest.main()

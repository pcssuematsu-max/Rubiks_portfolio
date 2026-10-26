import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI


class Search3BranchTraceTests(unittest.TestCase):
    def setUp(self):
        self.ai = Rubiks_3_AI([8], cube_size=3, search_mode='search3')
        self.ai.search_batch3 = 1
        self.ai.search_depth3 = 3
        self.ai.search3_widening_visit_growth = 0.0
        self.ai.cube.reset()
        self.ai.cube.scramble(0, Move=(' U ', ' R '))
        self.engine = self.ai.search3_engine

    def _root_with_priors(self, preferred, target):
        self.engine.reset_tree()
        self.engine._ensure_root_node(C=0.5)
        root = self.engine.root_node
        root.P[:] = 0.0
        root.P[self.ai.cube.key_to_num[preferred]] = 0.9
        root.P[self.ai.cube.key_to_num[target]] = 0.1
        return root

    def test_trace_distinguishes_width_block_from_first_visit(self):
        target = ' D '
        self._root_with_priors(' F ', target)
        self.ai.search3_widening_initial_actions = 1
        blocked = self.engine.search3(5, C=0.5, trace_root_move=target).root_action_trace
        self.assertIsNone(blocked['firstEligiblePlayout'])
        self.assertIsNone(blocked['firstVisitPlayout'])
        self.assertEqual(blocked['finalVisits'], 0)

        self._root_with_priors(target, ' F ')
        self.ai.search3_widening_initial_actions = 0
        visited = self.engine.search3(5, C=0.5, trace_root_move=target).root_action_trace
        self.assertEqual(visited['firstEligiblePlayout'], 1)
        self.assertEqual(visited['firstVisitPlayout'], 1)
        self.assertIsNotNone(visited['firstChildValue'])
        self.assertGreater(visited['finalVisits'], 0)
        self.assertEqual(visited['visitSamples'][0]['playouts'], 1)

    def test_normal_search_does_not_attach_trace(self):
        self._root_with_priors(' F ', ' D ')
        self.ai.search3_widening_initial_actions = 0
        result = self.engine.search3(2, C=0.5)
        self.assertFalse(hasattr(result, 'root_action_trace'))

    def test_eligible_move_can_remain_unvisited(self):
        target = ' D '
        self._root_with_priors(' F ', target)
        self.ai.search3_widening_initial_actions = 0
        trace = self.engine.search3(1, C=0.5, trace_root_move=target).root_action_trace
        self.assertEqual(trace['firstEligiblePlayout'], 1)
        self.assertIsNone(trace['firstVisitPlayout'])
        self.assertIsNone(trace['firstChildValue'])

    def test_trace_does_not_change_search_result(self):
        self.ai.search3_widening_initial_actions = 0
        self._root_with_priors(' F ', ' D ')
        plain = self.engine.search3(12, C=0.5)
        self._root_with_priors(' F ', ' D ')
        traced = self.engine.search3(12, C=0.5, trace_root_move=' D ')
        self.assertEqual(plain.end_reason, traced.end_reason)
        self.assertEqual(plain.moves, traced.moves)
        np.testing.assert_array_equal(plain.policy_target, traced.policy_target)


if __name__ == '__main__':
    unittest.main()

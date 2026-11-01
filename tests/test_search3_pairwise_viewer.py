import unittest
from types import SimpleNamespace

from managers.solve_session import SolveSessionManager, SolveSessionState
from model.search_result import SearchResult
from ui.viewers import MoveViewer, color_for_search2_value, color_for_search3_value


class Search3PairwiseViewerTests(unittest.TestCase):
    def test_pairwise_search3_colors_use_search2_goal_relative_scale(self):
        manager = SolveSessionManager.__new__(SolveSessionManager)
        ai = SimpleNamespace(search3_value_loss_type='pairwise',perfect_val=10.0)
        manager.frame = SimpleNamespace(AIs=[ai],AI_idx=0)
        result = SimpleNamespace(search_mode='search3',value_trace=[5.0,7.0,10.0])
        relative_values = manager._display_value_deltas(result)
        self.assertEqual(relative_values,[-5.0,-3.0,0.0])
        viewer = SimpleNamespace(text_color='#FFFFFF')
        self.assertEqual(
            MoveViewer._move_square_color(viewer,relative_values,0,'search3',None,'pairwise'),
            color_for_search2_value(-3.0),
        )
        self.assertEqual(
            MoveViewer._move_square_color(viewer,relative_values,1,'search3',None,'pairwise'),
            color_for_search2_value(0.0),
        )

    def test_absolute_search3_retains_probability_colors(self):
        viewer = SimpleNamespace(text_color='#FFFFFF')
        self.assertEqual(
            MoveViewer._move_square_color(viewer,[0.2,0.8],0,'search3',None,'absolute'),
            color_for_search3_value(0.8),
        )

    def test_progressive_display_keeps_latest_move_above_current_attempts(self):
        state = SolveSessionState()
        state.move_lis = [(), ('R',)]
        state.val_lis2 = [[], [0.2, 0.4]]
        state.display_move_lis = [(), ('old attempt',), ('older attempt',)]
        state.display_key_lis = ['', 'S3-1', 'S3-2']
        state.display_root_lis = [0.2, 0.2, 0.2]
        state.display_val_lis = [0.2, 0.3, 0.4]
        state.display_val_lis2 = [[], [], []]
        ai = SimpleNamespace(search_mode='search3', search3_budget_mode='progressive')
        cube = SimpleNamespace(reduce=lambda moves: (tuple(moves), tuple(range(len(moves)))))
        manager = SolveSessionManager(SimpleNamespace(
            solve_state=state, AIs=[ai], AI_idx=0, cube=cube,
        ))
        previous = SearchResult(False, ('R',), 0.2, [0.2, 0.4], 0.4, [1, 100], search_mode='search3')

        self.assertTrue(manager._uses_progressive_search3(ai))
        self.assertFalse(manager._uses_progressive_search3(SimpleNamespace(
            search_mode='search3', search3_budget_mode='fixed',
        )))
        manager._show_latest_progressive_move(previous)
        self.assertEqual(state.display_move_lis, [(), ('R',)])
        self.assertEqual(state.display_key_lis, ['', 'この手を動かしました'])
        self.assertEqual(state.display_val_lis2[-1], [0.2, 0.4])

        attempt = SearchResult(
            False, ('U',), 0.4, [0.4, 0.6], 0.6, [2, 100],
            search_mode='search3', attempt_index=1,
        )
        manager._record_search_display(ai, attempt)
        self.assertEqual(state.display_key_lis, ['', 'この手を動かしました', 'S3-1:2/100'])
        self.assertEqual(state.display_move_lis[-1], ('U',))

        state.move_lis.append(('U',))
        state.val_lis2.append([0.4, 0.6])
        manager._show_latest_progressive_move(attempt)
        self.assertEqual(state.display_move_lis, [(), ('U',)])
        self.assertEqual(state.display_key_lis, ['', 'この手を動かしました'])


if __name__ == '__main__':
    unittest.main()

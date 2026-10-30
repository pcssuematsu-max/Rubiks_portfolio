import unittest
from types import SimpleNamespace

from managers.solve_session import SolveSessionManager
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


if __name__ == '__main__':
    unittest.main()

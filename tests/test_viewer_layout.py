import unittest

import numpy as np

from managers.debug_analysis import DebugAnalysisManager, VIEWER_RANGE_TEXT_WIDTH
from ui.viewers import MoveViewer, format_activity_status


class _LayoutProbe:
    _header_row_index = MoveViewer._header_row_index
    _row_height = MoveViewer._row_height
    _configure_layout = MoveViewer._configure_layout

    def __init__(self, move_labels):
        self.fixed_r_size = 400
        self.r_size = self.fixed_r_size
        self.c_size = 700
        self.c_dist = MoveViewer._move_column_width(move_labels)
        self.r_dist = 13
        self.c_start_cube_state = 100
        self.value_width = 95
        self.key_width = 180
        self.min_move_columns = 4
        self.last_configure = None

    def configure(self, **kwargs):
        self.last_configure = kwargs


class MoveViewerLayoutTests(unittest.TestCase):
    def test_move_width_uses_the_puzzles_longest_notation(self):
        labels = ("URF", "URF'", "mURF'")

        self.assertEqual(MoveViewer._move_column_width(labels), 45)

    def test_solve_updates_do_not_change_move_or_key_width(self):
        viewer = _LayoutProbe(("URF", "URF'", "mURF'"))
        initial_move_width = viewer.c_dist
        initial_key_width = viewer.key_width

        viewer._configure_layout(("A", "B"), ((), ("R",)))
        short_layout = (viewer.c_dist, viewer.key_width, viewer.c_start, viewer.words_in_a_row)
        viewer._configure_layout(
            ("A", "B"),
            ((), ("mURF'", "mURF'", "mURF'")),
        )
        long_layout = (viewer.c_dist, viewer.key_width, viewer.c_start, viewer.words_in_a_row)

        self.assertEqual(initial_move_width, 45)
        self.assertEqual(initial_key_width, 180)
        self.assertEqual(short_layout, long_layout)


class ActivityStatusLayoutTests(unittest.TestCase):
    def test_status_text_is_bounded_to_the_log_viewer_width(self):
        status = format_activity_status('探索中: transformer attempt 999 | AI 12 | Solve 3456', 24)

        self.assertEqual(len(status), 24)
        self.assertTrue(status.endswith('…'))

    def test_short_status_is_not_truncated(self):
        self.assertEqual(format_activity_status('待機中', 24), '状況: 待機中')


class DebugViewerRangeTextTests(unittest.TestCase):
    def setUp(self):
        self.manager = DebugAnalysisManager.__new__(DebugAnalysisManager)

    def test_range_text_uses_three_bounded_lines(self):
        vector = np.asarray((-9.876e20, -1.234e20, 1.234e20, 9.876e20))

        text = self.manager._viewer_range_text(vector, 2, positive=True)
        lines = text.splitlines()

        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith('High value'))
        self.assertTrue(all(len(line) <= VIEWER_RANGE_TEXT_WIDTH for line in lines))

    def test_empty_range_keeps_the_same_three_line_shape(self):
        text = self.manager._viewer_range_text((), 0, positive=False)

        self.assertEqual(text.splitlines(), ['Low value  n=0', 'min=-', 'max=-'])


if __name__ == "__main__":
    unittest.main()

import unittest

from ui.viewers import MoveViewer


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


if __name__ == "__main__":
    unittest.main()

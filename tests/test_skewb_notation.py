"""WCA Skewb turns coexist with o-prefixed opposite-vertex turns."""

import unittest

import numpy as np

from skewb.cube import SkewbCube, MOVE_TO_CORNER, OPPOSITE_CORNER_MOVES


class SkewbNotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cube = SkewbCube()

    def test_sixteen_canonical_moves_cover_all_eight_vertices(self):
        self.assertEqual(len(self.cube.move_keys), 16)
        self.assertEqual(set(self.cube.move_keys), {
            base + suffix for base in ("R", "U", "L", "B", "oR", "oU", "oL", "oB")
            for suffix in ("", "'")
        })
        self.assertEqual(OPPOSITE_CORNER_MOVES, {
            "oR": "UFL", "oU": "DFR", "oL": "UBR", "oB": "URF",
        })

    def test_old_three_face_names_remain_accepted(self):
        for move, corner in MOVE_TO_CORNER.items():
            for suffix in ("", "'"):
                self.assertEqual(self.cube.normalize_move_key(corner + suffix), move + suffix)
                np.testing.assert_array_equal(self.cube.move[corner + suffix], self.cube.move[move + suffix])
                self.cube.reset()
                self.cube.make_move(corner + suffix)
                legacy_state = self.cube.state.copy()
                self.cube.reset()
                self.cube.make_move(move + suffix)
                np.testing.assert_array_equal(self.cube.state, legacy_state)

    def test_display_and_transforms_use_canonical_names(self):
        self.assertEqual(self.cube.format_moves(("DRB", "ULB'", "URF")), ("R", "U'", "oB"))
        self.assertFalse(np.array_equal(self.cube.move["oR"], self.cube.move["R'"]))
        for index in range(len(self.cube.transformation_keys)):
            for move in self.cube.move_keys:
                self.assertIn(self.cube.transform((move,), index)[0], self.cube.move_keys)

    def test_registered_myperms_use_canonical_moves_with_unchanged_effects(self):
        cube = SkewbCube()
        cube._register_myperms2()
        self.assertEqual(
            cube.myperms2["Skewb-Sledgehammer-A"],
            ("oB'", "U", "oB", "U'"),
        )
        self.assertEqual(
            cube.myperms2["Skewb-Corner2Twist"],
            cube.normalize_move_sequence((
                "DRB'", "DBL", "DRB", "DFR", "DFL", "DBL", "DFR'", "DBL'", "DFR'", "DFL'",
            )),
        )
        for sequence in cube.myperms2.values():
            self.assertTrue(all(move in cube.move_keys for move in sequence))


if __name__ == "__main__":
    unittest.main()

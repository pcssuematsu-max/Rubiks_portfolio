import unittest

from core.myperm_keys import (
    MypermKey,
    format_myperm_key,
    make_myperm_key,
    myperm_base_key,
    myperm_transform_index,
    normalize_myperm_registry,
    single_move_myperm_name,
)
from cto.cube import CtoCube
from cube.rubiks_cube import Rubiks_3
from fto.cube import FtoCube
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube


class MypermKeyTest(unittest.TestCase):
    def test_key_separates_name_from_transform_index(self):
        key = make_myperm_key("Algorithm-", 7)

        self.assertEqual(key, MypermKey("Algorithm", 7))
        self.assertEqual(myperm_base_key(key), "Algorithm")
        self.assertEqual(myperm_transform_index(key), 7)
        self.assertEqual(format_myperm_key(key), "Algorithm#07")

    def test_numeric_algorithm_name_is_not_mistaken_for_transform_index(self):
        self.assertEqual(myperm_base_key("SuperTwist2"), "SuperTwist2")
        self.assertIsNone(myperm_transform_index("SuperTwist2"))
        self.assertEqual(myperm_base_key("SuperTwist2#12"), "SuperTwist2")
        self.assertEqual(myperm_transform_index("SuperTwist2#12"), 12)

    def test_single_move_name_removes_notation_padding(self):
        self.assertEqual(single_move_myperm_name(" U'"), "SingleMove-U'")

    def test_registry_rejects_names_that_normalize_to_same_key(self):
        with self.assertRaises(ValueError):
            normalize_myperm_registry({"Case-A": ("R",), "Case-A-": ("U",)})


class PuzzleMypermNamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles = {
            "rubiks": Rubiks_3(size=5),
            "megaminx": MegaminxCube(),
            "pyraminx": PyraminxCube(),
            "master_pyraminx": MasterPyraminxCube(),
            "skewb": SkewbCube(),
            "fto": FtoCube(),
            "cto": CtoCube(),
        }

    def test_every_expanded_key_uses_canonical_structure_and_name(self):
        for puzzle_name, puzzle in self.puzzles.items():
            with self.subTest(puzzle=puzzle_name):
                self.assertTrue(puzzle.myperms)
                self.assertTrue(all(isinstance(key, MypermKey) for key in puzzle.myperms))
                self.assertFalse(any(key.name.endswith("-") for key in puzzle.myperms))
                self.assertFalse(any(char.isspace() for key in puzzle.myperms for char in key.name))
                self.assertFalse(any(name.endswith("-") for name in puzzle.myperms2))

    def test_previously_overwritten_algorithms_are_distinct(self):
        rubiks_registry = self.puzzles["rubiks"].myperms2
        for effect_name in ("C8", "C8s", "C8[2x4]"):
            with self.subTest(effect=effect_name):
                first_name = f"{effect_name}~v01"
                second_name = f"{effect_name}~v02"
                self.assertIn(first_name, rubiks_registry)
                self.assertIn(second_name, rubiks_registry)
                self.assertNotEqual(rubiks_registry[first_name], rubiks_registry[second_name])

        megaminx_registry = self.puzzles["megaminx"].myperms2
        corner4_edge3_algorithms = [
            moves
            for name, moves in megaminx_registry.items()
            if name.startswith("C4[") and "+E3[" in name
        ]
        self.assertGreaterEqual(len(corner4_edge3_algorithms), 2)
        self.assertGreaterEqual(len(set(corner4_edge3_algorithms)), 2)


if __name__ == "__main__":
    unittest.main()

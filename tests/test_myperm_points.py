import unittest
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cto.cube import CtoCube
from cube import myperm_registry
from core import myperm_points
from core.myperm_points import (
    MypermPointCalculator,
    load_myperm_points,
    parse_myperm_points_text,
    point_representative_transform,
    reindex_myperms_by_points,
)
from core.myperm_effects import MypermEffectAnalyzer
from core.myperm_keys import make_myperm_key, resolve_myperm_key
from cube.rubiks_cube import Rubiks_3
from fto.cube import FtoCube
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube


class MypermPointTableTest(unittest.TestCase):
    def test_default_rubiks_registry_uses_highest_point_source(self):
        for size in (3, 7):
            with self.subTest(size = size):
                cube = Rubiks_3(size = size)
                self.assertEqual(set(cube.myperm_transform_points), set(cube.myperms2))
                for name, rows in cube.myperm_transform_points.items():
                    self.assertEqual(rows[0]["point"], max(row["point"] for row in rows))
                    self.assertEqual(
                        tuple(cube.myperms2[name]),
                        tuple(cube.myperms[make_myperm_key(name, 0)]),
                    )

    def test_point_order_cache_recalculates_only_when_points_change(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            points_path = root / "Points.txt"
            points_path.write_text(Path("Points.txt").read_text(encoding = "utf-8"), encoding = "utf-8")
            cache_directory = root / "cache"
            with patch.object(myperm_registry, "_PROJECT_ROOT", root), patch.object(
                myperm_registry, "_POINT_CACHE_DIRECTORY", cache_directory,
            ):
                Rubiks_3(size = 3)
                self.assertTrue((cache_directory / "cube-3.json").exists())

                with patch.object(myperm_points, "reindex_myperms_by_points", side_effect = AssertionError("cache miss")):
                    Rubiks_3(size = 3)

                original = points_path.read_text(encoding = "utf-8")
                changed = re.sub(r"(?m)^UFR:\S+", "UFR:12345", original, count = 1)
                self.assertNotEqual(changed, original)
                points_path.write_text(changed, encoding = "utf-8")
                with patch.object(myperm_points, "reindex_myperms_by_points", wraps = reindex_myperms_by_points) as score:
                    Rubiks_3(size = 3)
                self.assertEqual(score.call_count, 1)

    def test_parser_normalizes_corner_midedge_and_wing_positions(self):
        table = parse_myperm_points_text(
            """
            #Corners
            UFR:2 DBL:0.5
            #MidEdge
            UF:20 DB:10 FR:50
            #OuterEdge
            UL:7 UR@B:9
            #Wing
            FR@U:500 DB@L:100 BL@D:320
            #XCenter
            R@2F.2U:2000
            Others:0
            """
        )

        self.assertEqual(table.point_for_part("C", "URF"), 2)
        self.assertEqual(table.point_for_part("C", "LDB"), 0.5)
        self.assertEqual(table.point_for_part("ME", "UF@M"), 20)
        self.assertEqual(table.point_for_part("ME", "RF@M"), 50)
        self.assertEqual(table.point_for_part("OE", "UL"), 7)
        self.assertEqual(table.point_for_part("OE", "LU"), 7)
        self.assertEqual(table.point_for_part("OE", "UR@B"), 9)
        self.assertEqual(table.point_for_part("OE", "RU@B"), 9)
        self.assertEqual(table.point_for_part("OE", "UR@L"), 0)
        self.assertEqual(table.point_for_part("W2", "FR@2U"), 500)
        self.assertEqual(table.point_for_part("W2", "RF@2U"), 500)
        self.assertEqual(table.point_for_part("W2", "LB@2D"), 320)
        self.assertEqual(table.point_for_part("CtrX", "R@2F.2U"), 2000)
        self.assertEqual(table.point_for_part("CtrX", "R@2U.2F"), 2000)
        self.assertEqual(table.point_for_part("CtrX", "R@2F.2D"), 0)

    def test_current_points_file_scores_a_real_wing_myperm(self):
        cube = Rubiks_3(size = 7)
        table = load_myperm_points()
        calculator = MypermPointCalculator(cube, table)
        key = resolve_myperm_key(cube, "W2-6p[3x2][BR@D>FL@D>RF@U;BR@U>FL@U>RF@D]")

        self.assertEqual(calculator.point_for_key(key), 2610)
        self.assertIn("BR@D>FL@D>RF@U", key[0])

    def test_current_points_file_applies_wing_layers_separately(self):
        table = load_myperm_points()

        self.assertEqual(table.point_for_part("W2", "FR@2U"), 550)
        self.assertEqual(table.point_for_part("W3", "FR@3U"), 0)
        self.assertEqual(table.point_for_part("W2", "RF@2U"), 550)

    def test_layered_wing_points_leave_unscoped_format_compatible(self):
        layered = parse_myperm_points_text("#Wing\nLayer2:\nFR@U:500\nLayer3: 0\n")
        unscoped = parse_myperm_points_text("#Wing\nFR@U:500\n")

        self.assertEqual(layered.point_for_part("W2", "FR@2U"), 500)
        self.assertEqual(layered.point_for_part("W3", "FR@3U"), 0)
        self.assertEqual(unscoped.point_for_part("W3", "FR@3U"), 500)

    def test_edge_bundle_scores_mid_edge_and_each_wing_layer(self):
        cube = Rubiks_3(size = 7)
        table = load_myperm_points()

        self.assertEqual(table.edge_bundle_point(cube, "UF@M"), 420)

    def test_parser_can_load_megaminx_points_separately(self):
        text = """
        #Corners
        UFR:2
        ###megaminx
        #Corners
        (U,F,R):40
        #Edges
        (U,bR):1100
        """
        rubik_table = parse_myperm_points_text(text)
        megaminx_table = parse_myperm_points_text(text, puzzle = "megaminx")

        self.assertEqual(rubik_table.point_for_part("C", "UFR"), 2)
        self.assertEqual(rubik_table.point_for_part("C", "URF"), 2)
        self.assertEqual(megaminx_table.point_for_part("C", "UFR"), 40)
        self.assertEqual(megaminx_table.point_for_part("E", "U.bR"), 1100)
        self.assertEqual(megaminx_table.point_for_part("E", "bR.U"), 1100)

    def test_megaminx_source_names_are_point_representative_effect_names(self):
        cube = MegaminxCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "megaminx"))

        for name in cube.myperms2:
            with self.subTest(name = name):
                key = make_myperm_key(name, 0)
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(key))
                current_point = calculator.point_for_key(key)
                best_point = max(
                    calculator.point_for_key(make_myperm_key(name, transform_index))
                    for transform_index in range(len(cube.transformation_keys))
                )
                self.assertEqual(current_point, best_point)

    def test_pyraminx_source_names_are_point_representative_effect_names(self):
        cube = PyraminxCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "pyraminx"))

        self.assertEqual(cube.myperm_point_puzzle, "pyraminx")
        self.assertEqual(cube.myperm_name_aliases, {})
        for name in cube.myperms2:
            with self.subTest(name = name):
                key = make_myperm_key(name, 0)
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(key))
                current_point = calculator.point_for_key(key)
                best_point = max(
                    calculator.point_for_key(make_myperm_key(name, transform_index))
                    for transform_index in range(len(cube.transformation_keys))
                )
                self.assertEqual(current_point, best_point)

    def test_master_pyraminx_source_names_are_point_representative_effect_names(self):
        cube = MasterPyraminxCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "masterpyraminx"))

        self.assertEqual(cube.myperm_point_puzzle, "masterpyraminx")
        self.assertEqual(cube.myperm_name_aliases, {})
        for name in cube.myperms2:
            with self.subTest(name = name):
                key = make_myperm_key(name, 0)
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(key))
                current_point = calculator.point_for_key(key)
                best_point = max(
                    calculator.point_for_key(make_myperm_key(name, transform_index))
                    for transform_index in range(len(cube.transformation_keys))
                )
                self.assertEqual(current_point, best_point)

    def test_cto_source_names_are_point_representative_effect_names(self):
        cube = CtoCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "cto"))

        self.assertEqual(cube.myperm_point_puzzle, "cto")
        for name in cube.myperms2:
            with self.subTest(name = name):
                key = make_myperm_key(name, 0)
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(key))
                current_point = calculator.point_for_key(key)
                best_point = max(
                    calculator.point_for_key(make_myperm_key(name, transform_index))
                    for transform_index in range(len(cube.transformation_keys))
                )
                self.assertEqual(current_point, best_point)

    def test_fto_source_names_are_point_representative_effect_names(self):
        cube = FtoCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "fto"))

        self.assertEqual(cube.myperm_point_puzzle, "fto")
        for name in cube.myperms2:
            with self.subTest(name = name):
                key = make_myperm_key(name, 0)
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(key))
                current_point = calculator.point_for_key(key)
                best_point = max(
                    calculator.point_for_key(make_myperm_key(name, transform_index))
                    for transform_index in range(len(cube.transformation_keys))
                )
                self.assertEqual(current_point, best_point)

    def test_skewb_source_names_are_point_representative_effect_names(self):
        cube = SkewbCube()
        analyzer = MypermEffectAnalyzer(cube)
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "skewb"))

        for name in cube.myperms2:
            with self.subTest(name = name):
                self.assertEqual(name.split("~v", 1)[0], analyzer.proposed_name(make_myperm_key(name, 0)))
                self.assertEqual(
                    calculator.point_for_key(make_myperm_key(name, 0)),
                    max(
                        calculator.point_for_key(make_myperm_key(name, index))
                        for index in range(len(cube.transformation_keys))
                    ),
                )

    def test_megaminx_lp_transform_uses_megaminx_points(self):
        cube = MegaminxCube()
        moves = ("L2'", "U'", "R", "U", "L2", "U'", "R'", "U")
        calculator = MypermPointCalculator(cube, load_myperm_points(puzzle = "megaminx"))

        row = point_representative_transform(cube, moves)
        representative_name = MypermEffectAnalyzer(cube).analyze(row.moves).concise_name()

        self.assertEqual(row.transform_index, 56)
        self.assertEqual(row.point, 85)
        self.assertEqual(representative_name, "C3[U.bR.R>L.sL.bL>LUF]")
        self.assertEqual(
            calculator.point_for_moves(row.moves),
            max(
                calculator.point_for_moves(cube.transform(moves, transform_index))
                for transform_index in range(len(cube.transformation_keys))
            ),
        )


if __name__ == "__main__":
    unittest.main()

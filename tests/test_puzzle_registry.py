import unittest

from core.puzzle_registry import PUZZLE_REGISTRY, get_puzzle_adapter
from cto.cube import CtoCube
from fto.cube import FtoCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube
from ui.frame_config import FrameConfig


class PuzzleRegistryTest(unittest.TestCase):
    def test_registered_puzzles_are_resolved_by_key_and_alias(self):
        self.assertIs(get_puzzle_adapter('fto'), get_puzzle_adapter('face_turning_octahedron'))
        self.assertIs(get_puzzle_adapter('cto'), get_puzzle_adapter('corner_turning_octahedron'))
        self.assertIs(get_puzzle_adapter('master_pyraminx'), get_puzzle_adapter('master-pyraminx'))
        self.assertEqual(
            {adapter.key for adapter in PUZZLE_REGISTRY.adapters()},
            {'fto', 'cto', 'pyraminx', 'master_pyraminx', 'skewb'},
        )

    def test_registered_adapters_create_the_expected_models(self):
        config = FrameConfig(puzzle_type='fto', cube_size=3)
        self.assertIsInstance(get_puzzle_adapter('fto').create_cube(config), FtoCube)
        self.assertIsInstance(get_puzzle_adapter('cto').create_cube(config), CtoCube)
        self.assertIsInstance(get_puzzle_adapter('pyraminx').create_cube(config), PyraminxCube)
        self.assertIsInstance(get_puzzle_adapter('master_pyraminx').create_cube(config), MasterPyraminxCube)
        self.assertIsInstance(get_puzzle_adapter('skewb').create_cube(config), SkewbCube)

    def test_registered_adapters_share_notation_and_effect_hooks(self):
        adapter = get_puzzle_adapter('fto')
        cube = adapter.create_cube(FrameConfig(puzzle_type='fto', cube_size=3))
        moves = ("URF", "UFL", "URF'", "UFL'")

        self.assertEqual(adapter.format_moves(cube, moves), moves)
        self.assertNotEqual(adapter.analyze_effect(cube, moves).concise_name(), 'Identity')
        self.assertEqual(
            adapter.default_priority_groups,
            ('Corner', 'Edge', 'CenterA', 'CenterB'),
        )
        self.assertEqual(
            get_puzzle_adapter('master_pyraminx').default_priority_groups,
            ('Corner', 'Edge', 'MidEdge', 'Center'),
        )


if __name__ == '__main__':
    unittest.main()

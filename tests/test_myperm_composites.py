import hashlib
import unittest
from types import SimpleNamespace

from ai.myperm_composites import (
    EXISTING_COMPOSITE_STATE_HASH, make_4x4_myperm_fixtures,
    make_4x4_myperm_training_scrambles,
)
from cube.rubiks_cube import Rubiks_3


class MypermCompositeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cube = Rubiks_3(size=4, PointReindex=False)

    def test_training_and_fixed_states_are_distinct(self):
        cube = self.cube
        original = cube.state.copy()
        fixtures = make_4x4_myperm_fixtures(cube)
        held_out = {row['initialStateHash'] for row in fixtures}
        self.assertEqual(len(fixtures), 5)
        self.assertEqual(len(held_out), 5)
        self.assertNotIn(EXISTING_COMPOSITE_STATE_HASH, held_out)
        self.assertTrue(all(row['setupEffect']['movedPieceCount'] <= 8
                            for row in fixtures[:3]))
        self.assertTrue(all(row['setupEffect']['movedPieceCount'] >= 20
                            for row in fixtures[3:]))
        training = make_4x4_myperm_training_scrambles(cube, held_out)
        self.assertGreaterEqual(len(training), 20)
        training_hashes = set()
        for moves in training:
            cube.reset()
            cube.scramble(0, Move=moves)
            state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
            self.assertNotIn(state_hash, held_out | {EXISTING_COMPOSITE_STATE_HASH})
            training_hashes.add(state_hash)
        self.assertEqual(len(training), len(training_hashes))
        cube.state[:] = original

    def test_reindexed_app_cube_resolves_source_myperm_names(self):
        # Frame's live cube uses PointReindex=True, which renames source keys.
        runtime_cube = Rubiks_3(size=4)
        expected = make_4x4_myperm_fixtures(self.cube)
        actual = make_4x4_myperm_fixtures(runtime_cube)
        self.assertEqual(actual, expected)
        held_out = {row['initialStateHash'] for row in expected}
        self.assertEqual(
            make_4x4_myperm_training_scrambles(runtime_cube, held_out),
            make_4x4_myperm_training_scrambles(self.cube, held_out),
        )

    def test_experiment_bootstrap_accepts_live_cube(self):
        from ui.frame import Frame
        runtime_cube = Rubiks_3(size=4)
        ai = SimpleNamespace(datas=[], indices=[])
        frame = SimpleNamespace(cube_size=4, cube=runtime_cube, AIs=[ai], AInum=1)
        Frame._append_bootstrap_datas(
            frame, bootstrap_datas=(), supplemental_myperm_composites=True,
        )
        self.assertGreaterEqual(len(frame._supplemental_myperm_scrambles), 20)
        self.assertEqual(len(ai.datas), len(frame._supplemental_myperm_scrambles))
        self.assertEqual(ai.indices, list(range(len(ai.datas))))


if __name__ == '__main__':
    unittest.main()

import unittest
from types import SimpleNamespace

import numpy as np

from managers.solve_session import SolveSessionManager


class LevelPromotionTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(
            AInum = 3,
            stage_num = 1,
            AI_idx = 0,
            stage = 0,
            level = np.ones((3,1), dtype = 'i'),
            level_play_counts = np.zeros((3,1), dtype = 'i'),
            level_success_counts = np.zeros((3,1), dtype = 'i'),
        )
        self.manager = SolveSessionManager(self.frame)

    def _record_trials(self, ai_index, results):
        self.frame.AI_idx = ai_index
        for result in results:
            self.manager._record_level_trial(result)

    def test_only_ai_with_all_plays_successful_levels_up(self):
        self._record_trials(0, (True, True, True))
        self._record_trials(1, (True, False, True))
        self._record_trials(2, (True, True, True))

        promoted = self.manager._promote_perfect_ai_levels(0)

        self.assertEqual(promoted, [(0, 3), (2, 3)])
        self.assertEqual(self.frame.level[:,0].tolist(), [2, 1, 2])
        self.assertEqual(self.frame.level_play_counts[:,0].tolist(), [0, 0, 0])
        self.assertEqual(self.frame.level_success_counts[:,0].tolist(), [0, 0, 0])

    def test_ai_without_a_play_does_not_level_up(self):
        self._record_trials(0, (True,))

        promoted = self.manager._promote_perfect_ai_levels(0)

        self.assertEqual(promoted, [(0, 1)])
        self.assertEqual(self.frame.level[:,0].tolist(), [2, 1, 1])


if __name__ == '__main__':
    unittest.main()

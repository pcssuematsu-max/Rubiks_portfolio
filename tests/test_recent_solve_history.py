import unittest

from managers.solve_session import SolveSessionState


class RecentSolveHistoryTests(unittest.TestCase):
    def test_keeps_the_ten_newest_completed_solves(self):
        state = SolveSessionState()

        for solve_index in range(12):
            state.add_recent_solve(
                solve_index,
                0,
                solve_index % 2 == 0,
                ("R",),
                ("R'",),
            )

        self.assertEqual(len(state.recent_solve_history), 10)
        self.assertEqual(state.recent_solve_history[0].solve_index, 2)
        self.assertEqual(state.recent_solve_history[-1].solve_index, 11)
        self.assertTrue(state.recent_solve_history[0].succeeded)
        self.assertFalse(state.recent_solve_history[-1].succeeded)

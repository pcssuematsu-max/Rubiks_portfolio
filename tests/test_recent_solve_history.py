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
                timestamp = "2026-09-10T00:00:00+00:00",
                puzzle_type = "rubiks-7x7",
                search_mode = "search3",
                elapsed_seconds = 1.25,
                score = 0.75,
                outcome = "greedy_fallback_success",
            )

        self.assertEqual(len(state.recent_solve_history), 10)
        self.assertEqual(state.recent_solve_history[0].solve_index, 2)
        self.assertEqual(state.recent_solve_history[-1].solve_index, 11)
        self.assertTrue(state.recent_solve_history[0].succeeded)
        self.assertFalse(state.recent_solve_history[-1].succeeded)
        self.assertEqual(state.recent_solve_history[-1].puzzle_type, "rubiks-7x7")
        self.assertEqual(state.recent_solve_history[-1].search_mode, "search3")
        self.assertEqual(state.recent_solve_history[-1].elapsed_seconds, 1.25)
        self.assertEqual(state.recent_solve_history[-1].score, 0.75)
        self.assertEqual(
            state.recent_solve_history[-1].outcome,
            "greedy_fallback_success",
        )

    def test_paused_time_is_not_counted_in_the_solve_timer(self):
        state = SolveSessionState()
        state.phase = 0
        state.start_solve_timer(now = 10.0)
        state.pause_solve_timer(now = 15.0)
        self.assertEqual(state.elapsed_solve_seconds(now = 25.0), 5.0)
        state.resume_solve_timer(now = 30.0)
        self.assertEqual(state.elapsed_solve_seconds(now = 33.0), 8.0)

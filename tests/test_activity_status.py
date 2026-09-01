import unittest

from managers.solve_session import SolveSessionManager


class _StatusFrame:
    AI_idx = 2
    N = 7

    def __init__(self):
        self.messages = []

    def set_activity_status(self, message):
        self.messages.append(message)


class ActivityStatusTests(unittest.TestCase):
    def test_solve_status_includes_phase_ai_and_solve_number(self):
        frame = _StatusFrame()

        SolveSessionManager(frame)._set_status('探索中', 'search3 attempt 4')

        self.assertEqual(
            frame.messages,
            ['探索中: search3 attempt 4 | AI 2 | Solve 7'],
        )

    def test_solve_status_is_optional_for_non_gui_frames(self):
        manager = SolveSessionManager(object())

        manager._set_status('探索中')


if __name__ == '__main__':
    unittest.main()

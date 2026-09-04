"""Tests for GUI-to-web playback link construction."""

import unittest
from urllib.parse import parse_qs, urlparse
from types import SimpleNamespace
from unittest.mock import patch

from core.web_playback import build_web_playback_url, web_puzzle_key
from ui.frame import Frame


class WebPlaybackTests(unittest.TestCase):
    def test_cube_sizes_use_cubing_puzzle_keys(self):
        self.assertEqual(web_puzzle_key("rubiks", 3), "3x3x3")
        self.assertEqual(web_puzzle_key("cube", 7), "7x7x7")
        self.assertIsNone(web_puzzle_key("cube", 8))

    def test_non_cube_puzzles_use_matching_keys(self):
        self.assertEqual(web_puzzle_key("megaminx", 3), "megaminx")
        self.assertEqual(web_puzzle_key("square1", 3), "square1")
        self.assertIsNone(web_puzzle_key("cto", 3))

    def test_playback_url_preserves_moves_and_setup(self):
        url = build_web_playback_url("3x3x3", ("R", "U", "R'"), ("F2", "E"))
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query, {
            "puzzle": ["3x3x3"],
            "moves": ["R U R'"],
            "setup": ["F2 E"],
        })

    def test_gui_opens_the_current_solve_with_setup_and_moves(self):
        class FakeFrame:
            puzzle_type = "rubiks"
            cube_size = 3
            solve_state = SimpleNamespace(
                s=("F2", "E"),
                move_lis=[("E'",), ("F2",)],
            )

            def __init__(self):
                self.messages = []

            def display_move_sequence(self, moves):
                return tuple(move.strip() for move in moves)

            def append_log(self, message):
                self.messages.append(message)

        frame = FakeFrame()
        with patch("ui.frame.webbrowser.open_new_tab") as open_new_tab:
            Frame.open_web_playback(frame)

        query = parse_qs(urlparse(open_new_tab.call_args.args[0]).query)
        self.assertEqual(query["puzzle"], ["3x3x3"])
        self.assertEqual(query["setup"], ["F2 E"])
        self.assertEqual(query["moves"], ["E' F2"])
        self.assertEqual(frame.messages, ["Web replay: 現在の解法をブラウザで開きました。"])

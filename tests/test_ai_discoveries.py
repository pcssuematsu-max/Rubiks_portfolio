"""Tests for the persistent AI discovery feed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ai_discoveries import AiDiscoveryStore


class AiDiscoveryStoreTests(unittest.TestCase):
    def test_keeps_the_shortest_solution_for_the_same_start_position(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            store = AiDiscoveryStore(path)

            self.assertEqual(store.save("3x3x3", ("R",), ("U", "R", "U'")), "added")
            self.assertEqual(store.save("3x3x3", ("R",), ("F", "U", "R", "U'")), "unchanged")
            self.assertEqual(store.save("3x3x3", ("R",), ("R2", "U2")), "shorter")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["discoveries"]), 1)
            self.assertEqual(payload["discoveries"][0]["setup"], ["R"])
            self.assertEqual(payload["discoveries"][0]["moves"], ["R2", "U2"])
            self.assertEqual(payload["discoveries"][0]["moveCount"], 2)

    def test_keeps_different_start_positions_as_separate_discoveries(self):
        with TemporaryDirectory() as temporary_directory:
            store = AiDiscoveryStore(Path(temporary_directory) / "ai-discoveries.json")
            store.save("3x3x3", ("R",), ("R'",))
            store.save("3x3x3", ("U",), ("U'",))

            payload = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["discoveries"]), 2)

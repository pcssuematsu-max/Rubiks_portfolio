"""Tests for the persistent AI discovery feed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ai_discoveries import AiDiscoveryStore, point_canonical_discovery_sequences
from cube.rubiks_cube import Rubiks_3


class AiDiscoveryStoreTests(unittest.TestCase):
    def test_keeps_the_shortest_solution_for_the_same_start_position(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            store = AiDiscoveryStore(path)

            self.assertEqual(store.save("3x3x3", ("R",), ("U", "R", "U'")), "added")
            self.assertEqual(store.save("3x3x3", ("R",), ("F", "U", "R", "U'")), "unchanged")
            self.assertEqual(store.save("3x3x3", ("R",), ("R2", "U2")), "shorter")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schemaVersion"], 1)
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

    def test_rejects_a_discovery_with_a_wrong_move_count(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "updatedAt": "2026-09-09T00:00:00+00:00",
                        "discoveries": [
                            {
                                "id": "e07091d3e192eb0d",
                                "puzzle": "3x3x3",
                                "setup": ["R"],
                                "moves": ["R'"],
                                "moveCount": 2,
                                "foundAt": "2026-09-09T00:00:00+00:00",
                                "updatedAt": "2026-09-09T00:00:00+00:00",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "moveCount"):
                AiDiscoveryStore(path)._read()

    def test_rejects_an_invalid_found_at_timestamp(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            store = AiDiscoveryStore(path)
            store.save("3x3x3", (), ("R",))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["discoveries"][0]["foundAt"] = "not-a-timestamp"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "foundAt"):
                store._read()

    def test_migrates_the_legacy_version_field(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            store = AiDiscoveryStore(path)
            self.assertEqual(store.save("3x3x3", (), ("R",)), "added")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["version"] = payload.pop("schemaVersion")
            path.write_text(json.dumps(payload), encoding="utf-8")

            store.save("3x3x3", ("U",), ("U'",))
            migrated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["schemaVersion"], 1)
            self.assertNotIn("version", migrated)

    def test_point_canonical_sequences_keep_the_solution_valid_for_setup(self):
        cube = Rubiks_3(size=3)
        setup = (" R ", " U ", " F ")
        moves = cube.invert_moves(setup)

        canonical_setup, canonical_moves = point_canonical_discovery_sequences(cube, setup, moves)

        cube.reset()
        for move in canonical_setup + canonical_moves:
            cube.make_move(move)
        self.assertTrue(cube.is_perfect())

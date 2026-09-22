"""Tests for the persistent AI discovery feed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ai_discoveries import (
    AiDiscoveryStore,
    discovery_effect_metadata,
    point_canonical_discovery_sequences,
    terminal_last_perm_sequences,
)
from core.myperm_effects import MypermEffectAnalyzer
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

    def test_saves_complete_effect_metadata_when_provided(self):
        with TemporaryDirectory() as temporary_directory:
            cube = Rubiks_3(size=3, RegisterMyperms=False)
            effect = MypermEffectAnalyzer(cube).analyze((" R ",))
            metadata = discovery_effect_metadata(effect)
            store = AiDiscoveryStore(Path(temporary_directory) / "ai-discoveries.json")

            store.save("3x3x3", (), ("R",), effect_metadata=metadata)

            record = json.loads(store.path.read_text(encoding="utf-8"))["discoveries"][0]
            self.assertEqual(record["effectName"], metadata["effectName"])
            self.assertEqual(record["effectClass"], metadata["effectClass"])
            self.assertEqual(record["effectLabel"], metadata["effectLabel"])
            self.assertEqual(record["effectCount"], metadata["effectCount"])
            self.assertEqual(record["orientationCount"], metadata["orientationCount"])

    def test_keeps_one_replay_for_the_same_discovered_procedure(self):
        with TemporaryDirectory() as temporary_directory:
            store = AiDiscoveryStore(Path(temporary_directory) / "ai-discoveries.json")
            self.assertEqual(store.save("3x3x3", ("R", "U"), ("F",)), "added")
            self.assertEqual(store.save("3x3x3", ("R", "U", "L"), ("F",)), "unchanged")
            self.assertEqual(store.save("3x3x3", (), ("F",)), "shorter")

            records = json.loads(store.path.read_text(encoding="utf-8"))["discoveries"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["setup"], [])
            self.assertEqual(records[0]["moves"], ["F"])

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

    def test_terminal_last_perm_keeps_the_preceding_state_as_setup(self):
        terminal = terminal_last_perm_sequences(
            ("R",),
            (("U", "F"), ("L", "D")),
        )

        self.assertEqual(terminal, (("R", "U", "F"), ("L", "D")))
        self.assertIsNone(terminal_last_perm_sequences(("R",), (("U",),)))
        self.assertIsNone(terminal_last_perm_sequences(("R",), ((), ("L",))))

    def test_terminal_last_perm_has_an_independent_record_id(self):
        with TemporaryDirectory() as temporary_directory:
            store = AiDiscoveryStore(Path(temporary_directory) / "ai-discoveries.json")
            metadata = {
                "effectName": "C3",
                "effectClass": "C3",
                "effectLabel": "コーナー：3巡回",
                "effectCount": 3,
                "orientationCount": 0,
            }

            store.save("3x3x3", ("R",), ("R'",), effect_metadata=metadata)
            store.save(
                "3x3x3",
                ("R",),
                ("U",),
                effect_metadata=metadata,
                discovery_kind="terminal-last-perm",
            )

            records = json.loads(store.path.read_text(encoding="utf-8"))["discoveries"]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record.get("discoveryKind", "full-solve") for record in records},
                {"full-solve", "terminal-last-perm"},
            )

    def test_point_canonical_sequences_keep_the_solution_valid_for_setup(self):
        cube = Rubiks_3(size=3)
        setup = (" R ", " U ", " F ")
        moves = cube.invert_moves(setup)

        canonical_setup, canonical_moves = point_canonical_discovery_sequences(cube, setup, moves)

        cube.reset()
        for move in canonical_setup + canonical_moves:
            cube.make_move(move)
        self.assertTrue(cube.is_perfect())

import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.ai_discoveries import _record_id, point_canonical_discovery_sequences
from core.myperm_points import MypermPointCalculator, load_myperm_points
from cube.rubiks_cube import Rubiks_3
from tools.migrate_ai_discoveries_points import migrate_file, migrate_payload


def record(cube, setup, moves, found_at):
    display_setup = [move.strip() for move in setup]
    display_moves = [move.strip() for move in moves]
    return {
        "id": _record_id(f"{cube.size}x{cube.size}x{cube.size}", display_setup),
        "puzzle": f"{cube.size}x{cube.size}x{cube.size}",
        "setup": display_setup,
        "moves": display_moves,
        "moveCount": len(display_moves),
        "foundAt": found_at,
        "updatedAt": found_at,
    }


class DiscoveryPointMigrationTests(unittest.TestCase):
    def test_file_migration_keeps_a_backup_and_original_permissions(self):
        cube = Rubiks_3(size=3, RegisterMyperms=False)
        payload = {
            "schemaVersion": 1,
            "updatedAt": "2026-09-01T00:00:00+00:00",
            "discoveries": [record(cube, (" R ",), (" R'",), "2026-09-01T00:00:00+00:00")],
        }
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ai-discoveries.json"
            original = (json.dumps(payload) + "\n").encode("utf-8")
            path.write_bytes(original)
            os.chmod(str(path), 0o644)

            stats, backup_path = migrate_file(path, write=True)

            self.assertEqual(stats["source_count"], 1)
            self.assertIsNotNone(backup_path)
            self.assertEqual(gzip.decompress(backup_path.read_bytes()), original)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))["discoveries"]), 1)

    def test_equivalent_orientations_share_one_canonical_record(self):
        cube = Rubiks_3(size=3, RegisterMyperms=False)
        setup = (" R ", " U ", " F ")
        moves = cube.invert_moves(setup)
        rotated_setup = cube.transform(setup, 7)
        rotated_moves = cube.transform(moves, 7)
        payload = {
            "schemaVersion": 1,
            "updatedAt": "2026-09-01T00:00:00+00:00",
            "discoveries": [
                record(cube, setup, moves, "2026-09-01T00:00:00+00:00"),
                record(cube, rotated_setup, rotated_moves, "2026-09-02T00:00:00+00:00"),
            ],
        }

        migrated, stats = migrate_payload(payload, migrated_at="2026-10-03T00:00:00+00:00")

        self.assertEqual(stats["source_count"], 2)
        self.assertEqual(stats["result_count"], 1)
        self.assertEqual(stats["merged_same_start"], 1)
        result = migrated["discoveries"][0]
        self.assertEqual(result["foundAt"], "2026-09-01T00:00:00+00:00")
        self.assertEqual(result["updatedAt"], "2026-09-02T00:00:00+00:00")
        self.assertEqual(result["id"], _record_id(result["puzzle"], result["setup"]))
        self.assertIn("effectName", result)

        calculator = MypermPointCalculator(cube, load_myperm_points())
        canonical_setup, canonical_moves = point_canonical_discovery_sequences(cube, setup, moves)
        self.assertEqual(result["setup"], [move.strip() for move in canonical_setup])
        self.assertEqual(result["moves"], [move.strip() for move in canonical_moves])
        self.assertEqual(
            calculator.point_for_moves(canonical_moves),
            max(calculator.point_for_moves(cube.transform(moves, index)) for index in range(48)),
        )

        second, second_stats = migrate_payload(migrated, migrated_at="2026-10-04T00:00:00+00:00")
        self.assertEqual(second, migrated)
        self.assertEqual(second_stats["reoriented"], 0)

    def test_same_procedure_with_redundant_setup_keeps_shorter_setup(self):
        cube = Rubiks_3(size=3, RegisterMyperms=False)
        setup = (" R ", " U ", " F ")
        moves = cube.invert_moves(setup)
        longer_setup = (" U ", " U'", *setup)
        payload = {
            "schemaVersion": 1,
            "updatedAt": "2026-09-01T00:00:00+00:00",
            "discoveries": [
                record(cube, longer_setup, moves, "2026-09-01T00:00:00+00:00"),
                record(cube, setup, moves, "2026-09-02T00:00:00+00:00"),
            ],
        }

        migrated, stats = migrate_payload(payload, migrated_at="2026-10-03T00:00:00+00:00")

        self.assertEqual(stats["result_count"], 1)
        self.assertEqual(stats["merged_same_procedure"], 1)
        self.assertEqual(len(migrated["discoveries"][0]["setup"]), len(setup))

    def test_same_canonical_start_keeps_shorter_solution(self):
        cube = Rubiks_3(size=3, RegisterMyperms=False)
        setup = (" R ",)
        short_moves = (" R'",)
        long_moves = (" R'", " U ", " U'")
        payload = {
            "schemaVersion": 1,
            "updatedAt": "2026-09-01T00:00:00+00:00",
            "discoveries": [
                record(cube, cube.transform(setup, 1), cube.transform(long_moves, 1), "2026-09-01T00:00:00+00:00"),
                record(cube, setup, short_moves, "2026-09-02T00:00:00+00:00"),
            ],
        }

        migrated, stats = migrate_payload(payload, migrated_at="2026-10-03T00:00:00+00:00")

        self.assertEqual(stats["merged_same_start"], 1)
        self.assertEqual(stats["result_count"], 1)
        result = migrated["discoveries"][0]
        self.assertEqual(result["moveCount"], 1)
        self.assertEqual(result["foundAt"], "2026-09-01T00:00:00+00:00")
        self.assertEqual(result["updatedAt"], "2026-09-02T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()

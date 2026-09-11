import csv
import json
from pathlib import Path
import tempfile
import unittest

from core.experiment_log import (
    ExperimentLogStore,
    completed_experiment_record,
    experiment_outcome,
)


class ExperimentLogStoreTests(unittest.TestCase):
    def test_appends_a_jsonl_and_csv_record_with_benchmark_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            jsonl_path = output_directory / "runs.jsonl"
            csv_path = output_directory / "runs.csv"
            record = completed_experiment_record(
                puzzle_type = "rubiks", cube_size = 7, search_mode = "search3",
                ai_index = 2, solve_index = 11, stage = 1, succeeded = True,
                search_succeeded = True, fallback_used = False,
                elapsed_seconds = 1.23456789, setup = (" R ", "U"),
                moves = ("R'",), root_score = 0.25, best_score = 0.75,
                end_reason = "solved", stats = (12, 34),
            )

            ExperimentLogStore(jsonl_path, csv_path).append(record)

            payload = json.loads(jsonl_path.read_text(encoding = "utf-8"))
            self.assertEqual(payload["schemaVersion"], 2)
            self.assertEqual(payload["puzzle"], "rubiks-7x7")
            self.assertEqual(payload["searchMode"], "search3")
            self.assertEqual(payload["elapsedSeconds"], 1.234568)
            self.assertEqual(payload["moveCount"], 1)
            self.assertEqual(payload["score"], 0.75)
            self.assertEqual(payload["setup"], ["R", "U"])
            self.assertTrue(payload["searchSucceeded"])
            self.assertFalse(payload["fallbackUsed"])
            self.assertEqual(payload["outcome"], "search_success")

            with csv_path.open(encoding = "utf-8", newline = "") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["puzzle"], "rubiks-7x7")
            self.assertEqual(rows[0]["moves"], "[\"R'\"]")

    def test_classifies_greedy_fallback_separately_from_search_success(self):
        self.assertEqual(
            experiment_outcome(
                search_succeeded = False,
                succeeded = True,
                fallback_used = True,
            ),
            "greedy_fallback_success",
        )
        self.assertEqual(
            experiment_outcome(
                search_succeeded = False,
                succeeded = False,
                fallback_used = True,
            ),
            "greedy_fallback_failed",
        )

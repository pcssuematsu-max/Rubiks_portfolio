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

    def test_summarizes_search_modes_and_selects_short_direct_searches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / "runs.jsonl",
                output_directory / "runs.csv",
            )
            store.append(self._record("search2", 14, 2.0, True, False))
            store.append(self._record("search2", 5, 1.0, False, True))
            store.append(self._record("search2", 0, 0.5, False, False))
            store.append(self._record("search3", 8, 3.0, True, False))

            summary = store.summarize()
            search2 = next(
                group for group in summary["groups"]
                if group["searchMode"] == "search2"
            )
            self.assertEqual(summary["totalRuns"], 4)
            self.assertEqual(search2["directSearchSuccessCount"], 1)
            self.assertEqual(search2["fallbackCompletedCount"], 1)
            self.assertEqual(search2["failedCount"], 1)
            self.assertEqual(search2["directSolutionMoves"]["minimum"], 14)
            self.assertEqual(search2["interestingDiscoveries"][0]["moveCount"], 14)

            summary_path = store.export_summary()
            exported = json.loads(summary_path.read_text(encoding = "utf-8"))
            self.assertEqual(exported["groups"], summary["groups"])

    @staticmethod
    def _record(search_mode, move_count, elapsed_seconds, search_succeeded, fallback_used):
        succeeded = search_succeeded or fallback_used
        return completed_experiment_record(
            puzzle_type = "rubiks",
            cube_size = 7,
            search_mode = search_mode,
            ai_index = 0,
            solve_index = move_count,
            stage = 0,
            succeeded = succeeded,
            search_succeeded = search_succeeded,
            fallback_used = fallback_used,
            elapsed_seconds = elapsed_seconds,
            setup = ("R",),
            moves = tuple("R" for _ in range(move_count)),
            root_score = 0.1,
            best_score = 0.2,
            end_reason = "solved" if succeeded else "budget",
        )

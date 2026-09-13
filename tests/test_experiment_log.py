import csv
import json
from pathlib import Path
import tempfile
import unittest

from core.experiment_log import (
    CSV_FIELDS,
    ExperimentLogStore,
    completed_experiment_record,
    experiment_outcome,
    rank_ai_metric_differences,
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
            self.assertEqual(payload["schemaVersion"], 3)
            self.assertEqual(payload["puzzle"], "rubiks-7x7")
            self.assertEqual(payload["searchMode"], "search3")
            self.assertEqual(payload["elapsedSeconds"], 1.234568)
            self.assertEqual(payload["moveCount"], 1)
            self.assertEqual(payload["score"], 0.75)
            self.assertEqual(payload["setup"], ["R", "U"])
            self.assertTrue(payload["searchSucceeded"])
            self.assertFalse(payload["fallbackUsed"])
            self.assertEqual(payload["outcome"], "search_success")
            self.assertEqual(payload["aiSettings"], {})

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

            csv_summary = store.summarize("csv")
            self.assertEqual(csv_summary["groups"], summary["groups"])

            summary_path = store.export_summary()
            exported = json.loads(summary_path.read_text(encoding = "utf-8"))
            self.assertEqual(exported["groups"], summary["groups"])
            with summary_path.with_name("ai-experiment-summary.csv").open(
                encoding = "utf-8", newline = ""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["directMovesMinimum"], "14")

            csv_source_path = store.export_summary(
                output_directory / "csv-source-summary.json",
                source = "csv",
            )
            csv_exported = json.loads(csv_source_path.read_text(encoding = "utf-8"))
            self.assertEqual(csv_exported["source"], "csv")
            self.assertEqual(csv_exported["groups"], summary["groups"])

    def test_summarizes_ai_settings_separately_and_ranks_result_differences(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / 'runs.jsonl',
                output_directory / 'runs.csv',
            )
            store.append(self._record(
                'search2', 10, 1.0, True, False,
                ai_index = 0,
                ai_settings = {'model': 'linear', 'learningRate': 0.001},
            ))
            store.append(self._record(
                'search2', 20, 3.0, True, False,
                ai_index = 1,
                ai_settings = {'model': 'transformer', 'learningRate': 0.01},
            ))
            store.append(self._record(
                'search2', 0, 2.0, False, False,
                ai_index = 1,
                ai_settings = {'model': 'transformer', 'learningRate': 0.01},
            ))

            summary = store.summarize()
            self.assertEqual(len(summary['aiGroups']), 2)
            transformer = next(group for group in summary['aiGroups'] if group['aiIndex'] == 1)
            self.assertEqual(transformer['aiSettings']['model'], 'transformer')
            self.assertEqual(transformer['directSearchSuccessRate'], 0.5)
            differences = rank_ai_metric_differences(summary['aiGroups'])
            self.assertIn('directSolutionMoves', {item['key'] for item in differences})
            self.assertIn('directSearchSuccessRate', {item['key'] for item in differences})

            store.export_summary()
            with (output_directory / 'ai-experiment-summary.csv').open(
                encoding = 'utf-8', newline = '',
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['aiIndex'], '0')
            self.assertIn('learningRate', rows[0]['aiSettings'])

    def test_extends_a_legacy_csv_header_before_appending_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            csv_path = output_directory / 'runs.csv'
            legacy_fields = CSV_FIELDS[:-1]
            with csv_path.open('w', encoding = 'utf-8', newline = '') as stream:
                writer = csv.DictWriter(stream, fieldnames = legacy_fields)
                writer.writeheader()
                writer.writerow({field: '' for field in legacy_fields})
            store = ExperimentLogStore(output_directory / 'runs.jsonl', csv_path)
            store.append(self._record(
                'search2', 3, 1.0, True, False,
                ai_settings = {'model': 'linear'},
            ))

            with csv_path.open(encoding = 'utf-8', newline = '') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]['aiSettings'], '{"model":"linear"}')

    @staticmethod
    def _record(
        search_mode,
        move_count,
        elapsed_seconds,
        search_succeeded,
        fallback_used,
        ai_index = 0,
        ai_settings = None,
    ):
        succeeded = search_succeeded or fallback_used
        return completed_experiment_record(
            puzzle_type = "rubiks",
            cube_size = 7,
            search_mode = search_mode,
            ai_index = ai_index,
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
            ai_settings = ai_settings,
        )

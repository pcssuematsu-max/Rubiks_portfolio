import csv
import json
from pathlib import Path
import tempfile
import unittest

from core.experiment_log import (
    CSV_FIELDS,
    EXPERIMENT_RECORDS_PER_AI_MODE,
    ExperimentLogStore,
    completed_experiment_record,
    experiment_outcome,
    rank_ai_metric_differences,
    _setup_effect_bands,
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
                direct_search = {
                    'attemptCount': 1,
                    'terminalEndReason': 'solved',
                    'totals': {'playoutCount': 34},
                },
            )

            ExperimentLogStore(jsonl_path, csv_path).append(record)

            payload = json.loads(jsonl_path.read_text(encoding = "utf-8"))
            self.assertEqual(payload["schemaVersion"], 5)
            self.assertEqual(payload["puzzle"], "rubiks-7x7")
            self.assertEqual(payload["searchMode"], "search3")
            self.assertEqual(payload["elapsedSeconds"], 1.234568)
            self.assertEqual(payload["moveCount"], 1)
            self.assertEqual(payload["score"], 0.75)
            self.assertEqual(payload["setup"], [])
            self.assertEqual(payload["setupEffect"], None)
            self.assertTrue(payload["searchSucceeded"])
            self.assertFalse(payload["fallbackUsed"])
            self.assertEqual(payload["outcome"], "search_success")
            self.assertEqual(payload['directSearch']['terminalEndReason'], 'solved')
            self.assertEqual(payload['directSearch']['totals']['playoutCount'], 34)
            self.assertEqual(payload["aiSettings"], {})

            with csv_path.open(encoding = "utf-8", newline = "") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["puzzle"], "rubiks-7x7")
            self.assertEqual(rows[0]["moves"], "[]")
            self.assertEqual(
                json.loads(rows[0]['directSearch'])['attemptCount'],
                1,
            )

    def test_records_orientation_normalized_cube_effect_metrics(self):
        record = completed_experiment_record(
            puzzle_type = "cube", cube_size = 3, search_mode = "search2",
            ai_index = 0, solve_index = 1, stage = 0, succeeded = False,
            search_succeeded = False, fallback_used = False, elapsed_seconds = 0.0,
            setup = ("x",), moves = (), root_score = None, best_score = None,
            end_reason = "budget",
        )

        self.assertEqual(record.setup_effect["movedPieceCount"], 0)
        self.assertEqual(record.setup_effect["orientationCount"], 0)
        self.assertEqual(record.setup_effect["orientationNormalizationMoveCount"], 1)

    def test_compact_bounds_each_ai_search_mode_window(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / "runs.jsonl", output_directory / "runs.csv",
            )
            for index in range(EXPERIMENT_RECORDS_PER_AI_MODE + 5):
                store.append(self._record("search2", index + 1, 1.0, True, False))

            original, retained = store.compact()
            self.assertEqual(original, EXPERIMENT_RECORDS_PER_AI_MODE + 5)
            self.assertEqual(retained, EXPERIMENT_RECORDS_PER_AI_MODE)
            self.assertEqual(store.summarize()["totalRuns"], EXPERIMENT_RECORDS_PER_AI_MODE)

    def test_compaction_keeps_fallback_policy_rank_without_stage_arrays(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / "runs.jsonl", output_directory / "runs.csv",
            )
            record = completed_experiment_record(
                puzzle_type="rubiks",cube_size=7,search_mode="search3",
                ai_index=10,solve_index=1,stage=0,succeeded=True,
                search_succeeded=False,fallback_used=True,elapsed_seconds=1.0,
                setup=(),moves=(),root_score=0.0,best_score=1.0,
                end_reason="greedy",direct_search={
                    "attemptCount":1,"terminalEndReason":"budget",
                    "totals":{"playoutCount":10000},
                    "finalAttempt":{
                        "fallbackFirstMovePolicyRank":4,
                        "fallbackFirstMovePolicyProbability":0.08,
                        "fallbackRootPolicyEntropy":2.1,
                        "fallbackRootPolicyTop1Probability":0.2,
                        "fallbackFirstMoveVisitShare":0.12,
                        "returnedFirstMoveVisitShare":0.45,
                        "returnedFirstMoveValueDelta":0.3,
                        "budgetStages":[{"stageIndex":1}],
                    },
                },
            )
            store.append(record)
            store.compact()

            saved = json.loads(store.path.read_text(encoding="utf-8").strip())
            final = saved["directSearch"]["finalAttempt"]
            self.assertEqual(final["fallbackFirstMovePolicyRank"],4)
            self.assertEqual(final["fallbackRootPolicyEntropy"],2.1)
            self.assertEqual(final["fallbackFirstMoveVisitShare"],0.12)
            self.assertEqual(final["returnedFirstMoveVisitShare"],0.45)
            self.assertEqual(final["returnedFirstMoveValueDelta"],0.3)
            self.assertNotIn("budgetStages",final)

    def test_effect_bands_separate_success_and_failed_root_visit_shares(self):
        effect = {'movedPieceCount':8,'orientationCount':2,'reducedMoveCount':12}
        rows = [
            {'setupEffect':effect,'outcome':'search_success','searchSucceeded':True,
             'searchMode':'search3','directSearch':{'finalAttempt':{
                 'returnedFirstMoveVisitShare':0.8,'returnedFirstMoveValueDelta':0.4,
             }}},
            {'setupEffect':effect,'outcome':'greedy_fallback_success','searchSucceeded':False,
             'searchMode':'search3','directSearch':{'finalAttempt':{
                 'returnedFirstMoveVisitShare':0.3,'fallbackFirstMoveVisitShare':0.1,
                 'returnedFirstMoveValueDelta':-0.2,
                 'fallbackFirstMovePolicyRank':4,
             }}},
        ]

        band = next(item for item in _setup_effect_bands(rows) if item['label']=='8-13')
        self.assertEqual(band['directFirstMoveVisitShare'],0.8)
        self.assertEqual(band['directFirstMoveVisitCount'],1)
        self.assertEqual(band['failedFirstMoveVisitShare'],0.3)
        self.assertEqual(band['failedFirstMoveVisitCount'],1)
        self.assertEqual(band['fallbackFirstMoveVisitShare'],0.1)
        self.assertEqual(band['fallbackFirstMoveVisitCount'],1)
        self.assertEqual(band['fallbackFirstMovePolicyRank'],4.0)
        self.assertEqual(band['directFirstMoveValueIncreaseRate'],1.0)
        self.assertEqual(band['failedFirstMoveValueIncreaseRate'],0.0)

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

    def test_returns_a_recent_direct_search_window_for_one_ai(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / 'runs.jsonl',
                output_directory / 'runs.csv',
            )
            store.append(self._record('search3', 8, 1.0, True, False, ai_index = 2))
            store.append(self._record('search3', 9, 1.0, False, True, ai_index = 2))
            store.append(self._record('search3', 10, 1.0, True, False, ai_index = 2))
            store.append(self._record('search2', 11, 1.0, True, False, ai_index = 2))
            store.append(self._record('search3', 12, 1.0, True, False, ai_index = 1))

            recent = store.recent_ai_results(
                2,
                puzzle = 'rubiks-7x7',
                search_mode = 'search3',
                limit = 2,
            )

            self.assertEqual(recent['runCount'], 2)
            self.assertEqual(recent['directSearchSuccessCount'], 1)
            self.assertEqual(recent['fallbackCount'], 1)
            self.assertEqual(recent['directSearchSuccessRate'], 0.5)

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

    def test_keeps_search3_budget_stage_settings_in_the_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / 'runs.jsonl',
                output_directory / 'runs.csv',
            )
            store.append(self._record(
                'search3', 3, 1.0, True, False,
                ai_settings={'search3Budget': {
                    'mode': 'progressive', 'stagePlayouts': [1000, 3000, 6000],
                }},
            ))

            settings = store.summarize()['aiGroups'][0]['aiSettings']
            self.assertEqual(settings['search3Budget']['mode'], 'progressive')
            self.assertEqual(settings['search3Budget']['stagePlayouts'], [1000, 3000, 6000])

    def test_summarizes_direct_success_in_setup_length_bands(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / 'runs.jsonl',
                output_directory / 'runs.csv',
            )
            store.append(self._record('search3', 3, 1.0, True, False, setup_length = 12))
            store.append(self._record('search3', 3, 1.0, False, True, setup_length = 12))
            store.append(self._record('search3', 4, 1.0, False, True, setup_length = 28))

            group = store.summarize()['groups'][0]
            bands = {band['label']: band for band in group['setupLengthBands']}

            self.assertEqual(bands['0-19']['classifiedRunCount'], 2)
            self.assertEqual(bands['0-19']['directSearchSuccessCount'], 1)
            self.assertEqual(bands['0-19']['directSearchSuccessRate'], 0.5)
            self.assertEqual(bands['20-29']['directSearchSuccessRate'], 0.0)
            self.assertEqual(bands['30-39']['runCount'], 0)

    def test_summarizes_search3_outcomes_by_observed_playout_depth(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            store = ExperimentLogStore(
                output_directory / 'runs.jsonl',
                output_directory / 'runs.csv',
            )
            store.append(self._record(
                'search3', 3, 1.0, True, False,
                direct_search = {'finalAttempt': {
                    'playoutDepthMedian': 4.0,
                    'playoutDepthMean': 4.5,
                    'treeNodeCount': 120,
                    'selectionCMean': 2.5,
                }},
            ))
            store.append(self._record(
                'search3', 3, 1.0, False, True,
                direct_search = {'finalAttempt': {
                    'playoutDepthMedian': 4.0,
                    'playoutDepthMean': 5.0,
                    'treeNodeCount': 180,
                    'selectionCMean': 3.0,
                }},
            ))

            group = store.summarize()['groups'][0]
            bands = {band['label']: band for band in group['search3DepthBands']}
            medium = bands['3-5']

            self.assertEqual(medium['runCount'], 2)
            self.assertEqual(medium['directSearchSuccessRate'], 0.5)
            self.assertEqual(medium['averageNodeCount'], 150.0)
            self.assertEqual(medium['averageSelectionC'], 2.75)
            self.assertEqual(medium['averagePlayoutDepth'], 4.75)
            self.assertEqual(
                store.recent_ai_search3_depth_bands(0)[1]['averageNodeCount'],
                150.0,
            )

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
        setup_length = 1,
        direct_search = None,
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
            setup = tuple("R" for _ in range(setup_length)),
            moves = tuple("R" for _ in range(move_count)),
            root_score = 0.1,
            best_score = 0.2,
            end_reason = "solved" if succeeded else "budget",
            ai_settings = ai_settings,
            direct_search = direct_search,
        )

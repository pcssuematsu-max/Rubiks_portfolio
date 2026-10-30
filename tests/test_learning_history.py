import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from managers.search_data import SearchDataManager
from managers.solve_session import SolveSessionManager
from core.learning_history import (
    LEARNING_HISTORY_RECORDS_PER_AI,
    LearningHistoryStore,
    completed_learning_record,
)


class _FakeAI:
    search_mode = 'search3'
    lr = 0.001
    lr_C = 0.25
    lr_v = 0.9
    lr_h = 0.8
    update_scale_shared = 1.0
    update_scale_policy = 0.5
    update_scale_value = 2.0
    search2_value_loss_type = 'myloss2'
    search3_rank_loss_mix = 0.05
    last_training_metrics = {
        'policyLoss': 0.12,
        'valueLoss': 0.34,
        'policyCePerState': 1.23,
        'policyEffectiveStateCount': 160.0,
        'valueBcePerState': 0.56,
        'valueMae': 0.12,
        'valueStartToEndDelta': 0.31,
        'valueTargetStartToEndDelta': 0.45,
        'valueEffectiveStateCount': 240.5,
        'valueSequenceCount': 80,
        'updatesDuringSolve': 8,
        'trainingDataCount': 100,
        'retainedDataCount': 60,
        'trainingSample': {
            'originalBatchCount': 50,
            'selectedBatchCount': 10,
            'recentBatchCount': 5,
            'batchMixMode': 'mixed',
            'recentReplayRatio': 0.5,
            'recentWindowSelectedItemCount': 8,
            'mixedBatchCount': 10,
            'selectedUniqueItemCount': 20,
            'shortSequenceMaxSteps': 11,
            'shortEligibleItemCount': 31,
            'shortSelectedItemCount': 4,
            'mediumBatchCount': 2,
            'longBatchCount': 3,
            'randomBatchCount': 2,
            'originalItemCount': 100,
            'selectedItemCount': 20,
            'selectedStateCount': 320,
            'remainderItemCount': 80,
            'longSequenceMinSteps': 20,
            'mediumSequenceMinSteps': 12,
            'mediumSequenceMaxSteps': 19,
            'mediumEligibleItemCount': 24,
            'mediumReservedItemCount': 4,
            'mediumSelectedItemCount': 5,
            'longEligibleItemCount': 30,
            'longReservedItemCount': 6,
            'longSelectedItemCount': 7,
            'longSelectedStepMean': 29.5,
            'longSelectedStepMax': 48,
            'longReplayRatio': 0.25,
            'shortReplayMinRatio': 0.2,
            'shortSelectedRatio': 0.2,
            'longReplayMaxRatio': 0.6,
            'longSelectedRatio': 0.35,
            'directSearchSelectedItemCount': 9,
            'fallbackSelectedItemCount': 11,
            'fallbackSelectedRatio': 0.55,
            'pairwiseFallbackMaxRatio': 0.25,
            'originalLongDirectEligibleItemCount': 17,
            'originalLongDirectRequiredItemCount': 3,
            'originalLongDirectSelectedItemCount': 4,
            'originalLongDirectOversampledItemCount': 1,
            'originalLongDirectMinRatio': 0.1,
            'hardEffectDirectMinRatio': 0.25,
            'hardEffectDirectEligibleItemCount': 5,
            'hardEffectDirectRequiredItemCount': 5,
            'hardEffectDirectSelectedBeforeItemCount': 3,
            'hardEffectDirectSelectedItemCount': 5,
            'hardEffectDirectOversampledItemCount': 2,
        },
    }


class LearningHistoryTests(unittest.TestCase):
    def test_search2_source_objective_uses_producer_loss_type(self):
        for loss_type, expected in (
            ('myloss2', 'original_search2'),
            ('steps_to_goal', 'steps_to_goal'),
            ('myloss2_pairwise', 'pairwise'),
        ):
            with self.subTest(loss_type=loss_type):
                source_ai = SimpleNamespace(
                    search_mode='search2', search2_value_loss_type=loss_type,
                )
                self.assertEqual(SearchDataManager.source_objective(source_ai), expected)
                self.assertEqual(
                    SolveSessionManager._training_source_objective(source_ai, 'search2'),
                    expected,
                )

    def test_compacts_history_per_ai_without_mixing_the_recent_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            records = []
            for index in range(LEARNING_HISTORY_RECORDS_PER_AI + 4):
                record = completed_learning_record(3, _FakeAI(), 1.0)
                record['timestamp'] = f'2026-01-01T00:00:{index:04d}+00:00'
                records.append(record)
            for index in range(2):
                record = completed_learning_record(4, _FakeAI(), 1.0)
                record['timestamp'] = f'2026-01-02T00:00:{index:04d}+00:00'
                records.append(record)

            LearningHistoryStore(path).append_many(records)

            retained = LearningHistoryStore(path).records()
            self.assertEqual(len(retained), LEARNING_HISTORY_RECORDS_PER_AI + 2)
            self.assertEqual(sum(record['aiIndex'] == 3 for record in retained), LEARNING_HISTORY_RECORDS_PER_AI)
            self.assertEqual(sum(record['aiIndex'] == 4 for record in retained), 2)

    def test_writes_a_learning_record_with_loss_settings_and_update_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            record = completed_learning_record(3, _FakeAI(), 1.23456789)
            LearningHistoryStore(path).append(record)

            payload = json.loads(path.read_text(encoding = 'utf-8'))
            saved = payload['records'][0]
            self.assertEqual(payload['schemaVersion'], 1)
            self.assertEqual(saved['aiIndex'], 3)
            self.assertEqual(saved['searchMode'], 'search3')
            self.assertEqual(saved['policyLoss'], 0.12)
            self.assertEqual(saved['policyCePerState'], 1.23)
            self.assertEqual(saved['valueBcePerState'], 0.56)
            self.assertEqual(saved['valueMae'], 0.12)
            self.assertEqual(saved['valueStartToEndDelta'], 0.31)
            self.assertEqual(saved['valueSequenceCount'], 80)
            self.assertEqual(saved['updatesDuringSolve'], 8)
            self.assertEqual(saved['learningRate']['base'], 0.001)
            self.assertEqual(saved['updateScales']['value'], 2.0)
            self.assertEqual(saved['trainingSample']['longReservedItemCount'], 6)
            self.assertEqual(saved['trainingSample']['batchMixMode'], 'mixed')
            self.assertEqual(saved['trainingSample']['mixedBatchCount'], 10)
            self.assertEqual(saved['trainingSample']['longSelectedStepMean'], 29.5)
            self.assertEqual(saved['trainingSample']['longReplayRatio'], 0.25)
            self.assertEqual(saved['trainingSample']['mediumSelectedItemCount'], 5)
            self.assertEqual(saved['trainingSample']['shortSelectedItemCount'], 4)
            self.assertEqual(saved['trainingSample']['shortReplayMinRatio'], 0.2)
            self.assertEqual(saved['trainingSample']['longReplayMaxRatio'], 0.6)
            self.assertEqual(saved['trainingSample']['longSelectedRatio'], 0.35)
            self.assertEqual(saved['trainingSample']['directSearchSelectedItemCount'], 9)
            self.assertEqual(saved['trainingSample']['fallbackSelectedItemCount'], 11)
            self.assertEqual(saved['trainingSample']['fallbackSelectedRatio'], 0.55)
            self.assertEqual(saved['trainingSample']['pairwiseFallbackMaxRatio'], 0.25)
            self.assertEqual(saved['trainingSample']['originalLongDirectEligibleItemCount'], 17)
            self.assertEqual(saved['trainingSample']['originalLongDirectRequiredItemCount'], 3)
            self.assertEqual(saved['trainingSample']['originalLongDirectSelectedItemCount'], 4)
            self.assertEqual(saved['trainingSample']['originalLongDirectOversampledItemCount'], 1)
            self.assertEqual(saved['trainingSample']['originalLongDirectMinRatio'], 0.1)
            self.assertEqual(saved['search3RankLossMix'], 0.05)

    def test_writes_multiple_records_with_one_history_operation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            first = completed_learning_record(1, _FakeAI(), 1.0)
            second = completed_learning_record(2, _FakeAI(), 2.0)

            LearningHistoryStore(path).append_many([first, second])

            saved = LearningHistoryStore(path).records()
            self.assertEqual([record['aiIndex'] for record in saved], [1, 2])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['updatedAt'], second['timestamp'])

    def test_writes_steps_to_goal_regression_calibration_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.last_training_metrics = dict(ai.last_training_metrics)
            ai.last_training_metrics['stepsToGoalRegression'] = {
                'stateCount': 12, 'trajectoryCount': 2,
                'targetMin': -8.0, 'targetMean': -4.0, 'targetMax': 0.0,
                'predictionMin': -3.0, 'predictionMean': -1.0, 'predictionMax': 1.0,
                'signedErrorMean': 3.0, 'maePerState': 3.2, 'huberPerState': 2.7,
                'lossWeight': 1.0,
                'valueSelection': {
                    'candidateRouteCount': 6, 'selectedRouteCount': 4,
                    'sourceExcludedRouteCount': 1, 'detourExcludedRouteCount': 1,
                    'longerDuplicateExcludedRouteCount': 0, 'oversampledStateCount': 3,
                    'conflictingIntermediateStateCount': 2,
                    'discardedNonshortestStateCount': 1,
                    'candidateStateBands': {'0-10': 5, '11-30': 4, '31+': 3},
                    'selectedStateBands': {'0-10': 4, '11-30': 4, '31+': 4},
                },
                'origins': {
                    'direct_search': {
                        'stateCount': 12, 'trajectoryCount': 2,
                        'targetMin': -8.0, 'targetMean': -4.0, 'targetMax': 0.0,
                        'predictionMin': -3.0, 'predictionMean': -1.0, 'predictionMax': 1.0,
                        'signedErrorMean': 3.0, 'maePerState': 3.2, 'huberPerState': 2.7,
                    },
                },
            }
            LearningHistoryStore(path).append(completed_learning_record(20, ai, 1.0))

            saved = LearningHistoryStore(path).records()[0]['stepsToGoalRegression']
            self.assertEqual(saved['stateCount'], 12)
            self.assertEqual(saved['trajectoryCount'], 2)
            self.assertEqual(saved['signedErrorMean'], 3.0)
            self.assertEqual(saved['lossWeight'], 1.0)
            self.assertEqual(saved['valueSelection']['detourExcludedRouteCount'], 1)
            self.assertEqual(saved['valueSelection']['conflictingIntermediateStateCount'], 2)
            self.assertEqual(saved['valueSelection']['discardedNonshortestStateCount'], 1)
            self.assertEqual(saved['origins']['direct_search']['stateCount'], 12)

    def test_writes_steps_to_goal_replay_diagnostics_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.search2_value_loss_type = 'steps_to_goal'
            ai.last_training_metrics = dict(ai.last_training_metrics)
            population = {
                'routeCount': 4,
                'remainingStepBands': {'0-10': 1, '11-30': 2, '31+': 1},
                'efficiencyBands': {'efficient': 2, 'medium': 1, 'detour': 1, 'unknown': 0},
                'routeEfficiencyMean': 1.75,
                'duplicateStartStateCount': 1,
                'duplicateStartStateRouteCount': 2,
                'conflictingStartStateCount': 1,
                'conflictingStartStateRouteCount': 2,
                'duplicateTargetStdMean': 3.0,
            }
            ai.last_training_metrics['stepsToGoalRegression'] = {
                'stateCount': 12, 'trajectoryCount': 2,
                'targetMin': -8.0, 'targetMean': -4.0, 'targetMax': 0.0,
                'predictionMin': -3.0, 'predictionMean': -1.0, 'predictionMax': 1.0,
                'signedErrorMean': 3.0, 'maePerState': 3.2, 'huberPerState': 2.7,
                'replay': {
                    'stratifiedRatio': 0.5,
                    'stratifiedBatchCount': 1,
                    'stratifiedItemCount': 2,
                    'available': population,
                    'selected': population,
                },
            }

            LearningHistoryStore(path).append(completed_learning_record(20, ai, 1.0))

            replay = LearningHistoryStore(path).records()[0]['stepsToGoalRegression']['replay']
            self.assertEqual(replay['selected']['conflictingStartStateCount'], 1)
            self.assertEqual(replay['stratifiedItemCount'], 2)

    def test_writes_training_source_counts_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.last_training_metrics = dict(ai.last_training_metrics)
            ai.last_training_metrics['trainingDataSources'] = {
                'available': {
                    'search3:direct_search': {'trajectoryCount': 8, 'stateCount': 80},
                    'steps_to_goal:fallback': {'trajectoryCount': 3, 'stateCount': 30},
                },
                'selected': {
                    'search3:direct_search': {'trajectoryCount': 4, 'stateCount': 40},
                    'steps_to_goal:fallback': {'trajectoryCount': 1, 'stateCount': 10},
                },
            }
            LearningHistoryStore(path).append(completed_learning_record(20, ai, 1.0))

            saved = LearningHistoryStore(path).records()[0]['trainingDataSources']
            self.assertEqual(saved['available']['search3:direct_search']['trajectoryCount'], 8)
            self.assertEqual(saved['selected']['steps_to_goal:fallback']['stateCount'], 10)

    def test_search3_weight_log_separates_direct_prefix_and_greedy(self):
        data = [
            SimpleNamespace(moves=(0, 1), search_mode='search3', sample_weight=1.0,
                            source_objective='search3', trajectory_source='direct-search'),
            SimpleNamespace(moves=(0, 1, 2), search_mode='search3', sample_weight=1.0,
                            source_objective='search3', trajectory_source='search3-fallback-prefix'),
            SimpleNamespace(moves=(0, 1), search_mode='myval', sample_weight=0.2,
                            source_objective='search3', trajectory_source='greedy-fallback'),
            SimpleNamespace(moves=(0, 1), search_mode='search2', sample_weight=1.0,
                            source_objective='original_search2', trajectory_source='direct-search'),
            SimpleNamespace(moves=(0, 1), search_mode='search2', sample_weight=1.0,
                            source_objective='steps_to_goal', trajectory_source='direct-search'),
            SimpleNamespace(moves=(0, 1), search_mode='search2', sample_weight=1.0,
                            source_objective='pairwise', trajectory_source='direct-search'),
        ]
        summary = Rubiks_3_AI._search3_training_weight_summary(range(len(data)), data)
        self.assertEqual(summary['search3:direct_search']['trajectoryCount'], 1)
        self.assertEqual(summary['search3:direct_search']['stateCount'], 3)
        self.assertAlmostEqual(summary['search3:direct_search']['policyWeightSum'], 2.0)
        self.assertAlmostEqual(summary['search3:direct_search']['valueWeightSum'], 3.0)
        self.assertAlmostEqual(summary['search3:fallback_prefix']['policyWeightSum'], 3.0)
        self.assertAlmostEqual(summary['search3:fallback_prefix']['valueWeightSum'], 4.0)
        self.assertAlmostEqual(summary['search3:greedy_fallback']['policyWeightSum'], 0.1)
        self.assertAlmostEqual(summary['search3:greedy_fallback']['valueWeightSum'], 0.15)
        for source in ('original_search2:direct_search', 'steps_to_goal:direct_search',
                       'pairwise:direct_search'):
            self.assertAlmostEqual(summary[source]['policyWeightSum'], 1.0)
        self.assertAlmostEqual(summary['original_search2:direct_search']['valueWeightSum'], 0.6)
        self.assertAlmostEqual(summary['steps_to_goal:direct_search']['valueWeightSum'], 2.4)
        self.assertAlmostEqual(summary['pairwise:direct_search']['valueWeightSum'], 2.4)

        search3_ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        self.assertEqual(search3_ai._search3_step_weight(data[0], 0), 1.0)
        self.assertEqual(search3_ai._search3_step_weight(data[0], 1), 1.0)
        self.assertEqual(search3_ai._search3_step_weight(data[2], 0), 0.05)
        self.assertEqual(search3_ai._search3_policy_step_weight(data[2], 0), 0.05)
        self.assertEqual(search3_ai._search3_terminal_step_weight(data[2]), 0.05)
        self.assertEqual(
            SearchDataManager(None).search3_sample_weight(
                SimpleNamespace(search_mode='myval')
            ),
            0.05,
        )
        for item, expected_value_weight in zip(data[3:], (0.2, 0.8, 0.8)):
            self.assertAlmostEqual(search3_ai._search3_step_weight(item, 0), expected_value_weight)
            self.assertAlmostEqual(search3_ai._search3_step_weight(item, 1), expected_value_weight)
            self.assertAlmostEqual(search3_ai._search3_terminal_step_weight(item), expected_value_weight)
            self.assertAlmostEqual(search3_ai._search3_policy_step_weight(item, 0), 0.5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.last_training_metrics = dict(ai.last_training_metrics)
            ai.last_training_metrics['search3TrainingWeights'] = summary
            LearningHistoryStore(path).append(completed_learning_record(10, ai, 1.0))
            saved = LearningHistoryStore(path).records()[0]['search3TrainingWeights']
            self.assertEqual(saved, summary)

    def test_search3_tensors_keep_policy_uniform_when_value_depends_on_source(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.cube = SimpleNamespace(makedata=lambda: np.array([1.0, 2.0], dtype='f'))
        ai._search3_policy_target = lambda *args: np.array([1.0, 0.0], dtype='f')
        ai._search3_value_target = lambda *args: 0.75
        for objective, expected_value_weight in (
            ('original_search2', 0.2),
            ('steps_to_goal', 0.8),
            ('pairwise', 0.8),
        ):
            with self.subTest(objective=objective):
                item = SimpleNamespace(
                    search_mode='search2', source_objective=objective, sample_weight=1.0,
                )
                x = np.zeros((2, 1), dtype='f')
                policy_targets = np.zeros((2, 1), dtype='f')
                value_targets = np.zeros((1, 1), dtype='f')
                value_weights = np.zeros((1, 1), dtype='f')
                policy_weights = np.zeros((1, 1), dtype='f')
                ai._write_search3_step_tensors(
                    item, ('R',), np.array([0.75], dtype='f'), 0,
                    x, policy_targets, value_targets, value_weights, policy_weights, 0,
                )
                self.assertAlmostEqual(float(policy_weights[0, 0]), 0.5)
                self.assertAlmostEqual(float(value_weights[0, 0]), expected_value_weight)


    def test_training_source_summary_separates_objective_and_fallback(self):
        direct = SimpleNamespace(source_objective='search3', trajectory_source='direct-search', moves=(0, 1))
        fallback = SimpleNamespace(source_objective='steps_to_goal', trajectory_origin='fallback', moves=(0,))
        pairwise = SimpleNamespace(source_objective='pairwise', trajectory_source='direct-search', moves=(0, 1, 2))

        summary = Rubiks_3_AI._training_data_source_summary(
            available_indices=[0, 1, 2],
            selected_indices=[1, 2],
            data_source=[direct, fallback, pairwise],
            state_count_fn=lambda item: len(item.moves) + 1,
        )

        self.assertEqual(summary['available']['search3:direct_search']['state_count'], 3)
        self.assertEqual(summary['available']['steps_to_goal:fallback']['trajectory_count'], 1)
        self.assertEqual(summary['selected']['steps_to_goal:fallback']['state_count'], 2)
        self.assertEqual(summary['available']['pairwise:direct_search']['trajectory_count'], 1)
        self.assertEqual(summary['selected']['pairwise:direct_search']['state_count'], 4)

    def test_writes_fixed_validation_metrics_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            ai = _FakeAI()
            ai.last_training_metrics = dict(ai.last_training_metrics)
            ai.last_training_metrics['fixedValidation'] = {
                'fixtureId': 'canonical-inverse-v1',
                'fixtureLengths': [12, 24, 36],
                'fixtureCount': 9,
                'stateCount': 216,
                'policyTop1Accuracy': 0.2,
                'policyTop3Accuracy': 0.5,
                'policyTargetProbability': 0.1,
                'policyCrossEntropy': 2.3,
                'valueRankCorrelation': 0.4,
                'valuePearsonCorrelation': 0.5,
                'valuePredictionMean': 0.6,
                'valuePredictionStd': 0.2,
                'valueTargetMean': 0.3,
                'valueTargetStd': 0.1,
                'valueMae': 0.2,
                'valueSignedError': -0.1,
                'valueRmse': 0.3,
                'valueBce': 0.7,
                'valuePathCrossEntropy': None,
                'byLength': {
                    '60': {
                        'fixtureCount': 3,
                        'stateCount': 180,
                        'policyTop1Accuracy': 0.1,
                        'policyTop3Accuracy': 0.2,
                        'policyTargetProbability': 0.1,
                        'policyCrossEntropy': 2.3,
                        'valueRankCorrelation': 0.4,
                        'valuePearsonCorrelation': 0.5,
                        'valuePredictionMean': 0.6,
                        'valuePredictionStd': 0.2,
                        'valueTargetMean': 0.3,
                        'valueTargetStd': 0.1,
                        'valueMae': 0.2,
                        'valueSignedError': -0.2,
                        'valueRmse': 0.4,
                        'valueBce': 0.7,
                        'valuePathCrossEntropy': None,
                    },
                },
            }
            remaining_step_metrics = dict(ai.last_training_metrics['fixedValidation']['byLength']['60'])
            ai.last_training_metrics['fixedValidation']['stepsToGoalByRemainingSteps'] = {
                label: {**remaining_step_metrics, 'stateCount': state_count}
                for label,state_count in (('0-10',36),('11-30',84),('31+',96))
            }
            ai.last_training_metrics['fixedValidation']['byEffect'] = {
                '8-13': {
                    'stateCount': 36,
                    'valueStepCount': 30,
                    'policyTargetRankMean': 2.5,
                    'policyTop1Accuracy': 0.25,
                    'valueDeltaMean': 0.2,
                    'valueImprovementRate': 0.75,
                },
            }
            ai.last_training_metrics['fixedValidation']['byEffectChoices'] = {
                '8-13': {
                    'sampleCount': 4,
                    'valueTargetRankMean': 2.0,
                    'valueTop1Accuracy': 0.5,
                    'valueCorrectVsBestOtherDeltaMean': -0.2,
                },
            }
            ai.last_training_metrics['fixedValidation']['transformerBranches'] = {
                'sampleCount': 45,
                'attentionRms': 0.3,
                'linearRms': 0.2,
                'linearToAttentionRms': 0.666,
                'preactivationCosine': 0.1,
                'full': {'policyTargetRankMean': 2.0, 'policyTop1Accuracy': 0.4, 'valueRankCorrelation': 0.5},
                'withoutAttention': {'policyTargetRankMean': 3.0, 'policyTop1Accuracy': 0.3, 'valueRankCorrelation': 0.4},
                'withoutLinear': {'policyTargetRankMean': 4.0, 'policyTop1Accuracy': 0.2, 'valueRankCorrelation': 0.3},
            }
            record = completed_learning_record(3, ai, 1.0)
            LearningHistoryStore(path).append(record)

            saved = LearningHistoryStore(path).records()[0]

            self.assertEqual(saved['fixedValidation']['fixtureId'], 'canonical-inverse-v1')
            self.assertEqual(saved['fixedValidation']['stateCount'], 216)
            self.assertAlmostEqual(saved['fixedValidation']['valueRankCorrelation'], 0.4)
            self.assertEqual(saved['fixedValidation']['byLength']['60']['stateCount'], 180)
            self.assertEqual(saved['fixedValidation']['valueSignedError'], -0.1)
            self.assertEqual(saved['fixedValidation']['valueRmse'], 0.3)
            self.assertEqual(saved['fixedValidation']['byLength']['60']['valueRmse'], 0.4)
            self.assertEqual(saved['fixedValidation']['stepsToGoalByRemainingSteps']['31+']['stateCount'], 96)
            self.assertEqual(saved['fixedValidation']['stepsToGoalByRemainingSteps']['31+']['valueSignedError'], -0.2)
            self.assertEqual(saved['fixedValidation']['byEffect']['8-13']['valueImprovementRate'], 0.75)
            self.assertEqual(saved['fixedValidation']['byEffect']['8-13']['valueStepCount'], 30)
            self.assertEqual(saved['fixedValidation']['byEffectChoices']['8-13']['valueTargetRankMean'], 2.0)
            self.assertEqual(saved['fixedValidation']['transformerBranches']['withoutLinear']['policyTargetRankMean'], 4.0)

    def test_ai_training_metrics_uses_processed_batch_count_as_update_count(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(2.0, 4.0, 30, 18, 4)

        self.assertEqual(ai.last_training_metrics['policyLoss'], 0.5)
        self.assertEqual(ai.last_training_metrics['valueLoss'], 1.0)
        self.assertEqual(ai.last_training_metrics['updatesDuringSolve'], 4)
        self.assertEqual(ai.last_training_metrics['trainingDataCount'], 30)
        self.assertEqual(ai.last_training_metrics['retainedDataCount'], 18)
        self.assertIsNone(ai.last_training_metrics['valueBcePerState'])

    def test_gradient_metrics_are_compacted_and_persisted(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._last_training_gradient_metrics = {
            'update_count': 2,
            'global_l2_sum': 10.0,
            'global_l2_max': 7.0,
            'global_rms_sum': 0.4,
            'policy_l2_sum': 3.0,
            'value_l2_sum': 5.0,
            'trunk_l2_sum': 8.0,
            'policy_share_sum': 0.3,
            'value_share_sum': 0.5,
        }
        metrics = ai._gradient_history_metrics()

        self.assertEqual(metrics['updateCount'], 2)
        self.assertEqual(metrics['globalL2Mean'], 5.0)
        self.assertEqual(metrics['globalL2Max'], 7.0)
        self.assertEqual(metrics['globalRmsMean'], 0.2)
        self.assertEqual(metrics['policyGradientShareMean'], 0.15)
        self.assertEqual(metrics['valueGradientShareMean'], 0.25)

        fake_ai = _FakeAI()
        fake_ai.last_training_metrics = dict(fake_ai.last_training_metrics)
        fake_ai.last_training_metrics['gradientMetrics'] = metrics
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            LearningHistoryStore(path).append(completed_learning_record(3, fake_ai, 1.0))
            saved = LearningHistoryStore(path).records()[0]
        self.assertEqual(saved['gradientMetrics'], metrics)

    def test_gradient_collector_separates_policy_value_and_trunk(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.gradient_log_enabled = True
        epoch_state = {'gradient_metrics': ai._new_gradient_metrics()}
        ai._record_gradient_arrays(epoch_state, {
            'WO_P': np.array([3.0, 4.0]),
            'WO_V': np.array([12.0]),
            'W1': np.array([5.0]),
        })
        ai._last_training_gradient_metrics = epoch_state['gradient_metrics']
        metrics = ai._gradient_history_metrics()

        self.assertAlmostEqual(metrics['globalL2Mean'], 13.928388, places = 5)
        self.assertAlmostEqual(metrics['globalRmsMean'], 6.964194, places = 5)
        self.assertEqual(metrics['policyHeadL2Mean'], 5.0)
        self.assertEqual(metrics['valueHeadL2Mean'], 12.0)
        self.assertEqual(metrics['trunkL2Mean'], 5.0)
        self.assertAlmostEqual(metrics['policyGradientShareMean'], 25.0 / 194.0)
        self.assertAlmostEqual(metrics['valueGradientShareMean'], 144.0 / 194.0)

    def test_search3_quality_metrics_are_weighted_by_state_and_keep_sequence_deltas(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        metrics = ai._search3_quality_metrics(
            # The final row is the value logit.  Every policy action and
            # value logit is zero, so both CE values have closed-form checks.
            np.zeros((3, 3), dtype = 'f'),
            {
                'policy_targets': np.array([
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                ], dtype = 'f'),
                'policy_weights': np.array([[1.0, 2.0, 0.0]], dtype = 'f'),
                'value_targets': np.array([[0.0, 1.0, 0.5]], dtype = 'f'),
                'sample_weights': np.array([[1.0, 2.0, 1.0]], dtype = 'f'),
                'value_indices': [0, 2, 3],
            },
        )
        ai._record_training_metrics(2.0, 4.0, 30, 18, 4, metrics)

        self.assertAlmostEqual(ai.last_training_metrics['policyCePerState'], 0.693147, places = 5)
        self.assertEqual(ai.last_training_metrics['policyEffectiveStateCount'], 3.0)
        self.assertAlmostEqual(ai.last_training_metrics['valueBcePerState'], 0.693147, places = 5)
        self.assertAlmostEqual(ai.last_training_metrics['valueMae'], 0.375, places = 6)
        self.assertAlmostEqual(ai.last_training_metrics['valueStartToEndDelta'], 0.0, places = 6)
        self.assertAlmostEqual(ai.last_training_metrics['valueTargetStartToEndDelta'], 0.5, places = 6)
        self.assertEqual(ai.last_training_metrics['valueSequenceCount'], 2)

    def test_legacy_history_record_remains_readable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            legacy_record = completed_learning_record(3, _FakeAI(), 1.0)
            for field in (
                'policyCePerState', 'policyEffectiveStateCount',
                'valueBcePerState', 'valueMae', 'valueStartToEndDelta',
                'valueTargetStartToEndDelta', 'valueEffectiveStateCount',
                'valueSequenceCount',
            ):
                del legacy_record[field]
            path.write_text(json.dumps({
                'schemaVersion': 1,
                'updatedAt': legacy_record['timestamp'],
                'records': [legacy_record],
            }), encoding = 'utf-8')

            records = LearningHistoryStore(path).records()

            self.assertEqual(len(records), 1)
            self.assertNotIn('valueBcePerState', records[0])

    def test_legacy_training_sample_without_medium_fields_remains_readable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'learning.json'
            legacy_record = completed_learning_record(3, _FakeAI(), 1.0)
            for field in (
                'mediumBatchCount', 'mediumSequenceMinSteps', 'mediumSequenceMaxSteps',
                'mediumEligibleItemCount', 'mediumReservedItemCount', 'mediumSelectedItemCount',
                'shortSequenceMaxSteps', 'shortEligibleItemCount', 'shortSelectedItemCount',
                'shortReplayMinRatio', 'shortSelectedRatio',
                'longReplayMaxRatio', 'longSelectedRatio',
            ):
                del legacy_record['trainingSample'][field]
            path.write_text(json.dumps({
                'schemaVersion': 1,
                'updatedAt': legacy_record['timestamp'],
                'records': [legacy_record],
            }), encoding = 'utf-8')

            saved = LearningHistoryStore(path).records()[0]['trainingSample']

            self.assertEqual(saved['longReplayRatio'], 0.25)
            self.assertNotIn('mediumSelectedItemCount', saved)
            self.assertNotIn('longSelectedRatio', saved)

    def test_ai_training_metrics_marks_loss_as_unavailable_when_no_update_ran(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai._record_training_metrics(0.0, 0.0, 5, 5, 0)

        self.assertIsNone(ai.last_training_metrics['policyLoss'])
        self.assertIsNone(ai.last_training_metrics['valueLoss'])

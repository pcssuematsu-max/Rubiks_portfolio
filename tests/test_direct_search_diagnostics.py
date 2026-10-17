from types import SimpleNamespace
import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from managers.solve_session import SolveSessionManager, SolveSessionState
from model.search_result import SearchResult


class DirectSearchDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.state = SolveSessionState()
        self.manager = SolveSessionManager(SimpleNamespace(solve_state=self.state))

    def test_search2_diagnostics_keep_the_direct_budget_reason(self):
        ai = SimpleNamespace(
            search_mode='search2',
            _last_search2_frontier_peak=31,
            _last_search2_frontier_remaining=7,
            _last_search2_value_count=42,
        )
        result = SearchResult(
            False, ('R',), 2.0, [2.0, 3.0], 3.0, np.array([4, 20]),
            search_mode='search2', end_reason='budget',
        )

        self.manager._record_direct_search_attempt(ai, result, 0.125)
        summary = self.manager._direct_search_summary()

        self.assertEqual(summary['terminalEndReason'], 'budget')
        self.assertEqual(summary['totals']['evaluatedStateCount'], 20)
        self.assertEqual(summary['finalAttempt']['frontierPeak'], 31)
        self.assertEqual(summary['finalAttempt']['bestImprovement'], 1.0)

    def test_pairwise_marks_high_value_peak_that_did_not_reach_goal(self):
        ai = SimpleNamespace(
            search_mode='search2', search2_value_loss_type='myloss2_pairwise',
            search2_value_loss_margin=1.0,
        )
        result = SearchResult(
            False,('R',),2.0,[2.0,3.5],3.5,np.array([3,20]),
            search_mode='search2',end_reason='budget',
        )

        self.manager._record_direct_search_attempt(ai,result,0.1)
        summary = self.manager._direct_search_summary()

        self.assertTrue(summary['finalAttempt']['highValuePeakWithoutGoal'])
        self.assertEqual(summary['totals']['highValuePeakAttemptCount'],1)
        self.assertEqual(summary['totals']['highValuePeakWithoutGoalAttemptCount'],1)

    def test_search3_diagnostics_aggregate_playouts_and_reset_per_solve(self):
        ai = SimpleNamespace(
            search_mode='search3',
            search3_engine=SimpleNamespace(
                node_cache={'a': object(), 'b': object()},
                prediction_cache={'a': object()},
            ),
        )
        first = SearchResult(
            False, ('R',), 0.2, [0.2, 0.3], 0.3, np.array([2, 40]),
            policy_target=np.array([0, 1, 2]), search_mode='search3', end_reason='budget',
        )
        second = SearchResult(
            False, ('U',), 0.3, [0.3, 0.4], 0.4, np.array([3, 50]),
            policy_target=np.array([0, 0, 3]), search_mode='search3', end_reason='budget',
            search_diagnostics={
                'playoutDepthMedian': 17.5,
                'playoutDepthAtLeast20Rate': 0.4,
                'selectionCMean': 2.8,
            },
        )

        self.manager._record_direct_search_attempt(ai, first, 0.2)
        self.manager._record_direct_search_attempt(ai, second, 0.3)
        summary = self.manager._direct_search_summary()

        self.assertEqual(summary['attemptCount'], 2)
        self.assertEqual(summary['totals']['playoutCount'], 90)
        self.assertEqual(summary['totals']['maxRootChildVisits'], 3)
        self.assertEqual(summary['finalAttempt']['treeNodeCount'], 2)
        self.assertEqual(summary['finalAttempt']['predictionCacheCount'], 1)
        self.assertEqual(summary['finalAttempt']['playoutDepthMedian'], 17.5)
        self.assertEqual(summary['finalAttempt']['selectionCMean'], 2.8)

        self.state.reset_direct_search_diagnostics()
        self.assertEqual(self.manager._direct_search_summary(), {})

    def test_experiment_settings_keep_the_search3_rank_loss_coefficient(self):
        settings = SolveSessionManager._experiment_ai_settings(
            SimpleNamespace(search3_rank_loss_mix = 0.05)
        )

        self.assertEqual(settings['search3RankLossMix'], 0.05)

    def test_search3_budget_summary_uses_all_stages_for_playout_totals(self):
        ai = SimpleNamespace(search_mode='search3', search3_engine=SimpleNamespace(node_cache={}))
        result = SearchResult(
            False, ('R',), 0.2, [0.2, 0.3], 0.3, np.array([40, 600]),
            search_mode='search3', end_reason='budget',
            search_diagnostics={'budgetSummary': {
                'mode': 'progressive',
                'stopReason': 'budget_exhausted',
                'stageCount': 3,
                'requestedPlayoutCount': 1000,
                'consumedPlayoutCount': 1000,
                'stages': [{'stageIndex': 3, 'decision': 'budget_exhausted'}],
            }},
        )

        self.manager._record_direct_search_attempt(ai, result, 0.1)
        final = self.manager._direct_search_summary()['finalAttempt']

        self.assertEqual(final['playoutCount'], 1000)
        self.assertEqual(final['budgetMode'], 'progressive')
        self.assertEqual(final['budgetStopReason'], 'budget_exhausted')

    def test_search3_logs_returned_move_root_visits_without_extra_prediction(self):
        self.manager.frame = SimpleNamespace(
            solve_state=self.state,
            cube=SimpleNamespace(move_keys=('R','U','F'),key_to_num={'R':0,'U':1,'F':2}),
        )
        ai = SimpleNamespace(
            search_mode='search3',
            search3_engine=SimpleNamespace(root_node=SimpleNamespace(P=np.array([0.2,0.7,0.1]))),
        )
        result = SearchResult(
            True,('U',),0.2,[0.2,1.0],1.0,np.array([7,10]),
            policy_target=np.array([2,7,1]),search_mode='search3',end_reason='solved',
        )

        self.manager._record_direct_search_attempt(ai,result,0.1)
        attempt = self.state.direct_search_attempts[-1]
        self.assertEqual(attempt['returnedFirstMovePolicyRank'],1)
        self.assertAlmostEqual(attempt['returnedFirstMoveVisitShare'],0.7)
        self.assertAlmostEqual(attempt['returnedFirstMoveValueDelta'],0.8)
        self.assertAlmostEqual(attempt['rootVisitTopShare'],0.7)

    def test_transformer_search3_records_fallback_first_move_policy_rank_once(self):
        self.state.fallback_used = True
        self.state.val_lis.append(0.0)
        self.state.direct_search_attempts.append({'searchMode': 'search3'})
        cube = SimpleNamespace(
            move_keys=('R','U','F'),
            key_to_num={'R': 0, 'U': 1, 'F': 2},
            state=np.asarray(['A']),
            makedata=lambda: np.asarray([0.0],dtype='f'),
        )
        self.manager.frame = SimpleNamespace(
            solve_state=self.state, cube=cube, AI_idx=0, AInum=1,
            myval_AI=SimpleNamespace(),
        )
        self.manager._supports_myperms_greedy = lambda: True
        self.manager._set_status = lambda *args: None
        self.manager._myperms_sort_weights = lambda ai: None
        self.manager._current_search_scramble = lambda: ()
        self.manager._choose_and_apply_myperms = lambda ai, weights: self.state.move_lis.append(('U',)) or True
        self.manager._record_myval_history = lambda *args: None
        prediction_count = []

        def predict(x,policy,value):
            prediction_count.append(1)
            return np.asarray([[0.0],[1.0],[2.0]],dtype='f')

        # Use the production model type: the old SimpleNamespace fixture hid
        # a missing is_transformer attribute on Rubiks_3_AI.
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.use_transformer_attention = True
        ai.search_mode = 'search3'
        ai.search3_engine = SimpleNamespace(root_node=SimpleNamespace(
            state_key='A',visited=np.asarray([1,3,1]),
        ))
        ai.predict = predict
        self.assertTrue(self.manager._advance_greedy_step(ai))
        self.assertTrue(self.manager._advance_greedy_step(ai))

        attempt = self.state.direct_search_attempts[-1]
        self.assertEqual(attempt['fallbackFirstMovePolicyRank'],2)
        self.assertGreater(attempt['fallbackFirstMovePolicyProbability'],0.0)
        self.assertGreater(attempt['fallbackRootPolicyEntropy'],0.0)
        self.assertAlmostEqual(attempt['fallbackFirstMoveVisitShare'],0.6)
        self.assertEqual(len(prediction_count),1)

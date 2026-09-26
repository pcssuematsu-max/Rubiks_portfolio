from types import SimpleNamespace
import unittest

import numpy as np

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

    def test_search3_diagnostics_aggregate_playouts_and_reset_per_solve(self):
        ai = SimpleNamespace(
            search_mode='search3',
            search3_engine=SimpleNamespace(node_cache={'a': object(), 'b': object()}),
        )
        first = SearchResult(
            False, ('R',), 0.2, [0.2, 0.3], 0.3, np.array([2, 40]),
            policy_target=np.array([0, 1, 2]), search_mode='search3', end_reason='budget',
        )
        second = SearchResult(
            False, ('U',), 0.3, [0.3, 0.4], 0.4, np.array([3, 50]),
            policy_target=np.array([0, 0, 3]), search_mode='search3', end_reason='budget',
        )

        self.manager._record_direct_search_attempt(ai, first, 0.2)
        self.manager._record_direct_search_attempt(ai, second, 0.3)
        summary = self.manager._direct_search_summary()

        self.assertEqual(summary['attemptCount'], 2)
        self.assertEqual(summary['totals']['playoutCount'], 90)
        self.assertEqual(summary['totals']['maxRootChildVisits'], 3)
        self.assertEqual(summary['finalAttempt']['treeNodeCount'], 2)

        self.state.reset_direct_search_diagnostics()
        self.assertEqual(self.manager._direct_search_summary(), {})


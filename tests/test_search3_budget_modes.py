import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from model.search_result import SearchResult


def _result(max_visits, playouts, root_value=0.20, best_value=0.30, depth=4.0):
    return SearchResult(
        False,
        ('R', 'U'),
        root_value,
        [root_value, best_value],
        best_value,
        np.array([max_visits, playouts]),
        search_mode='search3',
        end_reason='budget',
        search_diagnostics={'playoutDepthMean': depth},
    )


class Search3BudgetModeTests(unittest.TestCase):
    def _ai(self, mode, results):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search3_budget_mode = mode
        ai.search3_budget_stage_playouts = (100, 300, 600)
        ai.search3_budget_confidence_visit_share = 0.70
        ai.search3_budget_min_improvement = 0.05
        ai.search3_budget_min_playout_depth = 3.0
        ai.search3_budget_rescue_playouts = 0
        ai.search3_budget_rescue_min_visit_share_gain = 1.0
        ai.search3_budget_rescue_min_improvement = 0.0
        ai.search_num3 = 100
        ai.search_repeat3 = 3
        requested = []

        def search3(playouts):
            requested.append(playouts)
            return results.pop(0)

        ai.search3 = search3
        return ai, requested

    def test_fixed_mode_preserves_repeated_allocation(self):
        ai, requested = self._ai('fixed', [
            _result(20, 100), _result(30, 100), _result(40, 100),
        ])

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 100, 100])
        self.assertEqual(result.search_diagnostics['budgetSummary']['mode'], 'fixed')
        self.assertEqual(result.search_diagnostics['budgetSummary']['consumedPlayoutCount'], 300)

    def test_progressive_mode_advances_after_a_confident_probe(self):
        ai, requested = self._ai('progressive', [_result(80, 100)])

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100])
        self.assertEqual(result.end_reason, 'improvement')
        self.assertEqual(result.moves, ('R',))
        summary = result.search_diagnostics['budgetSummary']
        self.assertEqual(summary['stopReason'], 'advance_confident')
        self.assertEqual(summary['stages'][0]['decision'], 'advance_confident')

    def test_progressive_mode_skips_final_tier_for_an_unpromising_middle(self):
        ai, requested = self._ai('progressive', [
            _result(20, 100, best_value=0.22, depth=1.0),
            _result(100, 300, best_value=0.22, depth=1.0),
        ])

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 300])
        self.assertEqual(
            result.search_diagnostics['budgetSummary']['stopReason'],
            'fallback_unpromising',
        )

    def test_progressive_mode_spends_final_tier_for_a_promising_middle(self):
        ai, requested = self._ai('progressive', [
            _result(20, 100, best_value=0.22, depth=1.0),
            _result(100, 300, best_value=0.30, depth=4.0),
            _result(100, 600, best_value=0.30, depth=4.0),
        ])

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 300, 600])
        summary = result.search_diagnostics['budgetSummary']
        self.assertEqual(summary['stageCount'], 3)
        self.assertEqual(summary['consumedPlayoutCount'], 1000)
        self.assertEqual(summary['stopReason'], 'budget_exhausted')

    def test_progressive_mode_escalates_when_root_concentration_improves(self):
        ai, requested = self._ai('progressive', [
            _result(20, 100, best_value=0.22, depth=1.0),
            _result(100, 300, best_value=0.22, depth=4.0),
            _result(100, 600, best_value=0.22, depth=4.0),
        ])
        ai.search3_budget_min_visit_share_gain = 0.03

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 300, 600])
        stages = result.search_diagnostics['budgetSummary']['stages']
        self.assertGreater(stages[1]['rootVisitShareGain'], 0.03)
        self.assertEqual(stages[1]['decision'], 'escalate')

    def test_progressive_mode_uses_rescue_tier_only_for_a_growing_final_root(self):
        ai, requested = self._ai('progressive', [
            _result(20, 100, best_value=0.22, depth=1.0),
            _result(100, 300, best_value=0.22, depth=4.0),
            _result(300, 600, best_value=0.22, depth=4.0),
            _result(190, 200, best_value=0.22, depth=4.0),
        ])
        ai.search3_budget_min_visit_share_gain = 0.03
        ai.search3_budget_rescue_playouts = 200
        ai.search3_budget_rescue_min_visit_share_gain = 0.05

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 300, 600, 200])
        summary = result.search_diagnostics['budgetSummary']
        self.assertEqual(summary['stopReason'], 'rescue_budget_exhausted')
        self.assertEqual(summary['stageCount'], 4)
        self.assertEqual(summary['stages'][2]['decision'], 'rescue')
        self.assertEqual(summary['stages'][3]['decision'], 'rescue_budget_exhausted')

    def test_progressive_mode_skips_rescue_when_value_regresses(self):
        ai, requested = self._ai('progressive', [
            _result(20, 100, best_value=0.22, depth=1.0),
            _result(100, 300, best_value=0.22, depth=4.0),
            _result(300, 600, best_value=0.15, depth=4.0),
        ])
        ai.search3_budget_min_visit_share_gain = 0.03
        ai.search3_budget_rescue_playouts = 200
        ai.search3_budget_rescue_min_visit_share_gain = 0.05
        ai.search3_budget_rescue_min_improvement = 0.0

        result = ai._search3_with_repeats()

        self.assertEqual(requested, [100, 300, 600])
        self.assertEqual(result.search_diagnostics['budgetSummary']['stopReason'], 'budget_exhausted')

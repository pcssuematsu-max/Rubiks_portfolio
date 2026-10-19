import random
import unittest
from types import SimpleNamespace

import numpy as np

from ai.paired_solve_benchmark import (
    EFFECT_BANDS, fixture_set_id, make_paired_fixtures, make_palm_fixtures,
    probe_position_neighborhood, run_direct_solve_case, summarize_direct_cases,
)
from cube.rubiks_cube import Rubiks_3
from ai.rubiks_ai import Rubiks_3_AI
from model.search_result import SearchResult
from tools.compare_ai_same_positions import _indices, _paired_summary


class _FakeAI:
    def __init__(self, results, mode='search2'):
        self.cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
        self.search_mode = mode
        self._results = iter(results)
        self.search3_engine = SimpleNamespace(reset_tree=lambda: None, advance_root=lambda _move: None)

    def search(self):
        return next(self._results)

    def clear_training_cache(self, collect=False):
        return None

    def learn(self):
        raise AssertionError('The benchmark must not train')


def _fixture(scramble, cube):
    import hashlib
    from core.experiment_log import setup_effect_metrics

    cube.reset()
    cube.scramble(0, Move=scramble)
    state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
    cube.reset()
    return {
        'fixtureId': 'test-1',
        'band': '0-7',
        'scramble': list(scramble),
        'setupEffect': setup_effect_metrics('cube-3x3', 3, scramble),
        'initialStateHash': state_hash,
    }


class PairedSolveBenchmarkTests(unittest.TestCase):
    def test_neighborhood_probe_restores_cube_and_reports_ranks(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2')
        ai.cube.scramble(2)
        original = ai.cube.state.copy()
        weights = {key: value.copy() for key,value in ai.params.items()}
        reference = ai.cube.move_keys[0]
        result = probe_position_neighborhood(ai,reference)
        self.assertGreaterEqual(result['referencePolicyRank'],1)
        self.assertGreaterEqual(result['referenceValueRank'],1)
        self.assertEqual(result['referencePolicyRank'],result['policyRankByMove'][reference])
        self.assertEqual(result['referenceValueRank'],result['valueRankByMove'][reference])
        self.assertEqual(len(result['topPolicyMoves']),3)
        self.assertEqual(len(result['topValueMoves']),3)
        self.assertLessEqual(result['improvingSuccessorCount'],len(ai.cube.move_keys))
        np.testing.assert_array_equal(ai.cube.state,original)
        for key,value in weights.items():
            np.testing.assert_array_equal(ai.params[key],value)

    def test_fixtures_are_balanced_unique_and_deterministic(self):
        before = random.getstate()
        fixtures = make_paired_fixtures(per_band=1)
        again = make_paired_fixtures(per_band=1)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(fixtures, again)
        self.assertEqual(len(fixtures), len(EFFECT_BANDS))
        self.assertEqual(len({row['initialStateHash'] for row in fixtures}), len(fixtures))
        self.assertEqual(fixture_set_id(fixtures), fixture_set_id(again))
        self.assertGreaterEqual(fixtures[-1]['setupEffect']['reducedMoveCount'], 40)

    def test_user_supplied_palms_have_low_effect_without_reduction(self):
        fixtures = make_palm_fixtures()
        self.assertEqual([row['fixtureId'] for row in fixtures], ['T-palm', 'Y-palm'])
        self.assertEqual(
            [row['setupEffect']['movedPieceCount'] for row in fixtures], [4, 4]
        )
        self.assertEqual(
            [row['setupEffect']['reducedMoveCount'] for row in fixtures], [14, 17]
        )
        self.assertEqual(
            [len(row['scramble']) for row in fixtures], [14, 17]
        )
        self.assertEqual(len({row['initialStateHash'] for row in fixtures}), 2)
        self.assertEqual(fixtures, make_palm_fixtures())

    def test_search2_follows_multiple_segments_and_restores_state(self):
        ai = _FakeAI([
            SearchResult(False, (" R'",), -2, [-2, -1], -1, [1, 10], end_reason='improvement'),
            SearchResult(True, (" U'",), -1, [-1, 0], 0, [1, 8], end_reason='solved'),
        ])
        ai.cube.scramble(2)
        original = ai.cube.state.copy()
        fixture = _fixture((' U ', ' R '), ai.cube)
        ai.cube.state[:] = original
        row = run_direct_solve_case(ai, fixture)
        self.assertTrue(row['directSolved'])
        self.assertEqual(row['searchCalls'], 2)
        self.assertEqual(row['evaluatedStates'], 18)
        self.assertEqual(row['moveCount'], 2)
        np.testing.assert_array_equal(ai.cube.state, original)

    def test_search3_budget_does_not_count_greedy_recovery(self):
        result = SearchResult(
            False, (" U'",), 0.2, [0.2, 0.5], 0.5, [3, 10],
            search_mode='search3', end_reason='budget',
            search_diagnostics={'playoutDepthMean': 4.0, 'playoutDepthMax': 7, 'treeNodeCount': 8},
        )
        ai = _FakeAI([result], mode='search3')
        fixture = _fixture((' U ',), ai.cube)
        row = run_direct_solve_case(ai, fixture)
        self.assertFalse(row['directSolved'])
        self.assertEqual(row['terminalReason'], 'greedy_fallback_required')
        self.assertEqual(row['moveCount'], 0)
        self.assertEqual(row['playouts'], 10)
        self.assertEqual(row['playoutDepthMean'], 4.0)
        self.assertTrue(ai.cube.is_perfect())

    def test_summary_and_index_selection(self):
        rows = [
            {'band': '0-7', 'directSolved': True, 'evaluatedStates': 10, 'playouts': 0, 'searchCalls': 2},
            {'band': '0-7', 'directSolved': False, 'evaluatedStates': 20, 'playouts': 0, 'searchCalls': 1},
        ]
        summary = summarize_direct_cases(rows)
        self.assertEqual(summary['solved'], 1)
        self.assertEqual(summary['byBand']['0-7']['meanEvaluatedStates'], 15)
        self.assertEqual(_indices('3,1,3', 5), [3, 1])
        with self.assertRaises(ValueError):
            _indices('5', 5)

    def test_paired_summary_rejects_different_start_states(self):
        fixtures = [{'fixtureId': '1-1', 'band': '0-7', 'initialStateHash': 'same'}]
        records = [
            {'aiIndex': 0, 'cases': [{'fixtureId': '1-1', 'initialStateHash': 'same', 'directSolved': True}]},
            {'aiIndex': 1, 'cases': [{'fixtureId': '1-1', 'initialStateHash': 'same', 'directSolved': False}]},
        ]
        self.assertEqual(_paired_summary(records, fixtures)[0]['solvedBy'], [0])
        records[1]['cases'][0]['initialStateHash'] = 'different'
        with self.assertRaises(ValueError):
            _paired_summary(records, fixtures)

    def test_paired_summary_compares_successful_linear_move_ranks(self):
        fixtures = [{'fixtureId':'1-1','band':'0-7','initialStateHash':'same'}]
        records = [
            {'aiIndex':20,'settings':{'model':'linear'},'cases':[{
                'fixtureId':'1-1','initialStateHash':'same','directSolved':True,
                'firstMove':' R ',
                'initialNeighborhood':{
                    'policyRankByMove':{' R ':1},'valueRankByMove':{' R ':2},
                },
            }]},
            {'aiIndex':22,'settings':{'model':'transformer'},'cases':[{
                'fixtureId':'1-1','initialStateHash':'same','directSolved':False,
                'initialNeighborhood':{
                    'policyRankByMove':{' R ':12},'valueRankByMove':{' R ':18},
                },
            }]},
        ]
        summary = _paired_summary(records,fixtures)[0]
        self.assertEqual(summary['linearReferenceMoves'][0]['ranksByAi']['22'],{
            'policyRank':12,'valueRank':18,
        })


if __name__ == '__main__':
    unittest.main()

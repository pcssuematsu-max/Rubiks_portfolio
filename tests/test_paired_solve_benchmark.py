import io
import json
import random
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai.paired_solve_benchmark import (
    EFFECT_BANDS, fixture_set_id, make_4x4_algorithm_fixtures,
    make_paired_fixtures, make_palm_fixtures, make_perm_fixtures,
    probe_position_neighborhood, probe_reference_route, probe_two_step_escape,
    run_direct_solve_case, run_staged_search3_case,
    summarize_direct_cases,
)
from cube.rubiks_cube import Rubiks_3
from ai.rubiks_ai import Rubiks_3_AI
from model.search_result import SearchResult
from tools.compare_ai_same_positions import _indices, _paired_summary, main


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

        detailed = probe_position_neighborhood(ai, reference, include_successors=True)
        self.assertEqual(len(detailed['successors']), len(ai.cube.move_keys))
        self.assertEqual(
            detailed['legalSuccessorCount'],
            sum(successor['isLegal'] for successor in detailed['successors']),
        )
        self.assertLessEqual(
            detailed['legalImprovingSuccessorCount'],
            detailed['improvingSuccessorCount'],
        )
        self.assertAlmostEqual(
            sum(successor['policyProbability'] for successor in detailed['successors']),
            1.0,
        )
        self.assertEqual(
            detailed['search2RootEligibleImprovingSuccessorCount'],
            sum(
                successor['isLegal'] and successor['valueDelta'] > 0
                and successor['policyProbability']
                >= detailed['search2RootEnqueueProbabilityThreshold']
                for successor in detailed['successors']
            ),
        )
        two_step = probe_two_step_escape(ai, detailed)
        self.assertLessEqual(two_step['sampledFirstMoveCount'], 6)
        np.testing.assert_array_equal(ai.cube.state, original)

    def test_reference_route_probe_finds_each_step_and_restores_cube(self):
        ai = Rubiks_3_AI([8], cube_size=3, search_mode='search2')
        scramble = (' U ', ' R ')
        fixture = _fixture(scramble, ai.cube)
        route = tuple(ai.cube.invert_moves(scramble))
        ai.cube.scramble(0, Move=(' F ',))
        original = ai.cube.state.copy()
        result = probe_reference_route(ai, fixture, route)
        self.assertEqual(result['routeMoveCount'], 2)
        self.assertEqual([row['referenceMove'] for row in result['steps']], list(route))
        self.assertEqual([row['step'] for row in result['steps']], [1, 2])
        self.assertTrue(all(row['policyRank'] >= 1 for row in result['steps']))
        self.assertTrue(all(row['valueRank'] >= 1 for row in result['steps']))
        self.assertAlmostEqual(
            result['steps'][0]['valueFromRouteStart'],
            result['steps'][0]['successorValue'] - result['routeStartValue'],
        )
        self.assertEqual(
            result['summary']['maxValueFromStart'],
            max(row['valueFromRouteStart'] for row in result['steps']),
        )
        self.assertNotIn('policyRankByMove', result['steps'][0])
        prefix = probe_reference_route(ai, fixture, route, max_probe_steps=1)
        self.assertEqual(prefix['routeMoveCount'], 2)
        self.assertEqual(prefix['probedMoveCount'], 1)
        self.assertEqual(len(prefix['steps']), 1)
        np.testing.assert_array_equal(ai.cube.state, original)
        with self.assertRaisesRegex(ValueError, 'does not solve'):
            probe_reference_route(ai, fixture, (' U ',))
        np.testing.assert_array_equal(ai.cube.state, original)

    def test_search3_fixed_route_probe_reports_prior_and_visit_leaders(self):
        ai = Rubiks_3_AI([8], cube_size=3, search_mode='search3')
        scramble = (' U ', ' R ')
        fixture = _fixture(scramble, ai.cube)
        route = tuple(ai.cube.invert_moves(scramble))
        probe = probe_reference_route(
            ai, fixture, route, max_probe_steps=1,
            search3_probe_playouts=12, trace_search3_root=True,
        )['steps'][0]['search3Root']
        self.assertIn(probe['topPolicyMove'], ai.cube.move_keys)
        self.assertIn(probe['topVisitedMove'], ai.cube.move_keys)
        self.assertGreaterEqual(probe['topPolicyPrior'], probe['referencePrior'])
        self.assertGreaterEqual(probe['topVisitShare'], probe['referenceVisitShare'])
        self.assertGreaterEqual(probe['topVisitShare'], probe['topPolicyVisitShare'])
        self.assertIsNotNone(probe['topVisitedPrior'])
        self.assertIsNotNone(probe['topVisitedBackedMean'])

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

    def test_user_supplied_perms_have_low_effect_without_reduction(self):
        fixtures = make_perm_fixtures()
        self.assertEqual([row['fixtureId'] for row in fixtures], ['T-perm', 'Y-perm'])
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
        self.assertEqual(fixtures, make_perm_fixtures())
        self.assertEqual(fixtures, make_palm_fixtures())

    def test_user_supplied_4x4_algorithms_are_distinct_low_effect_cases(self):
        before = random.getstate()
        fixtures = make_4x4_algorithm_fixtures()
        self.assertEqual(random.getstate(), before)
        self.assertEqual(
            [row['fixtureId'] for row in fixtures],
            ['wing-2-swap', 'wing-4-cycle', 'corner-2-swap',
             'wing-2-plus-corner-2-swap',
             'center-3-cycle-a', 'center-3-cycle-b'],
        )
        self.assertEqual([len(row['scramble']) for row in fixtures], [19, 15, 18, 37, 8, 8])
        self.assertEqual(fixtures[3]['scramble'], fixtures[0]['scramble'] + fixtures[2]['scramble'])
        self.assertEqual(
            [row['setupEffect']['movedPieceCount'] for row in fixtures], [2, 4, 2, 4, 3, 3]
        )
        self.assertEqual(
            [row['setupEffect']['reducedMoveCount'] for row in fixtures], [19, 15, 18, 37, 8, 8]
        )
        self.assertEqual(fixtures[3]['setupEffect']['componentCount'], 2)
        self.assertEqual(
            [row['initialStateHash'] for row in fixtures],
            ['ec2929098d478bda', 'e37c0092d84474b4', '56a84cc8217d6f55',
             '27badee768e25208',
             '11444ff63ac968ac', '05b13d7b6adff561'],
        )
        self.assertEqual(fixtures[3]['band'], 'low-effect/very-long-sequence')
        self.assertEqual(
            [row['band'] for row in fixtures[-2:]],
            ['low-effect/short-sequence', 'low-effect/short-sequence'],
        )
        self.assertEqual(fixtures, make_4x4_algorithm_fixtures())
        self.assertEqual(fixture_set_id(fixtures), fixture_set_id(make_4x4_algorithm_fixtures()))

    def test_4x4_profile_lists_algorithm_fixtures_by_default(self):
        output = io.StringIO()
        with patch(
            'tools.compare_ai_same_positions.build_experiment_frame_config',
            return_value=SimpleNamespace(cube_size=4),
        ), redirect_stdout(output):
            self.assertEqual(main(['--list-fixtures']), 0)
        listed = json.loads(output.getvalue())
        self.assertEqual(
            [row['fixtureId'] for row in listed['fixtures']],
            ['wing-2-swap', 'wing-4-cycle', 'corner-2-swap',
             'wing-2-plus-corner-2-swap',
             'center-3-cycle-a', 'center-3-cycle-b'],
        )

    def test_search2_follows_multiple_segments_and_restores_state(self):
        ai = _FakeAI([
            SearchResult(False, (" R'",), -2, [-2, -1], -1, [1, 10], end_reason='improvement'),
            SearchResult(True, (" U'",), -1, [-1, 0], 0, [1, 8], end_reason='solved'),
        ])
        ai.cube.scramble(2)
        original = ai.cube.state.copy()
        fixture = _fixture((' U ', ' R '), ai.cube)
        ai.cube.state[:] = original
        row = run_direct_solve_case(ai, fixture, record_route=True)
        self.assertTrue(row['directSolved'])
        self.assertEqual(row['searchCalls'], 2)
        self.assertEqual(row['evaluatedStates'], 18)
        self.assertEqual(row['moveCount'], 2)
        self.assertEqual(row['routeMoves'], [" R'", " U'"])
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

    def test_staged_search3_re_roots_without_changing_saved_cube(self):
        ai = _FakeAI([], mode='search3')
        results = iter([
            SearchResult(False, (" R'",), 0.2, [0.2, 0.4], 0.4,
                         [1, 7], search_mode='search3', end_reason='budget'),
            SearchResult(True, (" U'",), 0.4, [0.4, 1.0], 1.0,
                         [1, 5], search_mode='search3', end_reason='solved'),
        ])
        ai.search3 = lambda _limit: next(results)
        fixture = _fixture((' U ', ' R '), ai.cube)
        ai.cube.scramble(0, Move=(' F ',))
        original = ai.cube.state.copy()
        row = run_staged_search3_case(ai, fixture, (7, 7, 7))
        self.assertTrue(row['directSolved'])
        self.assertEqual(row['consumedPlayouts'], 12)
        self.assertEqual(row['routeMoves'], [" R'", " U'"])
        self.assertEqual(len(row['stages']), 2)
        np.testing.assert_array_equal(ai.cube.state, original)

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

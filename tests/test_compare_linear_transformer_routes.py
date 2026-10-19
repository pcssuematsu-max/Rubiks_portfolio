import hashlib
import unittest
from types import SimpleNamespace

import numpy as np

from ai.paired_solve_benchmark import run_direct_solve_case
from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config
from core.experiment_log import setup_effect_metrics
from cube.rubiks_cube import Rubiks_3
from model.search_result import SearchResult
from tools.compare_linear_transformer_routes import (
    common_prefix_moves, fixture_after_route, parse_pairs, probe_fixture_neighborhood,
    probe_route_value_shape, probe_witness_route, root_visit_evidence, sampled_step_indices,
    route_state_match_step, summarize_points,
)


def _fixture(scramble):
    cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
    cube.scramble(0, Move=scramble)
    return {
        'fixtureId': 'test', 'band': '0-7', 'scramble': list(scramble),
        'setupEffect': setup_effect_metrics('cube-3x3', 3, scramble),
        'initialStateHash': hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16],
    }


class _OneMoveAI:
    def __init__(self, move):
        self.cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
        self.search_mode = 'search2'
        self.move = move

    def search(self):
        return SearchResult(True, (self.move,), 0, [0, 1], 1, [1, 1], end_reason='solved')

    def clear_training_cache(self, collect=False):
        pass


class CompareLinearTransformerRoutesTests(unittest.TestCase):
    def test_pair_validation(self):
        config = build_experiment_frame_config()
        self.assertEqual(parse_pairs('20:22,20:22,21:23', config), [(20, 22), (21, 23)])
        for value in ('22:20', '20:10', '20:21', '20', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_pairs(value, config)

    def test_sampled_steps_keep_ends(self):
        self.assertEqual(sampled_step_indices(8, 3), [0, 4, 7])
        self.assertEqual(sampled_step_indices(3, 12), [0, 1, 2])
        self.assertEqual(sampled_step_indices(0, 12), [])
        self.assertEqual(common_prefix_moves((' R ', ' U '), (' R ', " U'")), 1)
        self.assertEqual(common_prefix_moves((' R ', ' U '), (' R ',)), 1)

    def test_route_recording_is_opt_in_and_does_not_change_cube(self):
        ai = _OneMoveAI(" R'")
        fixture = _fixture((' R ',))
        before = ai.cube.state.copy()
        recorded = run_direct_solve_case(
            ai, fixture, record_route=True, record_search_trace=True,
        )
        self.assertTrue(recorded['directSolved'])
        self.assertEqual(recorded['routeMoves'], [" R'"])
        self.assertEqual(recorded['searchTrace'][0]['acceptedMoves'], [" R'"])
        self.assertEqual(recorded['searchTrace'][0]['rootStateHash'], fixture['initialStateHash'])
        self.assertNotIn('routeMoves', run_direct_solve_case(ai, fixture))
        self.assertNotIn('searchTrace', run_direct_solve_case(ai, fixture))
        np.testing.assert_array_equal(ai.cube.state, before)

    def test_witness_probe_restores_state_and_rejects_non_solution(self):
        ai = Rubiks_3_AI([8], cube_size=3, search_mode='search2')
        ai.cube.scramble(2)
        before = ai.cube.state.copy()
        fixture = _fixture((' R ',))
        points = probe_witness_route(ai, fixture, (" R'",), max_points=2)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['witnessMove'], " R'")
        self.assertEqual(points[0]['remainingWitnessMoves'], 1)
        self.assertGreaterEqual(points[0]['valueRank'], 1)
        np.testing.assert_array_equal(ai.cube.state, before)
        with self.assertRaises(ValueError):
            probe_witness_route(ai, fixture, (' U ',))
        self.assertEqual(len(probe_witness_route(
            ai, fixture, (' U ',), require_goal=False,
        )), 1)
        np.testing.assert_array_equal(ai.cube.state, before)

    def test_terminal_fixture_and_rescue_probe_restore_cube(self):
        ai = Rubiks_3_AI([8], cube_size=3, search_mode='search2')
        ai.cube.scramble(2)
        before = ai.cube.state.copy()
        fixture = _fixture((' R ',))
        terminal = fixture_after_route(ai.cube, fixture, (' U ',))
        self.assertEqual(terminal['scramble'], [' R ', ' U '])
        probe = probe_fixture_neighborhood(ai, terminal, " U'")
        self.assertGreaterEqual(probe['policyRank'], 1)
        self.assertGreaterEqual(probe['valueRank'], 1)
        shape = probe_route_value_shape(ai, terminal, (" U'", " R'"))
        self.assertEqual(shape['moves'], 2)
        self.assertEqual(len(shape['valueDeltasFromStart']), 3)
        self.assertEqual(shape['valueDeltasFromStart'][0], 0)
        self.assertEqual(route_state_match_step(
            ai.cube, fixture, (' U ',), terminal['initialStateHash'],
        ), 1)
        np.testing.assert_array_equal(ai.cube.state, before)

    def test_visit_evidence_is_matched_to_terminal_root(self):
        trace = [
            {'rootStateHash': 'old', 'rootVisitCounts': {' R ': 10}},
            {'rootStateHash': 'terminal', 'rootVisitCounts': {' R ': 8, ' U ': 2}},
        ]
        self.assertEqual(root_visit_evidence(trace, 'terminal', ' U '), {
            'visits': 2, 'totalVisits': 10, 'visitShare': 0.2,
            'visitRank': 2, 'visitedActionCount': 2,
        })
        self.assertEqual(root_visit_evidence(trace, 'terminal', " F'" )['visits'], 0)
        self.assertIsNone(root_visit_evidence(trace, 'missing', ' U '))

    def test_summary_counts_policy_value_disagreement_and_checks_alignment(self):
        linear = [{'step': 0, 'stateHash': 'a', 'witnessMove': ' R ',
                   'remainingWitnessMoves': 11, 'policyRank': 2, 'valueRank': 1,
                   'valueDelta': 0.2}]
        transformer = [{'step': 0, 'stateHash': 'a', 'witnessMove': ' R ',
                        'remainingWitnessMoves': 11, 'policyRank': 1, 'valueRank': 20,
                        'valueDelta': -0.2, 'improvingSuccessorCount': 0}]
        summary = summarize_points(linear, transformer, 36)
        self.assertEqual(summary['overall']['transformerPolicyTop3ValueBottomHalf'], 1)
        self.assertEqual(summary['overall']['transformerPolicyTop3ValueBelowTop10'], 1)
        self.assertEqual(summary['overall']['transformerValueWorseThanLinear'], 1)
        self.assertEqual(summary['overall']['linearWitnessValueImproves'], 1)
        self.assertEqual(summary['byRemainingWitnessMoves']['early']['points'], 1)
        transformer[0]['stateHash'] = 'different'
        with self.assertRaises(ValueError):
            summarize_points(linear, transformer, 36)


if __name__ == '__main__':
    unittest.main()

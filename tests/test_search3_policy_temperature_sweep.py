import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from core.experiment_log import setup_effect_metrics
from cube.rubiks_cube import Rubiks_3
from tools.sweep_search3_policy_temperature import (
    collect_route_logits, load_witness_routes, parse_temperatures,
    policy_distribution_metrics, witness_policy_metrics,
)


class PolicyTemperatureSweepTests(unittest.TestCase):
    def test_temperatures_are_positive_finite_and_deduplicated(self):
        self.assertEqual(parse_temperatures('0.5,1,0.5'), [0.5, 1.0])
        for value in ('', '0', '-1', 'nan', 'inf', 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_temperatures(value)

    def test_temperature_changes_concentration_not_ranking(self):
        logits = np.asarray([4.0, 2.0, 0.0], dtype='f')
        sharp = policy_distribution_metrics(logits, 0.5)
        broad = policy_distribution_metrics(logits, 1.5)
        self.assertEqual(sharp['topActionIndex'], broad['topActionIndex'])
        self.assertLess(sharp['effectiveActions'], broad['effectiveActions'])
        self.assertGreater(sharp['top3Mass'], 0.99)
        rows = np.asarray([logits, logits], dtype='f')
        witness_sharp = witness_policy_metrics(rows, [0, 1], ['a', 'b', 'c'], 0.5)
        witness_broad = witness_policy_metrics(rows, [0, 1], ['a', 'b', 'c'], 1.5)
        self.assertEqual(witness_sharp['top3Count'], witness_broad['top3Count'])
        self.assertEqual(
            [item['rank'] for item in witness_sharp['bottlenecks']],
            [item['rank'] for item in witness_broad['bottlenecks']],
        )
        self.assertLess(
            witness_sharp['bottlenecks'][0]['probability'],
            witness_broad['bottlenecks'][0]['probability'],
        )

    def test_route_probe_preserves_cube_and_rejects_wrong_state(self):
        ai = Rubiks_3_AI([8], cube_size=3, search_mode='search3')
        fixture_cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
        fixture_cube.scramble(0, Move=(' R ',))
        fixture = {
            'fixtureId': 'test', 'scramble': [' R '],
            'initialStateHash': hashlib.sha256(
                ''.join(fixture_cube.state).encode('ascii'),
            ).hexdigest()[:16],
            'setupEffect': setup_effect_metrics('cube-3x3', 3, (' R ',)),
        }
        ai.cube.scramble(2)
        before = ai.cube.state.copy()
        logits, actions = collect_route_logits(ai, fixture, (" R'",))
        self.assertEqual(logits.shape, (1, len(ai.cube.move_keys)))
        self.assertEqual(actions, [ai.cube.key_to_num[" R'"]])
        np.testing.assert_array_equal(ai.cube.state, before)
        with self.assertRaises(ValueError):
            collect_route_logits(ai, fixture, (' U ',))
        np.testing.assert_array_equal(ai.cube.state, before)

    def test_report_routes_require_matching_state_hash(self):
        fixture = {'fixtureId': 'one', 'initialStateHash': 'abc'}
        report = {'pairs': [{
            'transformerIndex': 10,
            'cases': [{'fixtureId': 'one', 'initialStateHash': 'abc',
                       'linearDirectSolved': True, 'linearRouteMoves': [' R ']},
                      {'fixtureId': 'two', 'initialStateHash': 'other',
                       'linearDirectSolved': True, 'linearRouteMoves': [' U ']}],
        }]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'routes.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertEqual(load_witness_routes(path, 10, [fixture]), {'one': (' R ',)})
            self.assertEqual(load_witness_routes(path, 11, [fixture]), {})
            report['pairs'][0]['cases'][0]['initialStateHash'] = 'different'
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_witness_routes(path, 10, [fixture])


if __name__ == '__main__':
    unittest.main()

"""Sweep Search3 Policy temperature with C, search budget and weights fixed.

This is an offline, read-only comparison of saved checkpoints. It never learns,
uses greedy fallback, inserts replay, or saves parameters. Save GUI parameters
before running if the latest in-memory weights should be compared.

The optional route report is produced by compare_linear_transformer_routes.py.
Each Linear solution is one successful witness, not the unique correct path.
Temperature changes probabilities but cannot change the Policy action ranking.

Example:
  python3.14 tools/sweep_search3_policy_temperature.py \
      --ai-indices 10,11,18,19 --temperatures 0.5,0.75,1,1.25 \
      --route-report exports/linear-transformer-route-probe-EXAMPLE.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.paired_solve_benchmark import (
    fixture_set_id, make_paired_fixtures, make_perm_fixtures, run_direct_solve_case,
)
from configs.profiles import build_experiment_frame_config
from cube.search3_engine import softmax_1d
from tools.compare_ai_same_positions import _indices, _load_ai


def parse_temperatures(value):
    try:
        values = list(dict.fromkeys(float(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise ValueError('Temperatures must be comma-separated numbers') from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError('Every Policy temperature must be positive and finite')
    return values


def policy_distribution_metrics(logits, temperature):
    """Measure concentration on the exact probabilities Search3 uses."""
    probabilities = softmax_1d(logits, temperature=temperature).astype('d')
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('Non-finite Policy probabilities')
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
    order = np.argsort(-probabilities, kind='stable')
    return {
        'effectiveActions': float(np.exp(entropy)),
        'top1Mass': float(probabilities[order[0]]),
        'top3Mass': float(np.sum(probabilities[order[:3]])),
        'topActionIndex': int(order[0]),
    }


def collect_route_logits(ai, fixture, route):
    """Read logits along one independently successful route, restoring state."""
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError('Witness route starts from a different cube state')
        features = []
        action_indices = []
        for move in route:
            if move not in cube.key_to_num:
                raise ValueError(f'Unknown witness move: {move!r}')
            features.append(cube.makedata())
            action_indices.append(cube.key_to_num[move])
            cube.make_move(move)
        if not route or not cube.is_perfect():
            raise ValueError('Witness route does not reach the goal')
        logits = []
        for start in range(0, len(features), 64):
            batch = np.asarray(features[start:start + 64], dtype='f').T
            output = np.asarray(ai.predict(
                batch, policy=True, value=False, loss=False, retain_cache=False,
            ), dtype='f')
            logits.extend(output.T)
        logits = np.asarray(logits, dtype='f')
        if logits.shape != (len(route), len(cube.move_keys)) or not np.all(np.isfinite(logits)):
            raise ValueError('Invalid Policy output on witness route')
        return logits, action_indices
    finally:
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot


def witness_policy_metrics(logits, action_indices, moves, temperature):
    """Report probabilities and invariant ranks, without labeling alternatives wrong."""
    if len(logits) != len(action_indices) or not len(logits):
        raise ValueError('Witness logits and actions must be nonempty and aligned')
    steps = []
    for step, (row, action_index) in enumerate(zip(logits, action_indices)):
        probabilities = softmax_1d(row, temperature=temperature)
        probability = float(probabilities[action_index])
        rank = int(np.count_nonzero(row > row[action_index]) + 1)
        steps.append({
            'step': step, 'move': moves[action_index], 'rank': rank,
            'probability': probability,
        })
    bottlenecks = sorted(steps, key=lambda item: item['probability'])[:3]
    return {
        'routeLength': len(steps),
        'top3Count': sum(item['rank'] <= 3 for item in steps),
        'belowFivePercentCount': sum(item['probability'] < 0.05 for item in steps),
        'geometricMeanProbability': float(np.exp(np.mean([
            np.log(max(item['probability'], 1e-30)) for item in steps
        ]))),
        'bottlenecks': bottlenecks,
    }


def load_witness_routes(path, target_index, fixtures):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    by_id = {fixture['fixtureId']: fixture for fixture in fixtures}
    routes = {}
    for pair in report.get('pairs', ()):
        if pair.get('transformerIndex') != target_index:
            continue
        for case in pair.get('cases', ()):
            if not case.get('linearDirectSolved'):
                continue
            fixture = by_id.get(case.get('fixtureId'))
            if fixture is None:
                continue
            if fixture['initialStateHash'] != case.get('initialStateHash'):
                raise ValueError(f"Witness state differs for {fixture['fixtureId']}")
            routes[fixture['fixtureId']] = tuple(case['linearRouteMoves'])
    return routes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-indices', default='10,11,18,19')
    parser.add_argument('--temperatures', default='0.5,0.75,1,1.25')
    parser.add_argument('--c', type=float, default=0.5, help='Fixed C for every variant')
    parser.add_argument('--budget-mode', choices=('fixed', 'progressive'), default='fixed')
    parser.add_argument('--initial-width', type=int, default=5)
    parser.add_argument('--fixtures-per-band', type=int, default=2)
    parser.add_argument('--fixture-set', choices=('generated', 'perms', 'palms', 'all'), default='all')
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--max-search-calls', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--route-report', type=Path,
                        help='Optional successful Linear routes for per-step rank/probability checks')
    parser.add_argument('--output', type=Path,
                        help='JSONL output; default is a new timestamped file under exports/')
    args = parser.parse_args(argv)
    config = build_experiment_frame_config()
    try:
        indices = _indices(args.ai_indices, len(config.ai_search_modes))
        temperatures = parse_temperatures(args.temperatures)
        if config.cube_size != 3:
            raise ValueError('The fixed fixture set is defined only for 3x3')
        if any(config.ai_search_modes[index] != 'search3' for index in indices):
            raise ValueError('Every selected AI must use Search3')
        if not math.isfinite(args.c) or args.c <= 0:
            raise ValueError('C must be positive and finite')
        if args.initial_width < 1 or args.max_search_calls < 1 or args.max_moves < 1:
            raise ValueError('Width and search limits must be positive')
        fixtures = []
        if args.fixture_set in ('generated', 'all'):
            fixtures.extend(make_paired_fixtures(args.fixtures_per_band, args.seed))
        if args.fixture_set in ('perms', 'palms', 'all'):
            fixtures.extend(make_perm_fixtures())
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'exports' / f'search3-policy-temperature-{run_id}.jsonl'
    if output.exists():
        parser.error(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    set_id = fixture_set_id(fixtures)
    for index in indices:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        ai.search3_C = args.c
        ai.search3_C_depth_max = args.c
        ai.search3_budget_mode = args.budget_mode
        ai.search3_widening_initial_actions = args.initial_width
        ai.search3_budget_shallow_widening_options = (
            ai.search3_budget_shallow_widening_initial_actions,
        )
        routes = load_witness_routes(args.route_report, index, fixtures) if args.route_report else {}
        route_logits = {
            fixture['fixtureId']: collect_route_logits(ai, fixture, routes[fixture['fixtureId']])
            for fixture in fixtures if fixture['fixtureId'] in routes
        }
        for temperature in temperatures:
            ai.policy_temperature = temperature
            ai.search3_engine.reset_tree()
            for fixture in fixtures:
                cube = ai.cube
                cube.reset()
                cube.scramble(0, Move=tuple(fixture['scramble']))
                root_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
                if root_hash != fixture['initialStateHash']:
                    raise ValueError(f"Initial state differs for {fixture['fixtureId']}")
                root_output = np.asarray(ai.predict(
                    np.asarray(cube.makedata(), dtype='f').reshape(-1, 1),
                    policy=True, value=False, loss=False, retain_cache=False,
                ), dtype='f').reshape(-1)
                root_policy = policy_distribution_metrics(root_output, temperature)
                root_policy['topMove'] = cube.move_keys[root_policy.pop('topActionIndex')]
                case = run_direct_solve_case(
                    ai, fixture, max_search_calls=args.max_search_calls,
                    max_moves=args.max_moves,
                )
                route = route_logits.get(fixture['fixtureId'])
                record = {
                    'schemaVersion': 1, 'runId': run_id, 'fixtureSetId': set_id,
                    'aiIndex': index, 'checkpoint': checkpoint,
                    'fixtureId': fixture['fixtureId'],
                    'initialStateHash': fixture['initialStateHash'],
                    'settings': {
                        'policyTemperature': temperature, 'c': args.c,
                        'budgetMode': args.budget_mode,
                        'initialWidth': args.initial_width,
                        'searchNum3': ai.search_num3,
                        'searchRepeat3': ai.search_repeat3,
                    },
                    'rootPolicy': root_policy,
                    'witnessRoute': None if route is None else witness_policy_metrics(
                        route[0], route[1], cube.move_keys, temperature,
                    ),
                    'case': case,
                }
                with output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(
                        record, ensure_ascii=False, separators=(',', ':'), allow_nan=False,
                    ) + '\n')
                print(
                    f'AI{index} T={temperature:g} {fixture["fixtureId"]}: '
                    f'{"solved" if case["directSolved"] else case["terminalReason"]} '
                    f'({case["playouts"]} playouts)', flush=True,
                )
    print(f'Results: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

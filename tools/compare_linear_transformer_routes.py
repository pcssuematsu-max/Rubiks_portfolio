"""Compare Linear success routes against Transformer Policy/Value on the same states.

This is an offline diagnostic: no learning, replay insertion, fallback, or
parameter saving. Save GUI checkpoints before running if the latest in-memory
parameters matter. A Linear route is one witnessed solution, not the unique
correct route; other moves are not labeled as failures. Failed Transformer
states are also re-solved by Linear to measure multi-step Value valleys.

Examples:
  python3.14 tools/compare_linear_transformer_routes.py --pairs 20:22 --fixture-set perms
  python3.14 tools/compare_linear_transformer_routes.py --pairs 2:10,20:22,21:23 \
      --fixtures-per-band 1 --branches
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.paired_solve_benchmark import (
    fixture_set_id, make_paired_fixtures, make_perm_fixtures,
    probe_position_neighborhood, run_direct_solve_case,
)
from configs.profiles import build_experiment_frame_config
from core.experiment_log import setup_effect_metrics
from tools.compare_ai_same_positions import _load_ai, _settings


def _cube_snapshot(cube):
    return cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()


def _cube_restore(cube, snapshot):
    if hasattr(cube, '_restore'):
        cube._restore(snapshot)
    else:
        cube.state[:] = snapshot


def parse_pairs(value, config):
    """Require a Linear/Transformer pair with matching search/loss modes."""
    pairs = []
    seen = set()
    for part in value.split(','):
        fields = part.strip().split(':')
        try:
            if len(fields) != 2:
                raise ValueError
            linear_index, transformer_index = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError('Pairs must be LINEAR:TRANSFORMER indices') from exc
        if not (0 <= linear_index < len(config.ai_search_modes)
                and 0 <= transformer_index < len(config.ai_search_modes)):
            raise ValueError('Pair index is outside the configured AI range')
        if config.original_transformer_attention[linear_index]:
            raise ValueError(f'AI{linear_index} is not Linear')
        if not config.original_transformer_attention[transformer_index]:
            raise ValueError(f'AI{transformer_index} is not Transformer')
        if config.ai_search_modes[linear_index] != config.ai_search_modes[transformer_index]:
            raise ValueError('Paired AIs must use the same search mode')
        if config.search2_value_loss_types[linear_index] != config.search2_value_loss_types[transformer_index]:
            raise ValueError('Paired AIs must use the same Value loss type')
        pair = (linear_index, transformer_index)
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    if not pairs:
        raise ValueError('At least one pair is required')
    return pairs


def sampled_step_indices(route_length, max_points):
    """Spread probes over the route, always including its first and final moves."""
    if route_length < 1 or max_points < 1:
        return []
    if route_length <= max_points:
        return list(range(route_length))
    if max_points == 1:
        return [0]
    return sorted({int(round(i * (route_length - 1) / (max_points - 1)))
                   for i in range(max_points)})


def common_prefix_moves(first, second):
    return next((index for index, (a, b) in enumerate(zip(first, second)) if a != b),
                min(len(first), len(second)))


def route_state_match_step(cube, fixture, route, state_hash):
    """Return the first witnessed state equal to another route's endpoint."""
    snapshot = _cube_snapshot(cube)
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        for step in range(len(route) + 1):
            current_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
            if current_hash == state_hash:
                return step
            if step < len(route):
                cube.make_move(route[step])
        return None
    finally:
        _cube_restore(cube, snapshot)


def _branch_ranks(ai, reference_move):
    """First-layer ablation sensitivity; not an independent trained model."""
    cube = ai.cube
    if not getattr(ai, 'transformer_linear_skip', False):
        return None
    snapshot = _cube_snapshot(cube)
    moves = tuple(cube.move_keys)
    try:
        inputs = [cube.makedata()]
        for move in moves:
            _cube_restore(cube, snapshot)
            cube.make_move(move)
            inputs.append(cube.makedata())
        probe = ai.diagnose_first_layer_branches(np.asarray(inputs, dtype='f').T)
        if probe is None:
            return None
        result = {
            'attentionRms': probe['attentionRms'],
            'linearRms': probe['linearRms'],
            'linearToAttentionRms': probe['linearToAttentionRms'],
            'preactivationCosine': probe['preactivationCosine'],
        }
        move_index = moves.index(reference_move)
        for key, output_name in (('linearOnly', 'withoutAttention'),
                                 ('attentionOnly', 'withoutLinear')):
            output = np.asarray(probe[output_name])
            policy = output[:-1, 0]
            values = output[-1, 1:]
            if len(policy) != len(moves) or len(values) != len(moves):
                raise ValueError('Branch probe output shape does not match legal moves')
            if not (np.all(np.isfinite(policy)) and np.all(np.isfinite(values))):
                raise ValueError('Non-finite branch probe output')
            result[key] = {
                'policyRank': int(np.count_nonzero(policy > policy[move_index]) + 1),
                'valueRank': int(np.count_nonzero(values > values[move_index]) + 1),
            }
        return result
    finally:
        _cube_restore(cube, snapshot)


def probe_witness_route(ai, fixture, route, max_points=12, branches=False,
                        require_goal=True):
    """Score moves along one observed path; optionally require a solved endpoint."""
    cube = ai.cube
    snapshot = _cube_snapshot(cube)
    selected = set(sampled_step_indices(len(route), max_points))
    points = []
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError(f"Initial state differs for {fixture['fixtureId']}")
        for step, move in enumerate(route):
            if step in selected:
                neighborhood = probe_position_neighborhood(ai, move)
                point = {
                    'step': step,
                    'remainingWitnessMoves': len(route) - step,
                    'stateHash': hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16],
                    'witnessMove': move,
                    'policyRank': neighborhood['referencePolicyRank'],
                    'valueRank': neighborhood['referenceValueRank'],
                    'valueDelta': neighborhood['referenceValueDelta'],
                    'improvingSuccessorCount': neighborhood['improvingSuccessorCount'],
                    'topPolicyMoves': neighborhood['topPolicyMoves'],
                    'topValueMoves': neighborhood['topValueMoves'],
                }
                if branches:
                    point['branchSensitivity'] = _branch_ranks(ai, move)
                points.append(point)
            cube.make_move(move)
        if require_goal and not cube.is_perfect():
            raise ValueError('Recorded Linear route does not reach the goal')
        return points
    finally:
        _cube_restore(cube, snapshot)


def fixture_after_route(cube, fixture, route):
    """Rebuild the exact terminal state reached by a direct-search prefix."""
    snapshot = _cube_snapshot(cube)
    scramble = tuple(fixture['scramble']) + tuple(route)
    try:
        cube.reset()
        cube.scramble(0, Move=scramble)
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
    finally:
        _cube_restore(cube, snapshot)
    effect = setup_effect_metrics('cube-3x3', 3, scramble)
    if effect is None:
        raise ValueError('Could not calculate terminal setup Effect')
    return {
        'fixtureId': fixture['fixtureId'] + '-target-terminal',
        'band': fixture['band'],
        'scramble': list(scramble),
        'setupEffect': effect,
        'initialStateHash': state_hash,
    }


def probe_fixture_neighborhood(ai, fixture, reference_move):
    """Probe one fixed state while preserving the AI's current cube state."""
    cube = ai.cube
    snapshot = _cube_snapshot(cube)
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError('Probe fixture state differs from its recorded hash')
        probe = probe_position_neighborhood(ai, reference_move)
        return {
            'policyRank': probe['referencePolicyRank'],
            'valueRank': probe['referenceValueRank'],
            'valueDelta': probe['referenceValueDelta'],
            'improvingSuccessorCount': probe['improvingSuccessorCount'],
            'topPolicyMoves': probe['topPolicyMoves'],
            'topValueMoves': probe['topValueMoves'],
        }
    finally:
        _cube_restore(cube, snapshot)


def probe_route_value_shape(ai, fixture, route):
    """Measure whether a witnessed multi-step path crosses a Value valley."""
    cube = ai.cube
    snapshot = _cube_snapshot(cube)
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError('Value route starts from a different state')
        inputs = [cube.makedata()]
        for move in route:
            cube.make_move(move)
            inputs.append(cube.makedata())
        values = []
        for start in range(0, len(inputs), 64):
            batch = np.asarray(inputs[start:start + 64], dtype='f').T
            values.extend(np.asarray(ai.predict(
                batch, policy=False, value=True, loss=False, retain_cache=False,
            )).reshape(-1).tolist())
        if len(values) != len(inputs) or not np.all(np.isfinite(values)):
            raise ValueError('Non-finite or incomplete Value route trace')
        deltas = [float(value - values[0]) for value in values]
        skip_threshold = (
            float(ai.skip_difference) if ai.search_mode == 'search2' else None
        )
        return {
            'valueDomain': 'raw_logit' if ai.search_mode == 'search3' else 'raw',
            'moves': len(route),
            'valueDeltasFromStart': deltas,
            'firstPositiveGainStep': next(
                (step for step, delta in enumerate(deltas[1:], 1) if delta > 0), None
            ),
            'skipDifference': skip_threshold,
            'firstSkipDifferenceStep': None if skip_threshold is None else next(
                (step for step, delta in enumerate(deltas[1:], 1)
                 if delta >= skip_threshold), None
            ),
            'positiveAdjacentSteps': sum(
                values[step] > values[step - 1] for step in range(1, len(values))
            ),
            'maxGainByPrefix': {
                str(limit): max(deltas[:limit + 1])
                for limit in (5, 10, 20) if len(route) >= limit
            },
            'deepestInitialDip': min(deltas),
            'finalGain': deltas[-1],
        }
    finally:
        _cube_restore(cube, snapshot)


def root_visit_evidence(search_trace, state_hash, reference_move):
    """How often Search3 expanded one independently successful rescue move."""
    for trace in reversed(search_trace):
        if trace['rootStateHash'] != state_hash or 'rootVisitCounts' not in trace:
            continue
        visits = trace['rootVisitCounts']
        total = sum(visits.values())
        count = visits.get(reference_move, 0)
        return {
            'visits': count,
            'totalVisits': total,
            'visitShare': count / total if total else None,
            'visitRank': 1 + sum(value > count for value in visits.values()),
            'visitedActionCount': len(visits),
        }
    return None


def summarize_points(linear_points, transformer_points, action_count):
    """Aggregate conditional ranks, never compare raw Value scales across AIs."""
    if len(linear_points) != len(transformer_points):
        raise ValueError('Pair probes must contain the same steps')
    phases = {'early': [], 'middle': [], 'late': []}
    for linear, transformer in zip(linear_points, transformer_points):
        if (linear['step'], linear['stateHash'], linear['witnessMove']) != (
            transformer['step'], transformer['stateHash'], transformer['witnessMove']
        ):
            raise ValueError('Pair probes must refer to identical states and moves')
        phase = 'early' if linear['remainingWitnessMoves'] > 10 else (
            'middle' if linear['remainingWitnessMoves'] > 5 else 'late'
        )
        phases[phase].append((linear, transformer))

    def metrics(rows):
        total = len(rows)
        if not total:
            return {'points': 0}
        return {
            'points': total,
            'linearMeanPolicyRank': sum(a['policyRank'] for a, _ in rows) / total,
            'linearMeanValueRank': sum(a['valueRank'] for a, _ in rows) / total,
            'transformerMeanPolicyRank': sum(b['policyRank'] for _, b in rows) / total,
            'transformerMeanValueRank': sum(b['valueRank'] for _, b in rows) / total,
            'transformerPolicyTop3ValueBottomHalf': sum(
                b['policyRank'] <= 3 and b['valueRank'] > action_count / 2
                for _, b in rows
            ),
            'transformerPolicyTop3ValueBelowTop10': sum(
                b['policyRank'] <= 3 and b['valueRank'] > 10 for _, b in rows
            ),
            'transformerValueWorseThanLinear': sum(
                b['valueRank'] > a['valueRank'] for a, b in rows
            ),
            'linearWitnessValueImproves': sum(a['valueDelta'] > 0 for a, _ in rows),
            'transformerWitnessValueImproves': sum(b['valueDelta'] > 0 for _, b in rows),
            'transformerNoImprovingSuccessor': sum(
                b['improvingSuccessorCount'] == 0 for _, b in rows
            ),
        }

    all_rows = [row for rows in phases.values() for row in rows]
    return {'overall': metrics(all_rows), 'byRemainingWitnessMoves': {
        name: metrics(rows) for name, rows in phases.items()
    }}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', default='20:22', help='Matching Linear:Transformer indices')
    parser.add_argument('--fixture-set', choices=('generated', 'perms', 'palms', 'all'), default='all')
    parser.add_argument('--fixtures-per-band', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--max-probe-points', type=int, default=12)
    parser.add_argument('--max-search-calls', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--branches', action='store_true',
                        help='Include off-distribution first-layer branch-ablation sensitivity')
    parser.add_argument('--skip-target-search', action='store_true',
                        help='Only score the witnessed route; omit Transformer direct outcome')
    parser.add_argument('--skip-rescue-probe', action='store_true',
                        help='Do not retry Linear from a failed Transformer terminal state')
    parser.add_argument('--output', type=Path,
                        help='JSON output; default is a new timestamped file under exports/')
    args = parser.parse_args(argv)
    if min(args.max_probe_points, args.max_search_calls, args.max_moves) < 1:
        parser.error('Probe points and search limits must be positive')
    config = build_experiment_frame_config()
    if config.cube_size != 3:
        parser.error('This fixed fixture set is defined only for 3x3')
    try:
        pairs = parse_pairs(args.pairs, config)
        fixtures = []
        if args.fixture_set in ('generated', 'all'):
            fixtures.extend(make_paired_fixtures(args.fixtures_per_band, args.seed))
        if args.fixture_set in ('perms', 'palms', 'all'):
            fixtures.extend(make_perm_fixtures())
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'exports' / f'linear-transformer-route-probe-{run_id}.json'
    report = {
        'schemaVersion': 1,
        'runId': run_id,
        'fixtureSetId': fixture_set_id(fixtures),
        'limits': {'maxProbePoints': args.max_probe_points,
                   'maxSearchCalls': args.max_search_calls, 'maxMoves': args.max_moves,
                   'linearRescueProbe': not args.skip_rescue_probe},
        'interpretation': (
            'The Linear route is one successful witness, not a unique optimal path. '
            'Ranks are within each AI; raw Value scales must not be compared across AIs. '
            'A one-step Value decrease may still lie on a successful multi-step route. '
            'Branch ablation is a sensitivity check on a jointly trained network.'
        ),
        'pairs': [],
    }
    for linear_index, transformer_index in pairs:
        linear, linear_checkpoint = _load_ai(linear_index, config, args.checkpoint_root)
        transformer, transformer_checkpoint = _load_ai(transformer_index, config, args.checkpoint_root)
        pair_row = {
            'linearIndex': linear_index, 'transformerIndex': transformer_index,
            'linearCheckpoint': linear_checkpoint, 'transformerCheckpoint': transformer_checkpoint,
            'linearSettings': _settings(linear, config, linear_index),
            'transformerSettings': _settings(transformer, config, transformer_index),
            'cases': [],
        }
        for fixture in fixtures:
            reference = run_direct_solve_case(
                linear, fixture, args.max_search_calls, args.max_moves, record_route=True,
            )
            target = None if args.skip_target_search else run_direct_solve_case(
                transformer, fixture, args.max_search_calls, args.max_moves,
                terminal_probe=probe_position_neighborhood, record_route=True,
                record_search_trace=True,
            )
            case = {
                'fixtureId': fixture['fixtureId'], 'band': fixture['band'],
                'initialStateHash': fixture['initialStateHash'],
                'setupEffect': fixture['setupEffect'],
                'linearDirectSolved': reference['directSolved'],
                'linearTerminalReason': reference['terminalReason'],
                'transformerDirectSolved': None if target is None else target['directSolved'],
                'transformerTerminalReason': None if target is None else target['terminalReason'],
                'transformerTerminalStateHash': None if target is None else target['terminalStateHash'],
                'transformerSearchCalls': None if target is None else target['searchCalls'],
                'transformerPlayouts': None if target is None else target['playouts'],
                'transformerTerminalNeighborhood': None if target is None or not target.get(
                    'terminalNeighborhood'
                ) else {
                    key: target['terminalNeighborhood'][key] for key in (
                        'improvingSuccessorCount', 'bestSuccessorDelta',
                        'policyValueTop3Overlap', 'topPolicyMoves', 'topValueMoves',
                    )
                },
                'linearRouteMoves': reference['routeMoves'] if reference['directSolved'] else [],
                'routeLength': len(reference['routeMoves']) if reference['directSolved'] else None,
                'transformerRouteMoves': None if target is None else target['routeMoves'],
                'transformerSearchTrace': None if target is None else target['searchTrace'],
                'points': [],
                'transformerPathPoints': [],
            }
            if reference['directSolved']:
                route = reference['routeMoves']
                if target is not None:
                    shared = common_prefix_moves(route, target['routeMoves'])
                    case['sharedPrefixMoves'] = shared
                    case['firstDifferentMoveIndex'] = (
                        shared if shared < min(len(route), len(target['routeMoves'])) else None
                    )
                    case['transformerTerminalOnLinearRouteStep'] = route_state_match_step(
                        linear.cube, fixture, route, target['terminalStateHash'],
                    )
                linear_points = probe_witness_route(
                    linear, fixture, route, args.max_probe_points,
                )
                transformer_points = probe_witness_route(
                    transformer, fixture, route, args.max_probe_points, args.branches,
                )
                case['points'] = [
                    {'linear': a, 'transformer': b}
                    for a, b in zip(linear_points, transformer_points)
                ]
                case['summary'] = summarize_points(
                    linear_points, transformer_points, len(linear.cube.move_keys),
                )
            if target is not None and target['routeMoves']:
                route = target['routeMoves']
                linear_path_points = probe_witness_route(
                    linear, fixture, route, args.max_probe_points,
                    require_goal=False,
                )
                transformer_path_points = probe_witness_route(
                    transformer, fixture, route, args.max_probe_points,
                    require_goal=False,
                )
                case['transformerPathPoints'] = [
                    {'linear': a, 'transformer': b}
                    for a, b in zip(linear_path_points, transformer_path_points)
                ]
                case['transformerPathSummary'] = summarize_points(
                    linear_path_points, transformer_path_points, len(linear.cube.move_keys),
                )
            if target is not None and reference['directSolved'] and not target['directSolved'] \
                    and not args.skip_rescue_probe:
                terminal = fixture_after_route(linear.cube, fixture, target['routeMoves'])
                if terminal['initialStateHash'] != target['terminalStateHash']:
                    raise ValueError('Transformer terminal state does not match its route')
                rescue = run_direct_solve_case(
                    linear, terminal, args.max_search_calls, args.max_moves,
                    record_route=True,
                )
                rescue_row = {
                    'linearDirectSolved': rescue['directSolved'],
                    'linearTerminalReason': rescue['terminalReason'],
                    'linearRouteLength': len(rescue['routeMoves']) if rescue['directSolved'] else None,
                    'linearFirstMove': rescue['firstMove'] if rescue['directSolved'] else None,
                }
                if rescue['directSolved'] and rescue['firstMove'] is not None:
                    move = rescue['firstMove']
                    rescue_row['linearNeighborhood'] = probe_fixture_neighborhood(
                        linear, terminal, move,
                    )
                    rescue_row['transformerNeighborhood'] = probe_fixture_neighborhood(
                        transformer, terminal, move,
                    )
                    rescue_row['transformerRootVisits'] = root_visit_evidence(
                        target['searchTrace'], terminal['initialStateHash'], move,
                    )
                    terminal_trace = next((
                        trace for trace in reversed(target['searchTrace'])
                        if trace['rootStateHash'] == terminal['initialStateHash']
                    ), None)
                    rescue_row['transformerProposedRescueFirstMove'] = (
                        None if terminal_trace is None else
                        terminal_trace['proposedFirstMove'] == move
                    )
                    rescue_row['linearValueAlongRescue'] = probe_route_value_shape(
                        linear, terminal, rescue['routeMoves'],
                    )
                    rescue_row['transformerValueAlongRescue'] = probe_route_value_shape(
                        transformer, terminal, rescue['routeMoves'],
                    )
                case['linearRescueFromTransformerTerminal'] = rescue_row
            pair_row['cases'].append(case)
            print(f'AI{linear_index} → AI{transformer_index} {fixture["fixtureId"]}: '
                  f'Linear {"solved" if reference["directSolved"] else reference["terminalReason"]}, '
                  f'Transformer {"not run" if target is None else ("solved" if target["directSolved"] else target["terminalReason"])}; '
                  f'{len(case["points"])} route probes', flush=True)
        pair_row['summary'] = {
            'fixtures': len(pair_row['cases']),
            'linearDirectSolved': sum(case['linearDirectSolved'] for case in pair_row['cases']),
            'transformerDirectSolved': None if args.skip_target_search else sum(
                case['transformerDirectSolved'] for case in pair_row['cases']
            ),
            'pairedOutcome': {
                'bothSolved': None if args.skip_target_search else sum(
                    case['linearDirectSolved'] and case['transformerDirectSolved']
                    for case in pair_row['cases']
                ),
                'linearOnly': None if args.skip_target_search else sum(
                    case['linearDirectSolved'] and not case['transformerDirectSolved']
                    for case in pair_row['cases']
                ),
                'transformerOnly': None if args.skip_target_search else sum(
                    not case['linearDirectSolved'] and case['transformerDirectSolved']
                    for case in pair_row['cases']
                ),
            },
        }
        report['pairs'].append(pair_row)
        del linear, transformer
        gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(f'Results: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

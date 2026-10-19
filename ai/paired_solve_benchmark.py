"""Fixed, same-position direct-solve benchmark outside the 200-trial loop.

Fixtures are deterministic and are never inserted into replay.  This module
does not learn, save parameters, or use the GUI's greedy fallback.
"""

from __future__ import annotations

import hashlib
import json
import random
from time import perf_counter

import numpy as np

from core.experiment_log import setup_effect_metrics
from cube.rubiks_cube import Rubiks_3


FIXTURE_SEED = 20261017
BENCHMARK_VERSION = 1
EFFECT_BANDS = (
    ('0-7', 0, 7),
    ('8-13', 8, 13),
    ('14-19', 14, 19),
    ('20+', 20, None),
    ('14+ / reduced40+', 14, None),
)
PALM_SCRAMBLES = (
    ('T-palm', (
        'R', 'U', "R'", "U'", "R'", 'F', 'R2', "U'", "R'", "U'", 'R', 'U', "R'", "F'",
    )),
    ('Y-palm', (
        'F', 'R', "U'", "R'", "U'", 'R', 'U', "R'", "F'", 'R', 'U', "R'", "U'", "R'", 'F', 'R', "F'",
    )),
)


def _outer_moves(cube):
    return tuple(
        move for move in cube.move_keys
        if move.startswith(' ') and move.strip()[:1] in {'R', 'L', 'U', 'D', 'F', 'B'}
    )


def _random_moves(rng, candidates, count):
    moves = []
    previous_face = None
    for _ in range(count):
        choices = [move for move in candidates if move.strip()[:1] != previous_face]
        move = rng.choice(choices)
        moves.append(move)
        previous_face = move.strip()[:1]
    return tuple(moves)


def _candidate_scramble(cube, rng, candidates, band_index):
    if band_index in (0, 1):
        # Paired commutators can have many moves but a small net piece effect.
        commutators = []
        for _ in range(2):
            a = _random_moves(rng, candidates, 1)
            b = _random_moves(rng, candidates, 1)
            commutators.extend(a + b + tuple(cube.invert_moves(a)) + tuple(cube.invert_moves(b)))
        return tuple(commutators)
    if band_index == 2:
        return _random_moves(rng, candidates, rng.choice((8, 10, 12)))
    if band_index == 3:
        return _random_moves(rng, candidates, rng.choice((10, 12, 16)))
    return _random_moves(rng, candidates, 48)


def make_paired_fixtures(per_band=2, seed=FIXTURE_SEED):
    """Create balanced, reproducible 3x3 setups with no training side effects."""
    per_band = int(per_band)
    if not 1 <= per_band <= 10:
        raise ValueError('per_band must be in 1..10')
    cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
    candidates = _outer_moves(cube)
    fixtures = []
    seen_states = set()
    for band_index, (band, minimum, maximum) in enumerate(EFFECT_BANDS):
        rng = random.Random(int(seed) + 1009 * band_index)
        accepted = 0
        for _ in range(10000):
            scramble = _candidate_scramble(cube, rng, candidates, band_index)
            effect = setup_effect_metrics('cube-3x3', 3, scramble)
            if effect is None:
                continue
            moved = effect['movedPieceCount']
            reduced = effect['reducedMoveCount']
            if moved < minimum or (maximum is not None and moved > maximum):
                continue
            if reduced < (40 if band_index == 4 else 8):
                continue
            cube.reset()
            cube.scramble(0, Move=scramble)
            if cube.is_perfect():
                continue
            state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
            if state_hash in seen_states:
                continue
            seen_states.add(state_hash)
            accepted += 1
            fixtures.append({
                'fixtureId': f'{band_index + 1}-{accepted}',
                'band': band,
                'scramble': list(scramble),
                'setupEffect': effect,
                'initialStateHash': state_hash,
            })
            if accepted == per_band:
                break
        if accepted < per_band:
            raise RuntimeError(f'Could not construct enough fixtures for {band}')
    cube.reset()
    return fixtures


def make_palm_fixtures():
    """Return the two user-supplied low-Effect, unreduced long algorithms."""
    cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
    display_to_key = {str(key).strip(): key for key in cube.move_keys}
    fixtures = []
    seen_states = set()
    for fixture_id, notation in PALM_SCRAMBLES:
        scramble = tuple(display_to_key[move] for move in notation)
        effect = setup_effect_metrics('cube-3x3', 3, scramble)
        if effect is None or effect['movedPieceCount'] != 4:
            raise ValueError(f'{fixture_id} must have Effect 4')
        if effect['reducedMoveCount'] != len(scramble):
            raise ValueError(f'{fixture_id} unexpectedly reduces to a shorter route')
        cube.reset()
        cube.scramble(0, Move=scramble)
        if cube.is_perfect():
            raise ValueError(f'{fixture_id} unexpectedly produces a solved state')
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash in seen_states:
            raise ValueError('Palm fixtures must have distinct start states')
        seen_states.add(state_hash)
        fixtures.append({
            'fixtureId': fixture_id,
            'band': '0-7',
            'scramble': list(scramble),
            'setupEffect': effect,
            'initialStateHash': state_hash,
        })
    cube.reset()
    return fixtures


def fixture_set_id(fixtures):
    encoded = json.dumps(
        [(case['fixtureId'], case['scramble'], case['initialStateHash']) for case in fixtures],
        ensure_ascii=False, separators=(',', ':'),
    ).encode('utf-8')
    return f'paired-direct-v{BENCHMARK_VERSION}-' + hashlib.sha256(encoded).hexdigest()[:16]


def _finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def probe_position_neighborhood(ai, reference_move=None):
    """Read-only Policy/Value ranks for every legal successor of one state.

    The reference move is only a reproducible inverse-scramble comparison,
    not a claim that it is the unique or shortest good move. Raw Value scales
    differ by loss type; compare ranks within an AI, not scores across AIs.
    """
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    moves = tuple(cube.move_keys)
    try:
        current = np.asarray(cube.makedata(),dtype='f').reshape(-1,1)
        output = np.asarray(ai.predict(
            current,policy=True,value=True,loss=False,retain_cache=False,
        ),dtype='f')
        policy = output[:-1,0]
        current_value = float(output[-1,0])
        columns = []
        for move in moves:
            if hasattr(cube,'_restore'):
                cube._restore(snapshot)
            else:
                cube.state[:] = snapshot
            cube.make_move(move)
            columns.append(cube.makedata())
        values = []
        for start in range(0,len(columns),64):
            batch = np.asarray(columns[start:start + 64],dtype='f').T
            values.extend(np.asarray(ai.predict(
                batch,policy=False,value=True,loss=False,retain_cache=False,
            ),dtype='f').reshape(-1).tolist())
        values = np.asarray(values,dtype='f')
        if not (np.all(np.isfinite(policy)) and np.isfinite(current_value) and np.all(np.isfinite(values))):
            raise ValueError('Non-finite Policy/Value in fixed-position neighborhood probe')
        policy_order = np.argsort(-policy,kind='stable')
        value_order = np.argsort(-values,kind='stable')
        result = {
            'valueDomain': 'raw_logit' if ai.search_mode == 'search3' else 'raw',
            'currentValue': current_value,
            'goalValue': _finite_float(getattr(ai,'perfect_val',None)),
            'improvingSuccessorCount': int(np.count_nonzero(values > current_value)),
            'bestSuccessorDelta': float(np.max(values) - current_value),
            'topPolicyMoves': [moves[int(index)] for index in policy_order[:3]],
            'topValueMoves': [moves[int(index)] for index in value_order[:3]],
            'policyValueTop3Overlap': len(set(policy_order[:3]) & set(value_order[:3])),
            # Optional offline output only; it lets a successful peer's
            # actual first move be evaluated by every other AI afterward.
            'policyRankByMove': {
                move: int(np.count_nonzero(policy > policy[index]) + 1)
                for index,move in enumerate(moves)
            },
            'valueRankByMove': {
                move: int(np.count_nonzero(values > values[index]) + 1)
                for index,move in enumerate(moves)
            },
        }
        if reference_move in moves:
            index = moves.index(reference_move)
            result.update({
                'referenceMove': reference_move,
                'referencePolicyRank': int(np.count_nonzero(policy > policy[index]) + 1),
                'referenceValueRank': int(np.count_nonzero(values > values[index]) + 1),
                'referenceValueDelta': float(values[index] - current_value),
            })
        return result
    finally:
        if hasattr(cube,'_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot


def run_direct_solve_case(ai, fixture, max_search_calls=200, max_moves=300,
                          terminal_probe=None, record_route=False,
                          record_search_trace=False):
    """Run one AI from a fixture's exact state until direct solve or fallback.

    Search2 may make several improving segments. Search3 stops when the GUI
    would hand a budget result to greedy fallback; greedy is not run here.
    The cube and Search3 tree are restored even if search raises.
    """
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    result_row = {
        'fixtureId': fixture['fixtureId'],
        'band': fixture['band'],
        'scramble': list(fixture['scramble']),
        'setupEffect': dict(fixture['setupEffect']),
        'initialStateHash': fixture['initialStateHash'],
        'directSolved': False,
        'terminalReason': None,
        'lastSearchEndReason': None,
        'searchCalls': 0,
        'moveCount': 0,
        'firstMove': None,
        'evaluatedStates': 0,
        'playouts': 0,
        'treeNodePeak': 0,
        'playoutDepthMean': None,
        'playoutDepthMax': 0,
        'deep20Rate': None,
        'terminalRootValue': None,
        'terminalBestValue': None,
        'terminalStateHash': None,
        'elapsedSeconds': 0.0,
    }
    if record_route:
        result_row['routeMoves'] = []
    if record_search_trace:
        result_row['searchTrace'] = []
    depth_weighted = 0.0
    depth_count = 0
    deep20_weighted = 0.0
    deep20_count = 0
    started = perf_counter()
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError(f"Initial state differs for {fixture['fixtureId']}")
        if ai.search_mode == 'search3':
            ai.search3_engine.reset_tree()
        for _ in range(max_search_calls):
            before = cube.state.copy()
            result = ai.search()
            result_row['searchCalls'] += 1
            if not np.array_equal(cube.state, before):
                raise RuntimeError('Search changed the cube state')
            result_row['terminalReason'] = str(result.end_reason)
            result_row['lastSearchEndReason'] = str(result.end_reason)
            result_row['terminalRootValue'] = _finite_float(result.root_value)
            result_row['terminalBestValue'] = _finite_float(result.best_value)
            trace = None
            if record_search_trace:
                trace = {
                    'call': result_row['searchCalls'],
                    'rootStateHash': hashlib.sha256(
                        ''.join(before).encode('ascii')
                    ).hexdigest()[:16],
                    'moveOffset': result_row['moveCount'],
                    'endReason': str(result.end_reason),
                    'proposedFirstMove': result.moves[0] if result.moves else None,
                    'proposedMoveCount': len(result.moves),
                    'acceptedMoves': [],
                    'rootValue': _finite_float(result.root_value),
                    'bestValue': _finite_float(result.best_value),
                    'lastAttemptWorkCount': max(0, int(result.stats[1])),
                }
                if ai.search_mode == 'search3':
                    diagnostics = result.search_diagnostics or {}
                    budget = diagnostics.get('budgetSummary') or {}
                    trace['budget'] = {
                        'mode': budget.get('mode'),
                        'stopReason': budget.get('stopReason'),
                        'consumedPlayoutCount': budget.get('consumedPlayoutCount'),
                        'stages': [
                            {key: stage.get(key) for key in (
                                'playoutLimit', 'consumedPlayoutCount',
                                'rootVisitShare', 'bestImprovement',
                                'playoutDepthMean', 'decision',
                            )}
                            for stage in budget.get('stages', ())
                        ],
                    }
                    trace['playoutDepthMean'] = _finite_float(diagnostics.get('playoutDepthMean'))
                    trace['playoutDepthMax'] = int(diagnostics.get('playoutDepthMax') or 0)
                    root = getattr(ai.search3_engine, 'root_node', None)
                    if root is not None:
                        visits = np.asarray(root.visited).reshape(-1)
                        if len(visits) == len(cube.move_keys):
                            priors = np.asarray(root.P).reshape(-1)
                            backed_values = np.asarray(root.val).reshape(-1)
                            trace['rootVisitCounts'] = {
                                move: int(visits[index])
                                for index, move in enumerate(cube.move_keys)
                                if visits[index] > 0
                            }
                            trace['rootPolicyProbabilities'] = {
                                move: float(priors[index])
                                for index, move in enumerate(cube.move_keys)
                            }
                            trace['rootMeanBackedValues'] = {
                                move: float(backed_values[index] / visits[index])
                                for index, move in enumerate(cube.move_keys)
                                if visits[index] > 0
                            }
                result_row['searchTrace'].append(trace)
            if ai.search_mode == 'search3':
                attempts = list(result.attempt_results or (result,))
                for attempt in attempts:
                    count = max(0, int(attempt.stats[1]))
                    diagnostics = attempt.search_diagnostics or {}
                    result_row['playouts'] += count
                    result_row['treeNodePeak'] = max(
                        result_row['treeNodePeak'],
                        int(diagnostics.get('treeNodeCount') or 0),
                    )
                    result_row['playoutDepthMax'] = max(
                        result_row['playoutDepthMax'],
                        int(diagnostics.get('playoutDepthMax') or 0),
                    )
                    depth = _finite_float(diagnostics.get('playoutDepthMean'))
                    if depth is not None:
                        depth_weighted += depth * count
                        depth_count += count
                    deep20 = _finite_float(diagnostics.get('playoutDepthAtLeast20Rate'))
                    if deep20 is not None:
                        deep20_weighted += deep20 * count
                        deep20_count += count
                if result.end_reason in ('budget', 'square1-fallback'):
                    result_row['terminalReason'] = 'greedy_fallback_required'
                    break
            else:
                result_row['evaluatedStates'] += max(0, int(result.stats[1]))

            moves = tuple(cube.reduce(result.moves)[0])
            if not moves:
                result_row['terminalReason'] = 'no_progress'
                break
            if result_row['moveCount'] + len(moves) > max_moves:
                result_row['terminalReason'] = 'move_limit'
                break
            if result_row['firstMove'] is None:
                result_row['firstMove'] = moves[0]
            for move in moves:
                cube.make_move(move)
                if record_route:
                    result_row['routeMoves'].append(move)
                if ai.search_mode == 'search3':
                    ai.search3_engine.advance_root(move)
            if trace is not None:
                trace['acceptedMoves'] = list(moves)
            result_row['moveCount'] += len(moves)
            if cube.is_perfect():
                result_row['directSolved'] = True
                result_row['terminalReason'] = 'solved'
                break
            if result.succeeded:
                result_row['terminalReason'] = 'inconsistent_success_result'
                break
            if hasattr(ai, 'clear_training_cache'):
                ai.clear_training_cache(collect=False)
        else:
            result_row['terminalReason'] = 'search_call_limit'
        if not result_row['directSolved'] and terminal_probe is not None:
            result_row['terminalNeighborhood'] = terminal_probe(ai)
    finally:
        result_row['terminalStateHash'] = hashlib.sha256(
            ''.join(cube.state).encode('ascii')
        ).hexdigest()[:16]
        result_row['elapsedSeconds'] = round(perf_counter() - started, 6)
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
        if ai.search_mode == 'search3':
            ai.search3_engine.reset_tree()
        if hasattr(ai, 'clear_training_cache'):
            ai.clear_training_cache(collect=False)
    if depth_count:
        result_row['playoutDepthMean'] = depth_weighted / depth_count
    if deep20_count:
        result_row['deep20Rate'] = deep20_weighted / deep20_count
    return result_row


def summarize_direct_cases(cases):
    """Return compact per-band success and search-work counts."""
    by_band = {}
    for band, _, _ in EFFECT_BANDS:
        selected = [case for case in cases if case['band'] == band]
        if not selected:
            continue
        by_band[band] = {
            'solved': sum(bool(case['directSolved']) for case in selected),
            'trials': len(selected),
            'meanEvaluatedStates': sum(case['evaluatedStates'] for case in selected) / len(selected),
            'meanPlayouts': sum(case['playouts'] for case in selected) / len(selected),
            'meanSearchCalls': sum(case['searchCalls'] for case in selected) / len(selected),
        }
    return {
        'solved': sum(bool(case['directSolved']) for case in cases),
        'trials': len(cases),
        'byBand': by_band,
    }

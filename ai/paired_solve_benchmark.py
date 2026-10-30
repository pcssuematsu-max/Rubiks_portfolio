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
PERM_SCRAMBLES = (
    ('T-perm', (
        'R', 'U', "R'", "U'", "R'", 'F', 'R2', "U'", "R'", "U'", 'R', 'U', "R'", "F'",
    )),
    ('Y-perm', (
        'F', 'R', "U'", "R'", "U'", 'R', 'U', "R'", "F'", 'R', 'U', "R'", "U'", "R'", 'F', 'R', "F'",
    )),
)
# Keep the old import name for external scripts; fixture IDs use "Perm".
PALM_SCRAMBLES = PERM_SCRAMBLES
_WING_2_SWAP_4X4 = (
    '2U2', " F'", "2R'", ' F ', ' R2', " F'", '2R ', " F'", "2U'",
    ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2',
    '2U ', ' R2',
)
_CORNER_2_SWAP_4X4 = (
    ' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2',
    " B'", ' L ', ' B ', '2F2', ' R2', ' U2', '2F2', ' U2',
    ' R2', '2F2',
)
ALGORITHM_4X4_SCRAMBLES = (
    ('wing-2-swap', 'Wing 2 Swap', 'low-effect/long-sequence', _WING_2_SWAP_4X4),
    ('wing-4-cycle', 'Wing 4 Cycle', 'low-effect/long-sequence', (
        '2R2', ' F2', '2R ', ' U2', '2R ', ' U2', ' F2', '2R ',
        ' F2', "2R'", ' F2', '2R ', ' F2', '2R2', ' F2',
    )),
    ('corner-2-swap', 'Corner 2 Swap', 'low-effect/long-sequence', _CORNER_2_SWAP_4X4),
    ('wing-2-plus-corner-2-swap', 'Wing 2 + Corner 2 Swap',
     'low-effect/very-long-sequence', _WING_2_SWAP_4X4 + _CORNER_2_SWAP_4X4),
    ('center-3-cycle-a', 'Center 3 Cycle A', 'low-effect/short-sequence', (
        '2R ', ' U ', "2L'", " U'", "2R'", ' U ', '2L ', " U'",
    )),
    ('center-3-cycle-b', 'Center 3 Cycle B', 'low-effect/short-sequence', (
        '2R2', ' U ', '2L2', " U'", '2R2', ' U ', '2L2', " U'",
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


def make_perm_fixtures():
    """Return the two user-supplied low-Effect, unreduced long algorithms."""
    cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
    display_to_key = {str(key).strip(): key for key in cube.move_keys}
    fixtures = []
    seen_states = set()
    for fixture_id, notation in PERM_SCRAMBLES:
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
            raise ValueError('Perm fixtures must have distinct start states')
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


def make_palm_fixtures():
    """Backward-compatible alias for the correctly named Perm fixtures."""
    return make_perm_fixtures()


def make_4x4_algorithm_fixtures():
    """Return the user-supplied low-Effect 4x4 wing/corner/center algorithms."""
    cube = Rubiks_3(size=4, RegisterMyperms=False, PointReindex=False)
    legal_moves = set(cube.move_keys)
    fixtures = []
    seen_states = set()
    for fixture_id, name, band, scramble in ALGORITHM_4X4_SCRAMBLES:
        if any(move not in legal_moves for move in scramble):
            raise ValueError(f'{fixture_id} contains a move unavailable on 4x4')
        effect = setup_effect_metrics('cube-4x4', 4, scramble)
        if effect is None:
            raise ValueError(f'{fixture_id} has no measurable Effect')
        cube.reset()
        cube.scramble(0, Move=scramble)
        if cube.is_perfect():
            raise ValueError(f'{fixture_id} unexpectedly produces a solved state')
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash in seen_states:
            raise ValueError(f'{fixture_id} duplicates another 4x4 fixture')
        seen_states.add(state_hash)
        fixtures.append({
            'fixtureId': fixture_id,
            'algorithmName': name,
            'band': band,
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


def probe_position_neighborhood(ai, reference_move=None, include_successors=False):
    """Read-only Policy/Value ranks for every successor of one state.

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
        legal = []
        for move in moves:
            if hasattr(cube,'_restore'):
                cube._restore(snapshot)
            else:
                cube.state[:] = snapshot
            legal.append(not hasattr(cube, 'is_legal_move') or bool(cube.is_legal_move(move)))
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
        legal_mask = np.asarray(legal, dtype=bool)
        legal_indices = np.flatnonzero(legal_mask)
        legal_policy_order = sorted(legal_indices, key=lambda index: -float(policy[index]))
        legal_value_order = sorted(legal_indices, key=lambda index: -float(values[index]))
        result = {
            'valueDomain': 'raw_logit' if ai.search_mode == 'search3' else 'raw',
            'currentValue': current_value,
            'goalValue': _finite_float(getattr(ai,'perfect_val',None)),
            'improvingSuccessorCount': int(np.count_nonzero(values > current_value)),
            'bestSuccessorDelta': float(np.max(values) - current_value),
            'legalSuccessorCount': int(np.count_nonzero(legal_mask)),
            'legalImprovingSuccessorCount': int(np.count_nonzero(
                (values > current_value) & legal_mask,
            )),
            'bestLegalSuccessorDelta': (
                float(np.max(values[legal_mask]) - current_value)
                if legal_indices.size else None
            ),
            'topLegalPolicyMoves': [moves[int(index)] for index in legal_policy_order[:3]],
            'topLegalValueMoves': [moves[int(index)] for index in legal_value_order[:3]],
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
        if include_successors:
            root_policy_probabilities = None
            root_enqueue_threshold = None
            if ai.search_mode == 'search2' and legal_indices.size:
                masked_logits = policy.astype('d', copy=True)
                masked_logits[~legal_mask] = -np.inf
                shifted = masked_logits - np.max(masked_logits[legal_mask])
                exponentials = np.exp(shifted)
                exponentials[~legal_mask] = 0.0
                root_policy_probabilities = exponentials / np.sum(exponentials)
                budget = max(1, int(getattr(ai, 'search_num2', 10000)))
                # Search2's initial heap priority is -2*N, and it enqueues
                # a root child only if softmax(policy)*2*N >= 1.
                root_enqueue_threshold = 1.0 / (2 * budget)
                result['search2RootEnqueueProbabilityThreshold'] = root_enqueue_threshold
            result['successors'] = [
                {
                    'move': move,
                    'isLegal': bool(legal[index]),
                    'policyScore': float(policy[index]),
                    'policyProbability': (
                        float(root_policy_probabilities[index])
                        if root_policy_probabilities is not None else None
                    ),
                    'search2RootEnqueueEligible': (
                        bool(root_policy_probabilities[index] >= root_enqueue_threshold)
                        if root_policy_probabilities is not None and legal[index] else False
                    ) if ai.search_mode == 'search2' else None,
                    'policyRank': result['policyRankByMove'][move],
                    'value': float(values[index]),
                    'valueDelta': float(values[index] - current_value),
                    'valueRank': result['valueRankByMove'][move],
                }
                for index, move in enumerate(moves)
            ]
            if ai.search_mode == 'search2':
                result['search2RootEligibleImprovingSuccessorCount'] = sum(
                    successor['isLegal']
                    and successor['valueDelta'] > 0
                    and successor['search2RootEnqueueEligible']
                    for successor in result['successors']
                )
        if reference_move in moves:
            index = moves.index(reference_move)
            result.update({
                'referenceMove': reference_move,
                'referencePolicyRank': int(np.count_nonzero(policy > policy[index]) + 1),
                'referencePolicyScore': float(policy[index]),
                'referenceValueRank': int(np.count_nonzero(values > values[index]) + 1),
                'referenceSuccessorValue': float(values[index]),
                'referenceValueDelta': float(values[index] - current_value),
                'referenceValueGapToBest': float(np.max(values) - values[index]),
            })
        return result
    finally:
        if hasattr(cube,'_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot


def probe_two_step_escape(ai, root_probe=None, top_each=3):
    """Check whether likely first moves can climb above a stopped Search2 root.

    This deliberately inspects only the union of top Policy and top Value
    legal first moves. It is a bounded diagnostic, not an exhaustive proof
    that no two-step escape exists.
    """
    if not 1 <= top_each <= 3:
        raise ValueError('top_each must be in 1..3')
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    root = root_probe or probe_position_neighborhood(ai, include_successors=True)
    first_moves = list(dict.fromkeys(
        root['topLegalPolicyMoves'][:top_each]
        + root['topLegalValueMoves'][:top_each]
    ))
    rows = []
    try:
        for first_move in first_moves:
            if hasattr(cube, '_restore'):
                cube._restore(snapshot)
            else:
                cube.state[:] = snapshot
            cube.make_move(first_move)
            next_probe = probe_position_neighborhood(ai, include_successors=True)
            inverse = cube.invert_str(first_move)
            candidates = [
                successor for successor in next_probe['successors']
                if successor['isLegal'] and successor['move'] != inverse
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda successor: successor['value'])
            first = next(
                successor for successor in root['successors']
                if successor['move'] == first_move
            )
            rows.append({
                'firstMove': first_move,
                'firstPolicyRank': first['policyRank'],
                'firstPolicyProbability': first['policyProbability'],
                'firstRootEnqueueEligible': first['search2RootEnqueueEligible'],
                'firstValueDelta': first['valueDelta'],
                'bestSecondMove': best['move'],
                'bestSecondValueDeltaFromRoot': best['value'] - root['currentValue'],
            })
    finally:
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
    return {
        'sampledFirstMoveCount': len(rows),
        'sampledFirstMoves': first_moves,
        'bestTwoStepDeltaFromRoot': max(
            (row['bestSecondValueDeltaFromRoot'] for row in rows), default=None,
        ),
        'candidates': rows,
    }


def _probe_search3_root_for_reference_move(ai, move, playouts, trace_root=False):
    """Measure PUCT allocation at one route state without advancing the cube."""
    engine = ai.search3_engine
    engine.reset_tree()
    try:
        if trace_root:
            result = engine.search3(int(playouts), C=ai.search3_C, trace_root_move=move)
        else:
            result = ai.search3(int(playouts))
        root = engine.root_node
        if root is None:
            return None
        index = ai.cube.key_to_num[move]
        visits = np.asarray(root.visited, dtype='f').reshape(-1)
        priors = np.asarray(root.P, dtype='f').reshape(-1)
        backed = np.asarray(root.val, dtype='f').reshape(-1)
        total = float(np.sum(visits))
        top_policy_index = int(np.argmax(priors))
        top_visit_index = int(np.argmax(visits)) if total else None
        move_keys = ai.cube.move_keys
        diagnostics = result.search_diagnostics or {}
        probe = {
            'playouts': int(result.stats[1]),
            'endReason': str(result.end_reason),
            'referenceVisits': int(visits[index]),
            'referenceVisitShare': float(visits[index] / total) if total else 0.0,
            'referencePrior': float(priors[index]),
            'referenceBackedMean': (
                float(backed[index] / visits[index]) if visits[index] else None
            ),
            'topVisitShare': float(np.max(visits) / total) if total else 0.0,
            'topPolicyMove': move_keys[top_policy_index],
            'topPolicyPrior': float(priors[top_policy_index]),
            'topPolicyVisitShare': float(visits[top_policy_index] / total) if total else 0.0,
            'topVisitedMove': move_keys[top_visit_index] if top_visit_index is not None else None,
            'topVisitedPrior': float(priors[top_visit_index]) if top_visit_index is not None else None,
            'topVisitedBackedMean': (
                float(backed[top_visit_index] / visits[top_visit_index])
                if top_visit_index is not None and visits[top_visit_index] else None
            ),
            'playoutDepthMean': _finite_float(diagnostics.get('playoutDepthMean')),
            'playoutDepthMax': int(diagnostics.get('playoutDepthMax') or 0),
        }
        if trace_root:
            probe['rootActionTrace'] = result.root_action_trace
        return probe
    finally:
        engine.reset_tree()


def probe_reference_route(ai, fixture, route_moves, max_probe_steps=None,
                          search3_probe_playouts=0, trace_search3_root=False,
                          include_successors=False):
    """Evaluate every decision on one solved reference route without learning.

    The route is a comparison path, not a claim that its moves are uniquely
    optimal.  Only compact per-step metrics are retained; the full successor
    ranking maps from probe_position_neighborhood are discarded.
    """
    route = tuple(route_moves)
    if not route:
        raise ValueError('A reference route must contain at least one move')
    if max_probe_steps is not None and int(max_probe_steps) < 1:
        raise ValueError('max_probe_steps must be positive')
    if int(search3_probe_playouts) < 0:
        raise ValueError('search3_probe_playouts must be nonnegative')
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    steps = []
    route_start_value = None
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError(f"Initial state differs for {fixture['fixtureId']}")
        for step_index, move in enumerate(route, start=1):
            if move not in cube.move_keys:
                raise ValueError(f'Invalid reference move at step {step_index}: {move!r}')
            if max_probe_steps is None or step_index <= max_probe_steps:
                probe = probe_position_neighborhood(
                    ai, move, include_successors=include_successors,
                )
                if route_start_value is None:
                    route_start_value = probe['currentValue']
                step = {
                    'step': step_index,
                    'remainingMoveCount': len(route) - step_index + 1,
                    'stateHash': state_hash,
                    'referenceMove': move,
                    'policyRank': probe['referencePolicyRank'],
                    'policyScore': probe['referencePolicyScore'],
                    'valueRank': probe['referenceValueRank'],
                    'currentValue': probe['currentValue'],
                    'successorValue': probe['referenceSuccessorValue'],
                    'valueDelta': probe['referenceValueDelta'],
                    'valueFromRouteStart': probe['referenceSuccessorValue'] - route_start_value,
                    'valueGapToBest': probe['referenceValueGapToBest'],
                    'improvingSuccessorCount': probe['improvingSuccessorCount'],
                    'topPolicyMoves': probe['topPolicyMoves'],
                    'topValueMoves': probe['topValueMoves'],
                }
                if include_successors:
                    step['successors'] = probe['successors']
                if ai.search_mode == 'search3' and search3_probe_playouts:
                    step['search3Root'] = _probe_search3_root_for_reference_move(
                        ai, move, search3_probe_playouts,
                        trace_root=trace_search3_root,
                    )
                steps.append(step)
            cube.make_move(move)
            state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if not cube.is_perfect():
            raise ValueError(f"Reference route does not solve {fixture['fixtureId']}")
        def first_step(condition):
            return next((step['step'] for step in steps if condition(step)), None)

        route_probe = {
            'fixtureId': fixture['fixtureId'],
            'initialStateHash': fixture['initialStateHash'],
            'routeMoveCount': len(route),
            'probedMoveCount': len(steps),
            'valueDomain': 'raw_logit' if ai.search_mode == 'search3' else 'raw',
            'routeStartValue': route_start_value,
            'summary': {
                'policyTop3Count': sum(step['policyRank'] <= 3 for step in steps),
                'valueTop3Count': sum(step['valueRank'] <= 3 for step in steps),
                'bothTop3Count': sum(
                    step['policyRank'] <= 3 and step['valueRank'] <= 3 for step in steps
                ),
                'firstPolicyOutsideTop3Step': first_step(lambda step: step['policyRank'] > 3),
                'firstValueOutsideTop3Step': first_step(lambda step: step['valueRank'] > 3),
                'firstBothOutsideTop3Step': first_step(
                    lambda step: step['policyRank'] > 3 and step['valueRank'] > 3
                ),
                'firstValueDropStep': first_step(lambda step: step['valueDelta'] < 0.0),
                'firstValueAboveStartStep': first_step(
                    lambda step: step['valueFromRouteStart'] > 0.0
                ),
                'minValueFromStart': min(step['valueFromRouteStart'] for step in steps),
                'maxValueFromStart': max(step['valueFromRouteStart'] for step in steps),
            },
            'steps': steps,
        }
        if trace_search3_root and ai.search_mode == 'search3' and search3_probe_playouts:
            traces = [step['search3Root']['rootActionTrace'] for step in steps]
            route_probe['summary'].update({
                'rootNeverEligibleCount': sum(
                    trace['firstEligiblePlayout'] is None for trace in traces
                ),
                'rootEligibleUnvisitedCount': sum(
                    trace['firstEligiblePlayout'] is not None
                    and trace['firstVisitPlayout'] is None for trace in traces
                ),
                'rootVisitedCount': sum(
                    trace['firstVisitPlayout'] is not None for trace in traces
                ),
            })
        return route_probe
    finally:
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
        if hasattr(ai, 'clear_training_cache'):
            ai.clear_training_cache(collect=False)


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


def run_staged_search3_case(ai, fixture, stage_playouts, max_commit_moves=8):
    """Offline multi-root Search3 probe with a fixed *total* playout budget.

    On a budget result, commit a bounded principal variation and search again
    from its endpoint. This is an inference experiment only: it never updates
    parameters or inserts the route into replay. Compare it with one root
    search using the sum of ``stage_playouts``.
    """
    if ai.search_mode != 'search3':
        raise ValueError('Staged Search3 requires a Search3 AI')
    limits = tuple(int(value) for value in stage_playouts)
    if not limits or any(value < 1 for value in limits) or max_commit_moves < 1:
        raise ValueError('Stages and max_commit_moves must be positive')
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    row = {
        'fixtureId': fixture['fixtureId'],
        'initialStateHash': fixture['initialStateHash'],
        'stagePlayouts': list(limits),
        'maxCommitMoves': int(max_commit_moves),
        'directSolved': False,
        'terminalReason': None,
        'consumedPlayouts': 0,
        'routeMoves': [],
        'stages': [],
    }
    started = perf_counter()
    try:
        cube.reset()
        cube.scramble(0, Move=tuple(fixture['scramble']))
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if state_hash != fixture['initialStateHash']:
            raise ValueError(f"Initial state differs for {fixture['fixtureId']}")
        seen_states = {state_hash}
        ai.search3_engine.reset_tree()
        for limit in limits:
            result = ai.search3(limit)
            used = max(0, int(result.stats[1]))
            row['consumedPlayouts'] += used
            proposed = tuple(result.moves)
            accepted = proposed if result.succeeded else proposed[:max_commit_moves]
            stage = {
                'rootStateHash': state_hash,
                'requestedPlayouts': limit,
                'consumedPlayouts': used,
                'endReason': str(result.end_reason),
                'rootValue': _finite_float(result.root_value),
                'bestValue': _finite_float(result.best_value),
                'proposedMoveCount': len(proposed),
                'acceptedMoves': [],
                'playoutDepthMax': int((result.search_diagnostics or {}).get('playoutDepthMax') or 0),
            }
            row['stages'].append(stage)
            for move in accepted:
                if hasattr(cube, 'is_legal_move') and not cube.is_legal_move(move):
                    row['terminalReason'] = 'illegal_proposal'
                    break
                cube.make_move(move)
                ai.search3_engine.advance_root(move)
                state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
                stage['acceptedMoves'].append(move)
                row['routeMoves'].append(move)
                if cube.is_perfect():
                    row['directSolved'] = True
                    row['terminalReason'] = 'solved'
                    break
                if state_hash in seen_states:
                    row['terminalReason'] = 'revisited_state'
                    break
                seen_states.add(state_hash)
            if row['terminalReason'] is not None:
                break
            if not stage['acceptedMoves']:
                row['terminalReason'] = 'no_progress'
                break
        if row['terminalReason'] is None:
            row['terminalReason'] = 'total_budget_exhausted'
        row['terminalStateHash'] = state_hash
    finally:
        row['elapsedSeconds'] = round(perf_counter() - started, 6)
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
        ai.search3_engine.reset_tree()
        if hasattr(ai, 'clear_training_cache'):
            ai.clear_training_cache(collect=False)
    return row


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

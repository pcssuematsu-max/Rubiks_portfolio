"""Read-only, matched-root Search3 comparisons for one loaded checkpoint."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from core.experiment_log import setup_effect_metrics


DEFAULT_VARIANTS = (
    ('fixed_all', 'fixed', 0, False),
    ('fixed_width5', 'fixed', 5, False),
    ('progressive_width5', 'progressive', 5, False),
    ('progressive_width3', 'progressive', 3, False),
    ('progressive_root_all', 'progressive', 5, True),
)


def compare_search3_variants(ai, scrambles, variants = DEFAULT_VARIANTS):
    """Run each search setting from identical states and unchanged weights.

    A row describes the *initial root search*, not a complete multi-step
    solve.  The canonical inverse first move is only one valid teacher; its
    visit share is useful for paired comparisons, not an accuracy claim.
    """
    if getattr(ai, 'search_mode', None) != 'search3':
        raise ValueError('Search3 paired probe requires a Search3 AI')
    cube = ai.cube
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    setting_names = (
        'search3_budget_mode', 'search3_widening_initial_actions',
        'search3_budget_shallow_widening_initial_actions',
        'search3_budget_shallow_widening_options',
        'search3_widening_root_all_actions',
        'search3_widening_mode', 'search3_widening_policy_mass',
        'search3_widening_min_actions', 'search3_widening_max_actions',
    )
    settings = {
        name: getattr(ai, name)
        for name in setting_names if hasattr(ai, name)
    }
    rows = []
    try:
        for scramble in scrambles:
            scramble = tuple(scramble)
            if not scramble:
                continue
            inverse = tuple(cube.invert_moves(scramble))
            target_move = inverse[0]
            effect = setup_effect_metrics('cube-3x3',getattr(ai,'cube_size',3),scramble)
            for variant in variants:
                label, mode, width, root_all = variant[:4]
                widening = variant[4] if len(variant) > 4 else {}
                cube.reset()
                cube.scramble(0,scramble)
                ai.search3_engine.reset_tree()
                ai.search3_budget_mode = mode
                ai.search3_widening_initial_actions = int(width)
                ai.search3_budget_shallow_widening_initial_actions = int(width)
                ai.search3_budget_shallow_widening_options = ()
                ai.search3_widening_root_all_actions = bool(root_all)
                ai.search3_widening_mode = widening.get('mode', 'fixed')
                ai.search3_widening_policy_mass = float(widening.get('mass', 0.8))
                ai.search3_widening_min_actions = int(widening.get('minimum', 3))
                ai.search3_widening_max_actions = int(widening.get('maximum', 6))
                started = perf_counter()
                result = ai.search()
                elapsed = perf_counter() - started
                root = ai.search3_engine.root_node
                visits = np.asarray(root.visited,dtype = 'f') if root is not None else np.zeros(len(cube.move_keys),dtype = 'f')
                policy = np.asarray(root.P,dtype = 'f') if root is not None else np.zeros_like(visits)
                target_index = cube.key_to_num[target_move]
                total_visits = float(np.sum(visits))
                attempts = list(getattr(result,'attempt_results',()) or (result,))
                playouts = [max(0,int(attempt.stats[1])) for attempt in attempts]
                weighted_depth = sum(
                    playouts[index] * float((attempt.search_diagnostics or {}).get('playoutDepthMean',0.0))
                    for index, attempt in enumerate(attempts)
                ) / max(1,sum(playouts))
                max_depth = max(
                    (int((attempt.search_diagnostics or {}).get('playoutDepthMax',0)) for attempt in attempts),
                    default = 0,
                )
                weighted_width = sum(
                    playouts[index] * float((attempt.search_diagnostics or {}).get('wideningActionLimitMean',0.0))
                    for index, attempt in enumerate(attempts)
                ) / max(1,sum(playouts))
                rows.append({
                    'variant': label,
                    'mode': mode,
                    'initialWidth': int(width),
                    'wideningMode': ai.search3_widening_mode,
                    'wideningActionLimitMean': weighted_width,
                    'rootAllActions': bool(root_all),
                    'search3C': float(ai.search3_C),
                    'scramble': list(scramble),
                    'effectMovedPieceCount': None if effect is None else effect['movedPieceCount'],
                    'endReason': result.end_reason,
                    'solvedAtRoot': result.end_reason == 'solved',
                    'consumedPlayouts': sum(playouts),
                    'playoutDepthMean': weighted_depth,
                    'playoutDepthMax': max_depth,
                    'rootTopVisitShare': None if not total_visits else float(np.max(visits) / total_visits),
                    'rootTop3PolicyMass': float(np.sum(np.sort(policy)[-3:])),
                    'inverseFirstMovePolicyRank': int(np.count_nonzero(policy > policy[target_index]) + 1),
                    'inverseFirstMoveVisitShare': None if not total_visits else float(visits[target_index] / total_visits),
                    'elapsedSeconds': elapsed,
                })
    finally:
        for name, value in settings.items():
            setattr(ai,name,value)
        for name in setting_names:
            if name not in settings and hasattr(ai,name):
                delattr(ai,name)
        if hasattr(cube,'_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
        ai.search3_engine.reset_tree()
    return rows

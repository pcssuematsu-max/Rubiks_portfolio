"""Small, reproducible 4x4 myperm combinations for training and held-out tests.

The names resolve through the cube's source myperms2 registry.  Training and
held-out selections are disjoint by physical start-state hash, including the
existing user-supplied Wing 2 + Corner 2 fixture.
"""

from __future__ import annotations

import hashlib

from core.experiment_log import setup_effect_metrics
from cube.rubiks_cube import Rubiks_3


# The old user-supplied 37-move fixture stays held out.
EXISTING_COMPOSITE_STATE_HASH = '27badee768e25208'

HELD_OUT_SPECS = (
    ('myperm-wing-corner-a', ('W2-2s[FL@U<>UB@L]', 'C2[DFR>LUF]~v01'), 'low-effect/very-long-sequence'),
    ('myperm-wing-corner-b', ('W2-2s[RF@U<>UR@B]', 'C2[UFL>FLU;URF>FUR]'), 'low-effect/very-long-sequence'),
    ('myperm-wing-corner-c', ('W2-2s[FL@U<>RF@U]~v01', 'C2[DLF>LUF]~v03'), 'low-effect/very-long-sequence'),
    ('myperm-high-effect-a', ('C8[3x2]+W2-24p[3x8]',), 'high-effect/short-sequence'),
    ('myperm-high-effect-b', ('CtrX16p[4x4]+W2-4s[FL@D<>RF@D;FL@U<>RF@U]',), 'high-effect/short-sequence'),
)

TRAINING_SPECS = (
    ('W2-2s[BR@D<>FL@U]~v01', 'C2[DBL>RFU]'),
    ('W2-2s[RF@D<>RF@U]~v01', 'C2[DRB>BDR;UFL>FLU]'),
    ('W2-2s[FL@D<>RF@U]~v01', 'C2[UFL>RFU]~v01'),
    ('W2-2s[FL@U<>UB@R]', 'C2[DLF>FLU]~v01'),
    ('W2-2s[BR@U<>FL@U]~v01', 'C2[DFR>RFU]~v01'),
    ('W2-24p[3x8]',),
)
TRAINING_TRANSFORMS = (0, 6, 12, 18, 24, 30, 36, 42)


def _state_hash(cube, moves):
    cube.reset()
    cube.scramble(0, Move=tuple(moves))
    return hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]


def _source_moves(cube, names):
    sources = getattr(cube, 'myperms2', None)
    if not isinstance(sources, dict):
        raise ValueError('The 4x4 myperm source registry is unavailable')
    missing = [name for name in names if name not in sources]
    if missing:
        raise ValueError(f'Unavailable 4x4 source myperm names: {missing}')
    moves = tuple(move for name in names for move in sources[name])
    if any(move not in cube.move_keys for move in moves):
        raise ValueError(f'Unavailable 4x4 move in source myperms: {names}')
    return moves


def _measure(cube, moves):
    effect = setup_effect_metrics('cube-4x4', 4, moves)
    if effect is None:
        raise ValueError('Myperm composition has no measurable effect')
    return effect


def _unreindexed_source_cube(cube):
    """Resolve legacy myperm names before point/effect renaming at app startup."""
    if cube is not None and cube.size != 4:
        raise ValueError('Myperm fixtures require a 4x4 cube')
    if cube is None or getattr(cube, 'PointReindex', False):
        return Rubiks_3(size=4, PointReindex=False)
    return cube


def make_4x4_myperm_fixtures(cube=None):
    """Return distinct fixed cases from selected source myperms."""
    cube = _unreindexed_source_cube(cube)
    if cube.size != 4:
        raise ValueError('Myperm fixtures require a 4x4 cube')
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    cases = []
    seen = {EXISTING_COMPOSITE_STATE_HASH}
    try:
        for fixture_id, names, band in HELD_OUT_SPECS:
            moves = _source_moves(cube, names)
            effect = _measure(cube, moves)
            moved = effect['movedPieceCount']
            reduced = effect['reducedMoveCount']
            if band.startswith('low-effect') and not (moved <= 8 and reduced >= 30):
                raise ValueError(f'{fixture_id} is no longer a low-Effect long route')
            if band.startswith('high-effect') and not (moved >= 20 and len(moves) <= 18):
                raise ValueError(f'{fixture_id} is no longer a high-Effect short route')
            state_hash = _state_hash(cube, moves)
            if cube.is_perfect() or state_hash in seen:
                raise ValueError(f'{fixture_id} duplicates a held-out state')
            seen.add(state_hash)
            cases.append({
                'fixtureId': fixture_id,
                'algorithmName': ' + '.join(names),
                'sourceMyperms': list(names),
                'band': band,
                'scramble': list(moves),
                'setupEffect': effect,
                'initialStateHash': state_hash,
            })
    finally:
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
    return cases


def make_4x4_myperm_training_scrambles(cube, forbidden_state_hashes=()):
    """Build diverse, unique auxiliary examples outside the 200-solve loop."""
    if cube.size != 4:
        return []
    cube = _unreindexed_source_cube(cube)
    snapshot = cube._snapshot() if hasattr(cube, '_snapshot') else cube.state.copy()
    seen = set(forbidden_state_hashes) | {EXISTING_COMPOSITE_STATE_HASH}
    scrambles = []
    try:
        for names in TRAINING_SPECS:
            source = _source_moves(cube, names)
            for transform_index in TRAINING_TRANSFORMS:
                moves = tuple(cube.transform(source, transform_index))
                state_hash = _state_hash(cube, moves)
                if cube.is_perfect() or state_hash in seen:
                    continue
                seen.add(state_hash)
                scrambles.append(moves)
    finally:
        if hasattr(cube, '_restore'):
            cube._restore(snapshot)
        else:
            cube.state[:] = snapshot
    return scrambles

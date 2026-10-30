"""Read-only 4x4 probes for staged Search3 and relative-Value plateaus.

All output goes to an ignored log folder. Generated scrambles are never added
to replay, and the saved AI checkpoints are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.myperm_composites import make_4x4_myperm_fixtures
from ai.paired_solve_benchmark import (
    make_4x4_algorithm_fixtures, probe_position_neighborhood,
    probe_reference_route, probe_two_step_escape, run_direct_solve_case,
    run_staged_search3_case,
)
from configs.profiles import build_experiment_frame_config
from core.experiment_log import setup_effect_metrics
from cube.rubiks_cube import Rubiks_3
from tools.compare_ai_same_positions import _indices, _load_ai, _settings


def _write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def _selected_fixtures(value):
    fixtures = make_4x4_algorithm_fixtures() + make_4x4_myperm_fixtures()
    by_id = {fixture['fixtureId']: fixture for fixture in fixtures}
    ids = [part.strip() for part in value.split(',') if part.strip()]
    if not ids or any(fixture_id not in by_id for fixture_id in ids):
        raise ValueError(f'Unknown fixture ID; available: {", ".join(by_id)}')
    return [by_id[fixture_id] for fixture_id in dict.fromkeys(ids)]


def _regular_scrambles(seed, candidates, length, cube):
    """Generate fresh random setups, distinct from the named fixed fixtures."""
    rng = random.Random(seed)
    move_keys = tuple(cube.move_keys)
    seen = set()
    for attempt in range(candidates):
        moves = []
        previous = None
        for _ in range(length):
            choices = [move for move in move_keys if move[:2] != previous]
            chosen = rng.choice(choices)
            moves.append(chosen)
            previous = chosen[:2]
        scramble = tuple(moves)
        cube.reset()
        cube.scramble(0, Move=scramble)
        state_hash = hashlib.sha256(''.join(cube.state).encode('ascii')).hexdigest()[:16]
        if cube.is_perfect() or state_hash in seen:
            continue
        seen.add(state_hash)
        yield {
            'fixtureId': f'regular-seed{seed}-attempt{attempt + 1}',
            'band': 'generated_regular',
            'scramble': list(scramble),
            'setupEffect': setup_effect_metrics('cube-4x4', 4, scramble),
            'initialStateHash': state_hash,
        }


def _study_staged(args, config, run_id):
    fixtures = _selected_fixtures(args.fixture_ids)
    indices = _indices(args.staged_ai_indices, len(config.ai_search_modes))
    if any(config.ai_search_modes[index] != 'search3' for index in indices):
        raise ValueError('Staged indices must all be Search3')
    stages = tuple(int(part) for part in args.stage_playouts.split(','))
    if not stages or any(value < 1 for value in stages):
        raise ValueError('Stage playouts must be positive')
    for index in indices:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        for fixture in fixtures:
            for variant, limits in (
                ('one_root', (sum(stages),)),
                ('staged_reroot', stages),
            ):
                case = run_staged_search3_case(
                    ai, fixture, limits, max_commit_moves=args.max_commit_moves,
                )
                _write_row(args.output, {
                    'recordType': 'staged_search3', 'runId': run_id,
                    'variant': variant, 'aiIndex': index, 'checkpoint': checkpoint,
                    'settings': _settings(ai, config, index), 'case': case,
                })
                print(
                    f'AI{index:02d} {fixture["fixtureId"]} {variant}: '
                    f'{case["terminalReason"]}, {len(case["routeMoves"])} moves, '
                    f'{case["consumedPlayouts"]} playouts', flush=True,
                )


def _study_routes(args, config, run_id):
    reference_ai, reference_checkpoint = _load_ai(
        args.reference_ai_index, config, args.checkpoint_root,
    )
    if (reference_ai.search_mode != 'search2'
            or reference_ai.search2_value_loss_type != 'myloss'):
        raise ValueError('Reference must be Original Search2')
    fixtures_and_routes = []
    for fixture in _regular_scrambles(
        args.regular_seed, args.regular_candidates,
        args.regular_length, reference_ai.cube,
    ):
        case = run_direct_solve_case(reference_ai, fixture, record_route=True)
        if case['directSolved']:
            fixtures_and_routes.append((fixture, case['routeMoves']))
            print(
                f'Reference AI{args.reference_ai_index:02d} solved '
                f'{fixture["fixtureId"]} in {case["moveCount"]} moves', flush=True,
            )
        if len(fixtures_and_routes) >= args.regular_cases:
            break
    if not fixtures_and_routes:
        raise RuntimeError('No regular scramble was solved by the reference AI')
    _write_row(args.output, {
        'recordType': 'regular_reference_set', 'runId': run_id,
        'referenceAiIndex': args.reference_ai_index,
        'referenceCheckpoint': reference_checkpoint,
        'cases': [
            {'fixture': fixture, 'routeMoves': moves}
            for fixture, moves in fixtures_and_routes
        ],
    })
    indices = _indices(args.route_ai_indices, len(config.ai_search_modes))
    for index in indices:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        for fixture, route in fixtures_and_routes:
            probe = probe_reference_route(
                ai, fixture, route, max_probe_steps=args.max_probe_steps,
                include_successors=True,
            )
            _write_row(args.output, {
                'recordType': 'regular_route_siblings', 'runId': run_id,
                'aiIndex': index, 'checkpoint': checkpoint,
                'settings': _settings(ai, config, index),
                'referenceAiIndex': args.reference_ai_index,
                'referenceCheckpoint': reference_checkpoint,
                'probe': probe,
            })
            summary = probe['summary']
            print(
                f'AI{index:02d} {fixture["fixtureId"]}: '
                f'Policy top3 {summary["policyTop3Count"]}/'
                f'{probe["probedMoveCount"]}, Value top3 '
                f'{summary["valueTop3Count"]}/{probe["probedMoveCount"]}',
                flush=True,
            )


def _study_stops(args, config, run_id):
    fixtures = _selected_fixtures(args.fixture_ids)
    indices = _indices(args.stop_ai_indices, len(config.ai_search_modes))
    if any(config.ai_search_modes[index] != 'search2' for index in indices):
        raise ValueError('Stop indices must all be Search2')
    def terminal_probe(model):
        neighborhood = probe_position_neighborhood(model, include_successors=True)
        neighborhood['twoStepEscape'] = probe_two_step_escape(model, neighborhood)
        return neighborhood
    for index in indices:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        for fixture in fixtures:
            case = run_direct_solve_case(
                ai, fixture, max_search_calls=args.stop_max_calls,
                max_moves=args.stop_max_moves, record_route=True,
                record_search_trace=True,
                terminal_probe=terminal_probe,
            )
            _write_row(args.output, {
                'recordType': 'search2_stop_siblings', 'runId': run_id,
                'aiIndex': index, 'checkpoint': checkpoint,
                'settings': _settings(ai, config, index), 'case': case,
            })
            neighborhood = case.get('terminalNeighborhood') or {}
            print(
                f'AI{index:02d} {fixture["fixtureId"]}: '
                f'{case["terminalReason"]}, moves={case["moveCount"]}, '
                f'improving successors={neighborhood.get("improvingSuccessorCount")}',
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('staged', 'routes', 'stops'), required=True)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=(
        ROOT / 'AI_logs_storage' / 'Rubiks_4x4' / 'benchmarks' /
        'offline-search-improvements.jsonl'
    ))
    parser.add_argument('--fixture-ids', default='wing-2-plus-corner-2-swap')
    parser.add_argument('--staged-ai-indices', default='3,10')
    parser.add_argument('--stage-playouts', default='2500,2500,2500,2500')
    parser.add_argument('--max-commit-moves', type=int, default=8)
    parser.add_argument('--reference-ai-index', type=int, default=0)
    parser.add_argument('--route-ai-indices', default='3,10,20,21,22,23')
    parser.add_argument('--regular-seed', type=int, default=20261029)
    parser.add_argument('--regular-candidates', type=int, default=12)
    parser.add_argument('--regular-cases', type=int, default=2)
    parser.add_argument('--regular-length', type=int, default=12)
    parser.add_argument('--max-probe-steps', type=int, default=4)
    parser.add_argument('--stop-ai-indices', default='20,21,22,23')
    parser.add_argument('--stop-max-calls', type=int, default=40)
    parser.add_argument('--stop-max-moves', type=int, default=120)
    args = parser.parse_args(argv)
    if (args.max_commit_moves < 1 or args.regular_candidates < 1
            or args.regular_cases < 1 or args.regular_length < 1
            or args.max_probe_steps < 1 or args.stop_max_calls < 1
            or args.stop_max_moves < 1):
        parser.error('Counts and limits must be positive')
    config = build_experiment_frame_config()
    if config.cube_size != 4:
        parser.error('This probe requires the 4x4 experiment profile')
    run_id = datetime.now(timezone.utc).isoformat(timespec='seconds')
    try:
        if args.study == 'staged':
            _study_staged(args, config, run_id)
        elif args.study == 'routes':
            _study_routes(args, config, run_id)
        else:
            _study_stops(args, config, run_id)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

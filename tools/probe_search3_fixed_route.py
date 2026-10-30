"""Trace one known solution route on fixed 4x4 setups without training.

The inverse of each setup is a valid reference route, not necessarily the
shortest or the only useful route.  Probe only its first few decisions so the
cost stays bounded even for long composite algorithms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.myperm_composites import make_4x4_myperm_fixtures
from ai.paired_solve_benchmark import make_4x4_algorithm_fixtures, probe_reference_route
from configs.profiles import build_experiment_frame_config
from tools.compare_ai_same_positions import _indices, _load_ai, _settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-indices', default='3,10',
                        help='Comma-separated Search3 AI indices')
    parser.add_argument('--fixture-ids', default='wing-2-plus-corner-2-swap',
                        help='Comma-separated fixed 4x4 fixture IDs, or all')
    parser.add_argument('--playouts', type=int, default=10000,
                        help='Root playouts per probed step')
    parser.add_argument('--max-probe-steps', type=int, default=1,
                        help='How many initial moves of the valid inverse route to probe')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=(
        ROOT / 'AI_logs_storage' / 'Rubiks_4x4' / 'benchmarks' /
        'search3-fixed-route-prior-visits.jsonl'
    ))
    args = parser.parse_args(argv)
    if args.playouts < 1 or args.max_probe_steps < 1:
        parser.error('playouts and max-probe-steps must be positive')
    config = build_experiment_frame_config()
    if config.cube_size != 4:
        parser.error('The fixed route probe requires the 4x4 experiment profile')
    try:
        indices = _indices(args.ai_indices, len(config.ai_search_modes))
    except ValueError as error:
        parser.error(str(error))
    if any(config.ai_search_modes[index] != 'search3' for index in indices):
        parser.error('Every selected AI must use Search3')
    fixtures = make_4x4_algorithm_fixtures() + make_4x4_myperm_fixtures()
    if args.fixture_ids.strip().lower() != 'all':
        requested = [part.strip() for part in args.fixture_ids.split(',') if part.strip()]
        by_id = {fixture['fixtureId']: fixture for fixture in fixtures}
        if not requested or any(fixture_id not in by_id for fixture_id in requested):
            parser.error('Unknown or empty fixture ID list')
        fixtures = [by_id[fixture_id] for fixture_id in dict.fromkeys(requested)]

    run_id = datetime.now(timezone.utc).isoformat(timespec='seconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a', encoding='utf-8') as stream:
        for index in indices:
            ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
            for fixture in fixtures:
                route = tuple(ai.cube.invert_moves(tuple(fixture['scramble'])))
                probe = probe_reference_route(
                    ai, fixture, route,
                    max_probe_steps=args.max_probe_steps,
                    search3_probe_playouts=args.playouts,
                    trace_search3_root=True,
                )
                row = {
                    'recordType': 'search3_fixed_route_prior_visits',
                    'runId': run_id,
                    'aiIndex': index,
                    'checkpoint': checkpoint,
                    'settings': _settings(ai, config, index),
                    'referenceRouteSource': 'inverse_setup',
                    'referenceRouteIsKnownValid': True,
                    'referenceRouteIsOptimal': None,
                    'playoutsPerStep': args.playouts,
                    'probe': probe,
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                summary = probe['summary']
                print(
                    f"AI{index:02d} {fixture['fixtureId']}: "
                    f"{probe['probedMoveCount']} steps, "
                    f"never eligible={summary['rootNeverEligibleCount']}, "
                    f"eligible but unvisited={summary['rootEligibleUnvisitedCount']}",
                    flush=True,
                )
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

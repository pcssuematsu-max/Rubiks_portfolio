"""Compare complete Fixed/Progressive Search3 solves on identical saved weights.

The benchmark is read-only: no greedy fallback, replay insertion, training, or
parameter saving.  The actual number of playouts is retained because a
Progressive solve can stop early or use a rescue tier.
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

from ai.paired_solve_benchmark import (
    fixture_set_id, make_4x4_algorithm_fixtures, run_direct_solve_case,
)
from ai.myperm_composites import make_4x4_myperm_fixtures
from configs.profiles import build_experiment_frame_config
from tools.compare_ai_same_positions import _indices, _load_ai, _settings


def compare_modes(ai, fixtures, modes=('fixed', 'progressive'),
                  max_search_calls=200, max_moves=300):
    """Run the same checkpoint and starting states under each budget mode."""
    if ai.search_mode != 'search3':
        raise ValueError('A Search3 AI is required')
    original_mode = ai.search3_budget_mode
    rows = []
    try:
        for fixture in fixtures:
            for mode in modes:
                ai.search3_budget_mode = mode
                case = run_direct_solve_case(
                    ai, fixture, max_search_calls=max_search_calls,
                    max_moves=max_moves, record_search_trace=True,
                )
                case['budgetMode'] = mode
                rows.append(case)
    finally:
        ai.search3_budget_mode = original_mode
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-indices', default='2,3,4,5,6,7,10,11,18,19')
    parser.add_argument('--fixture-ids', default='wing-2-plus-corner-2-swap',
                        help='Comma-separated fixed fixture IDs; all to use the full set')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--max-search-calls', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports' / 'search3-budget-comparison.jsonl')
    args = parser.parse_args(argv)
    config = build_experiment_frame_config()
    if config.cube_size != 4:
        parser.error('The fixed 4x4 algorithms require a 4x4 experiment profile')
    if args.max_search_calls < 1 or args.max_moves < 1:
        parser.error('Search and move limits must be positive')
    try:
        indices = _indices(args.ai_indices, len(config.ai_search_modes))
    except ValueError as exc:
        parser.error(str(exc))
    if any(config.ai_search_modes[index] != 'search3' for index in indices):
        parser.error('All selected AI indices must use Search3')
    fixtures = make_4x4_algorithm_fixtures() + make_4x4_myperm_fixtures()
    if args.fixture_ids.strip().lower() != 'all':
        requested = [name.strip() for name in args.fixture_ids.split(',') if name.strip()]
        known = {fixture['fixtureId'] for fixture in fixtures}
        if not requested or any(name not in known for name in requested):
            parser.error('Unknown or empty fixture ID list')
        fixtures = [fixture for fixture in fixtures if fixture['fixtureId'] in requested]
    run_id = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for index in indices:
            ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
            settings = _settings(ai, config, index)
            for case in compare_modes(
                ai, fixtures, max_search_calls=args.max_search_calls,
                max_moves=args.max_moves,
            ):
                row = {
                    'recordType': 'search3_budget_case', 'runId': run_id,
                    'fixtureSetId': fixture_set_id(fixtures), 'aiIndex': index,
                    'checkpoint': checkpoint, 'settings': settings, **case,
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                print(
                    f"AI{index:02d} {case['fixtureId']} {case['budgetMode']}: "
                    f"{int(case['directSolved'])} solved, {case['playouts']} playouts, "
                    f"{case['searchCalls']} calls", flush=True,
                )
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

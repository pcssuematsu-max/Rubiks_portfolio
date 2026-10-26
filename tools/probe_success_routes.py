"""Compare several solved Original Search2 routes with other saved AI checkpoints.

Only the first few decisions are probed.  The whole route is checked for a
solved final state, but the probe does not train or store replay data.
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

from ai.paired_solve_benchmark import make_4x4_algorithm_fixtures, probe_reference_route, run_direct_solve_case
from ai.myperm_composites import make_4x4_myperm_fixtures
from configs.profiles import build_experiment_frame_config
from tools.compare_ai_same_positions import _indices, _load_ai, _settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-ai-indices', default='0,1,9,12,13')
    parser.add_argument('--target-ai-indices', default='2,3,4,5,6,7,10,11,18,19,20,21,22,23,24')
    parser.add_argument('--fixture-id', default='wing-2-plus-corner-2-swap')
    parser.add_argument('--max-probe-steps', type=int, default=5)
    parser.add_argument('--search3-playouts', type=int, default=1000)
    parser.add_argument('--trace-search3-root', action='store_true',
                        help='Record when each reference move becomes eligible and is first visited')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports' / 'reference-route-prefix-probe.jsonl')
    args = parser.parse_args(argv)
    config = build_experiment_frame_config()
    if config.cube_size != 4:
        parser.error('The selected fixed fixture requires the 4x4 experiment profile')
    if args.max_probe_steps < 1 or args.search3_playouts < 0:
        parser.error('Probe steps must be positive and Search3 playouts nonnegative')
    if args.trace_search3_root and args.search3_playouts == 0:
        parser.error('--trace-search3-root requires --search3-playouts > 0')
    available_fixtures = make_4x4_algorithm_fixtures() + make_4x4_myperm_fixtures()
    fixtures = [fixture for fixture in available_fixtures
                if fixture['fixtureId'] == args.fixture_id]
    if not fixtures:
        parser.error('Unknown fixture ID')
    fixture = fixtures[0]
    try:
        references = _indices(args.reference_ai_indices, len(config.ai_search_modes))
        targets = _indices(args.target_ai_indices, len(config.ai_search_modes))
    except ValueError as exc:
        parser.error(str(exc))
    if any(config.ai_search_modes[index] != 'search2'
           or config.search2_value_loss_types[index] != 'myloss'
           for index in references):
        parser.error('Reference AIs must use Original Search2')
    routes = []
    for index in references:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        case = run_direct_solve_case(ai, fixture, record_route=True)
        if case['directSolved']:
            routes.append((index, checkpoint, case['routeMoves']))
        print(f"Reference AI{index:02d}: {'solved' if case['directSolved'] else case['terminalReason']}", flush=True)
    if not routes:
        parser.error('None of the requested reference AIs solved the fixture')
    run_id = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for index in targets:
            ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
            for reference_index, reference_checkpoint, moves in routes:
                probe = probe_reference_route(
                    ai, fixture, moves, max_probe_steps=args.max_probe_steps,
                    search3_probe_playouts=args.search3_playouts,
                    trace_search3_root=args.trace_search3_root,
                )
                row = {
                    'recordType': 'reference_route_prefix_probe', 'runId': run_id,
                    'aiIndex': index, 'checkpoint': checkpoint,
                    'settings': _settings(ai, config, index),
                    'referenceAiIndex': reference_index,
                    'referenceCheckpoint': reference_checkpoint,
                    **probe,
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                summary = probe['summary']
                message = (
                    f"AI{index:02d} vs Original AI{reference_index:02d}: "
                    f"Policy top3 {summary['policyTop3Count']}/{probe['probedMoveCount']}, "
                    f"Value top3 {summary['valueTop3Count']}/{probe['probedMoveCount']}"
                )
                if 'rootVisitedCount' in summary:
                    message += (
                        f", width-blocked {summary['rootNeverEligibleCount']}, "
                        f"eligible-unvisited {summary['rootEligibleUnvisitedCount']}, "
                        f"visited {summary['rootVisitedCount']}"
                    )
                print(message, flush=True)
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Read-only Search3 C/shallow-width sweep on identical saved checkpoints.

Example:
  python3.14 tools/sweep_search3_inference.py --ai-indices 10,18 \
      --c-values 0.2,0.5,1.0 --shallow-widths 3,5 --fixtures-per-band 1

Each (AI, C, width) variant uses the same checkpoint and fixture hashes.  No
learning, fallback, replay insertion, or parameter save is performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.paired_solve_benchmark import (
    fixture_set_id, make_paired_fixtures, make_perm_fixtures,
    probe_position_neighborhood, run_direct_solve_case, summarize_direct_cases,
)
from configs.profiles import build_experiment_frame_config
from tools.compare_ai_same_positions import _indices, _load_ai


def _numbers(value, cast, label):
    try:
        result = list(dict.fromkeys(cast(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise ValueError(f'{label} must be comma-separated numbers') from exc
    if not result or any(number <= 0 for number in result):
        raise ValueError(f'{label} must contain positive numbers')
    return result


def apply_search3_variant(ai, c_value, shallow_width):
    """Change only exploration C and the shallow rescue tier's fixed width."""
    ai.search3_C = float(c_value)
    ai.search3_C_depth_max = float(c_value)
    ai.search3_budget_shallow_widening_initial_actions = int(shallow_width)
    ai.search3_budget_shallow_widening_options = (int(shallow_width),)
    ai.search3_engine.reset_tree()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-indices', default='10,11,18,19')
    parser.add_argument('--c-values', default='0.2,0.5,1.0')
    parser.add_argument('--shallow-widths', default='3,5')
    parser.add_argument('--fixtures-per-band', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--fixture-set', choices=('generated','perms','palms','all'), default='all')
    parser.add_argument('--max-search-calls', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports' / 'search3-inference-sweep.jsonl')
    parser.add_argument('--no-neighbor-probe', action='store_true')
    parser.add_argument('--search-trace', action='store_true',
                        help='Record per-search root Policy, backed-up Value, visits and budget stages')
    args = parser.parse_args(argv)
    config = build_experiment_frame_config()
    try:
        indices = _indices(args.ai_indices,len(config.ai_search_modes))
        c_values = _numbers(args.c_values,float,'C values')
        widths = _numbers(args.shallow_widths,int,'shallow widths')
    except ValueError as exc:
        parser.error(str(exc))
    if any(config.ai_search_modes[index] != 'search3' for index in indices):
        parser.error('Every selected AI must use Search3')
    if args.max_search_calls < 1 or args.max_moves < 1:
        parser.error('Search limits must be positive')
    fixtures = []
    if args.fixture_set in ('generated','all'):
        fixtures.extend(make_paired_fixtures(args.fixtures_per_band,args.seed))
    if args.fixture_set in ('perms','palms','all'):
        fixtures.extend(make_perm_fixtures())
    set_id = fixture_set_id(fixtures)
    run_id = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for index in indices:
        ai,checkpoint = _load_ai(index,config,args.checkpoint_root)
        baseline = {
            'c': ai.search3_C,
            'depthMaxC': ai.search3_C_depth_max,
            'shallowWidth': ai.search3_budget_shallow_widening_initial_actions,
            'normalWidth': ai.search3_widening_initial_actions,
            'budgetMode': ai.search3_budget_mode,
            'stagePlayouts': list(ai.search3_budget_stage_playouts),
            'rescuePlayouts': ai.search3_budget_rescue_playouts,
        }
        for c_value,shallow_width in product(c_values,widths):
            apply_search3_variant(ai,c_value,shallow_width)
            cases = []
            for fixture in fixtures:
                case = run_direct_solve_case(
                    ai,fixture,max_search_calls=args.max_search_calls,
                    max_moves=args.max_moves,
                    terminal_probe=None if args.no_neighbor_probe else probe_position_neighborhood,
                    record_search_trace=args.search_trace,
                )
                cases.append(case)
                print(
                    f'AI {index:02d} C={c_value:g} width={shallow_width} '
                    f'{fixture["fixtureId"]}: {case["terminalReason"]}',flush=True,
                )
            record = {
                'schemaVersion': 1,
                'recordType': 'search3_inference_sweep',
                'runId': run_id,
                'fixtureSetId': set_id,
                'aiIndex': index,
                'checkpoint': checkpoint,
                'baselineSettings': baseline,
                'variant': {'c': c_value,'depthMaxC': c_value,'shallowWidth': shallow_width},
                'summary': summarize_direct_cases(cases),
                'cases': cases,
            }
            with args.output.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False) + '\n')
            print(
                f'AI {index:02d} C={c_value:g} width={shallow_width}: '
                f'{record["summary"]["solved"]}/{record["summary"]["trials"]}',flush=True,
            )
    print(f'Results: {args.output}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

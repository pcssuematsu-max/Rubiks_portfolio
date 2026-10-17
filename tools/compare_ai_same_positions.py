"""Compare saved experiment AIs on identical held-out 3x3 positions.

Run separately from the GUI's 200-trial training loop. No learning, greedy
fallback, replay insertion, or parameter saving occurs here. Save the GUI's
parameters before running if its latest in-memory state should be compared.

Examples:
  python3.14 tools/compare_ai_same_positions.py --list-fixtures
  python3.14 tools/compare_ai_same_positions.py --ai-indices 0,3,10,20,21,22,23,24
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
    fixture_set_id, make_paired_fixtures, make_palm_fixtures, run_direct_solve_case,
    summarize_direct_cases,
)
from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config
from cube.rubiks_cube import Rubiks_3


def _indices(value, ai_count):
    if value.strip().lower() == 'all':
        return list(range(ai_count))
    try:
        result = list(dict.fromkeys(int(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise ValueError('AI indices must be comma-separated integers or all') from exc
    if not result or any(index < 0 or index >= ai_count for index in result):
        raise ValueError(f'AI indices must be within 0..{ai_count - 1}')
    return result


def _apply_inference_settings(ai, index, config):
    """Mirror Frame's search settings without constructing a GUI or replay."""
    ai.use_torch_predict = bool(config.use_torch_predict[index])
    ai.skip_search = bool(config.skip_search[index])
    ai.skip_difference = float(config.search2_skip_differences[index])
    ai.search_num2 = 10000
    ai.search2_max_frontier = int(config.search2_max_frontiers[index])
    ai.search2_torch_batch_size = int(config.search2_torch_batch_sizes[index])
    ai.search3_C = float(config.search3_cs[index])
    ai.search3_C_depth_max = float(config.search3_c_depth_maxes[index])
    ai.search3_C_depth_start_depth = int(config.search3_c_depth_start_depths[index])
    ai.search3_C_depth_ramp_depth = int(config.search3_c_depth_ramp_depths[index])
    ai.search3_widening_initial_actions = int(config.search3_widening_initial_actions[index])
    ai.search3_widening_visit_growth = float(config.search3_widening_visit_growths[index])
    ai.search3_budget_mode = str(config.search3_budget_modes[index])
    ai.search3_budget_stage_playouts = tuple(config.search3_budget_stage_playouts[index])
    ai.search3_budget_confidence_visit_share = float(config.search3_budget_confidence_visit_shares[index])
    ai.search3_budget_min_improvement = float(config.search3_budget_min_improvements[index])
    ai.search3_budget_min_playout_depth = float(config.search3_budget_min_playout_depths[index])
    ai.search3_budget_min_visit_share_gain = float(config.search3_budget_min_visit_share_gains[index])
    ai.search3_budget_rescue_playouts = int(config.search3_budget_rescue_playouts[index])
    ai.search3_budget_rescue_min_visit_share_gain = float(config.search3_budget_rescue_min_visit_share_gains[index])
    ai.search3_budget_rescue_min_improvement = float(config.search3_budget_rescue_min_improvements[index])
    ai.search3_budget_shallow_depth_threshold = float(config.search3_budget_shallow_depth_thresholds[index])
    ai.search3_budget_shallow_widening_initial_actions = int(
        config.search3_budget_shallow_widening_initial_actions[index]
    )
    # The GUI alternates (3, 5) between ordinary solves.  A benchmark must
    # choose one width deterministically and record it, not inherit a counter.
    ai.search3_budget_shallow_widening_options = tuple(
        config.search3_budget_shallow_widening_options[index]
    )
    ai.search3_max_node_cache = int(config.search3_max_node_caches[index])
    ai.search3_max_prediction_cache = int(config.search3_max_prediction_caches[index])


def _load_ai(index, config, checkpoint_root):
    cube = Rubiks_3(size=config.cube_size)
    ai = Rubiks_3_AI(
        [512] * 8,
        cube_size=config.cube_size,
        cube=cube,
        search_mode=config.ai_search_modes[index],
        Activation=config.activations[index],
        residual=config.residuals[index],
        use_transformer_attention=config.original_transformer_attention[index],
        transformer_attention_dim=config.original_transformer_attention_dims[index],
        transformer_attention_token_mode=config.original_transformer_attention_token_modes[index],
        transformer_linear_skip=config.original_transformer_linear_skips[index],
        transformer_linear_skip_variance=config.original_transformer_linear_skip_variances[index],
        piece_attention_backward_chunk_size=config.original_piece_attention_backward_chunk_sizes[index],
        search2_value_loss_type=config.search2_value_loss_types[index],
        search2_value_loss_margin=config.search2_value_loss_margins[index],
        w1_initializers=config.w1_initializers[index],
    )
    _apply_inference_settings(ai, index, config)
    directory = Path(checkpoint_root) / f'AIdatas{index}'
    digest = hashlib.sha256()
    modified = []
    for key in sorted(ai.params):
        source = directory / f'{key}.npy'
        if not source.is_file():
            raise FileNotFoundError(f'Incomplete checkpoint: {source}')
        loaded = np.load(source, mmap_mode='r', allow_pickle=False)
        if loaded.shape != ai.params[key].shape:
            raise ValueError(f'{source}: {loaded.shape} != {ai.params[key].shape}')
        ai.params[key][:] = loaded
        digest.update(key.encode('utf-8'))
        digest.update(np.ascontiguousarray(ai.params[key]).tobytes())
        modified.append(source.stat().st_mtime)
    ai.mark_params_dirty()
    ai.set_perfect_val()
    return ai, {
        'directory': str(directory.resolve()),
        'sha256': digest.hexdigest(),
        'newestFileModifiedAt': datetime.fromtimestamp(max(modified), timezone.utc).isoformat(),
    }


def _settings(ai, config, index):
    return {
        'model': 'transformer' if ai.is_transformer else 'linear',
        'searchMode': ai.search_mode,
        'valueLossType': config.search2_value_loss_types[index],
        'search3RankLossMix': float(config.search3_rank_loss_mixes[index]),
        'trainRecentRatio': float(config.original_train_recent_ratios[index]),
        'trainBatchMixMode': str(config.original_train_batch_mix_modes[index]),
        'learningRate': float(config.lrs[index]),
        'weightDecayRate': float(config.wdlrs[index]),
        'trainingTransformIndex': int(config.transform_idx[index]),
        'trainingFlipInside': bool(config.flip_inside_idx[index]),
        'search2SkipDifference': ai.skip_difference,
        'search2Budget': ai.search_num2,
        'search3C': ai.search3_C if ai.search_mode == 'search3' else None,
        'search3DepthCMax': ai.search3_C_depth_max if ai.search_mode == 'search3' else None,
        'search3BudgetMode': ai.search3_budget_mode if ai.search_mode == 'search3' else None,
        'search3StagePlayouts': list(ai.search3_budget_stage_playouts) if ai.search_mode == 'search3' else None,
        'search3RescuePlayouts': ai.search3_budget_rescue_playouts if ai.search_mode == 'search3' else None,
        'search3InitialWidth': ai.search3_widening_initial_actions if ai.search_mode == 'search3' else None,
        'search3ShallowWidth': ai.search3_budget_shallow_widening_initial_actions if ai.search_mode == 'search3' else None,
    }


def _paired_summary(records, fixtures):
    """Make the cross-AI result of each *identical* start state explicit."""
    by_ai = {record['aiIndex']: record for record in records}
    comparisons = []
    for fixture_index, fixture in enumerate(fixtures):
        for record in records:
            case = record['cases'][fixture_index]
            if (
                case['fixtureId'] != fixture['fixtureId']
                or case['initialStateHash'] != fixture['initialStateHash']
            ):
                raise ValueError('Paired results do not share the same initial state')
        solved = [
            index for index, record in by_ai.items()
            if record['cases'][fixture_index]['directSolved']
        ]
        comparisons.append({
            'fixtureId': fixture['fixtureId'],
            'band': fixture['band'],
            'initialStateHash': fixture['initialStateHash'],
            'solvedBy': solved,
            'failedBy': [index for index in by_ai if index not in solved],
        })
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-indices', default='all', help='Comma-separated AI indices; default: all 25')
    parser.add_argument('--fixtures-per-band', type=int, default=2)
    parser.add_argument(
        '--fixture-set', choices=('generated', 'palms', 'all'), default='generated',
        help='Use the original fixed set, the T/Y palm setups, or both',
    )
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--max-search-calls', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports' / 'ai-paired-benchmark.jsonl')
    parser.add_argument('--list-fixtures', action='store_true', help='Print the fixed setups without loading AIs')
    args = parser.parse_args(argv)
    if args.max_search_calls < 1 or args.max_moves < 1:
        parser.error('max-search-calls and max-moves must be positive')
    try:
        fixtures = []
        if args.fixture_set in ('generated', 'all'):
            fixtures.extend(make_paired_fixtures(args.fixtures_per_band, args.seed))
        if args.fixture_set in ('palms', 'all'):
            fixtures.extend(make_palm_fixtures())
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    fixture_id = fixture_set_id(fixtures)
    if args.list_fixtures:
        print(json.dumps({'fixtureSetId': fixture_id, 'fixtures': fixtures}, ensure_ascii=False, indent=2))
        return 0

    config = build_experiment_frame_config()
    if config.cube_size != 3:
        parser.error('The paired fixture set is only defined for the 3x3 cube')
    try:
        indices = _indices(args.ai_indices, len(config.ai_search_modes))
    except ValueError as exc:
        parser.error(str(exc))
    run_id = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'fixtureSet={fixture_id}; run={run_id}; AI count={len(indices)}', flush=True)
    records = []
    for index in indices:
        ai, checkpoint = _load_ai(index, config, args.checkpoint_root)
        cases = []
        for fixture in fixtures:
            case = run_direct_solve_case(
                ai, fixture,
                max_search_calls=args.max_search_calls,
                max_moves=args.max_moves,
            )
            cases.append(case)
            print(
                f'AI {index:02d} {fixture["fixtureId"]:>3} '
                f'{fixture["band"]:<19} {"solved" if case["directSolved"] else case["terminalReason"]} '
                f'{case["elapsedSeconds"]:.2f}s',
                flush=True,
            )
        summary = summarize_direct_cases(cases)
        record = {
            'schemaVersion': 1,
            'recordType': 'ai_result',
            'runId': run_id,
            'fixtureSetId': fixture_id,
            'completedAt': datetime.now(timezone.utc).isoformat(),
            'aiIndex': index,
            'checkpoint': checkpoint,
            'settings': _settings(ai, config, index),
            'summary': summary,
            'cases': cases,
        }
        with args.output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        records.append(record)
        print(f'AI {index:02d} direct success: {summary["solved"]}/{summary["trials"]}', flush=True)
        del ai
        gc.collect()
    comparison = {
        'schemaVersion': 1,
        'recordType': 'run_summary',
        'runId': run_id,
        'fixtureSetId': fixture_id,
        'completedAt': datetime.now(timezone.utc).isoformat(),
        'aiIndices': indices,
        'caseComparisons': _paired_summary(records, fixtures),
    }
    with args.output.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(comparison, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    print('Paired outcomes (same fixture order):', flush=True)
    print('AI   ' + ' '.join(case['fixtureId'].rjust(3) for case in fixtures), flush=True)
    for record in records:
        marks = ' '.join('  Y' if case['directSolved'] else '  .' for case in record['cases'])
        print(f'{record["aiIndex"]:02d}   {marks}', flush=True)
    print(f'Results: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

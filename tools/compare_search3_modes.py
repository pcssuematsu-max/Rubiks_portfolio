"""Compare Search3 budget/width settings on held-out scrambles and one checkpoint.

Example: python tools/compare_search3_modes.py --ai-index 10 --lengths 6,10,14
The command never trains, saves parameters, or adds fixtures to replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from ai.fixed_validation import _fixture_scramble
from ai.rubiks_ai import Rubiks_3_AI
from ai.search3_paired_probe import compare_search3_variants
from configs.profiles import build_experiment_frame_config
from cube.rubiks_cube import Rubiks_3


def _loaded_ai(index, config):
    if config.ai_search_modes[index] != 'search3':
        raise ValueError(f'AI {index} is not Search3')
    ai = Rubiks_3_AI(
        [512] * 8,
        cube_size = config.cube_size,
        cube = Rubiks_3(size = config.cube_size),
        search_mode = 'search3',
        Activation = config.activations[index],
        residual = config.residuals[index],
        use_transformer_attention = config.original_transformer_attention[index],
        transformer_attention_dim = config.original_transformer_attention_dims[index],
        transformer_attention_token_mode = config.original_transformer_attention_token_modes[index],
        transformer_linear_skip = config.original_transformer_linear_skips[index],
        transformer_linear_skip_variance = config.original_transformer_linear_skip_variances[index],
    )
    directory = ROOT / f'AIdatas{index}'
    for key, array in ai.params.items():
        source = directory / f'{key}.npy'
        if not source.exists():
            raise FileNotFoundError(source)
        loaded = np.load(source, mmap_mode = 'r')
        if loaded.shape != array.shape:
            raise ValueError(f'{source}: {loaded.shape} != {array.shape}')
        array[:] = loaded
    ai.mark_params_dirty()
    ai.set_perfect_val()
    ai.search3_C = float(config.search3_cs[index])
    ai.search3_C_depth_max = float(config.search3_c_depth_maxes[index])
    ai.search3_C_depth_start_depth = int(config.search3_c_depth_start_depths[index])
    ai.search3_C_depth_ramp_depth = int(config.search3_c_depth_ramp_depths[index])
    ai.search3_widening_visit_growth = float(config.search3_widening_visit_growths[index])
    ai.search3_budget_stage_playouts = tuple(config.search3_budget_stage_playouts[index])
    ai.search3_budget_confidence_visit_share = float(config.search3_budget_confidence_visit_shares[index])
    ai.search3_budget_min_improvement = float(config.search3_budget_min_improvements[index])
    ai.search3_budget_min_playout_depth = float(config.search3_budget_min_playout_depths[index])
    ai.search3_budget_min_visit_share_gain = float(config.search3_budget_min_visit_share_gains[index])
    ai.search3_budget_rescue_playouts = int(config.search3_budget_rescue_playouts[index])
    ai.search3_budget_rescue_min_visit_share_gain = float(config.search3_budget_rescue_min_visit_share_gains[index])
    ai.search3_budget_rescue_min_improvement = float(config.search3_budget_rescue_min_improvements[index])
    ai.search3_budget_shallow_depth_threshold = float(config.search3_budget_shallow_depth_thresholds[index])
    ai.search3_max_node_cache = int(config.search3_max_node_caches[index])
    ai.search3_max_prediction_cache = int(config.search3_max_prediction_caches[index])
    return ai


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--ai-index',type = int,default = 10)
    parser.add_argument('--lengths',default = '6,10,14')
    parser.add_argument('--fixtures-per-length',type = int,default = 2)
    parser.add_argument('--budget-scale',type = float,default = 1.0,
                        help = 'Use less than 1 only for a quick smoke test')
    parser.add_argument('--compare-widening',action = 'store_true',
                        help = 'Compare fixed width against bounded policy-mass width at the same budget')
    parser.add_argument('--budget-mode',choices = ('fixed','progressive'),
                        help = 'Override the profile budget mode for a matched comparison')
    parser.add_argument('--fixed-width',type = int,default = 5)
    parser.add_argument('--policy-mass',type = float,default = 0.8)
    parser.add_argument('--min-width',type = int,default = 3)
    parser.add_argument('--max-width',type = int,default = 6)
    args = parser.parse_args(argv)
    config = build_experiment_frame_config()
    if args.ai_index < 0 or args.ai_index >= len(config.ai_search_modes):
        parser.error('AI index is out of range')
    if args.fixtures_per_length < 1 or args.fixtures_per_length > 3:
        parser.error('fixtures-per-length must be 1..3')
    if not 0 < args.budget_scale <= 1:
        parser.error('budget-scale must be in (0,1]')
    if not 0 <= args.policy_mass <= 1 or not 1 <= args.min_width <= args.max_width or args.fixed_width < 1:
        parser.error('Widening settings require mass in [0,1] and 1 <= min <= max and fixed-width >= 1')
    lengths = tuple(int(value) for value in args.lengths.split(','))
    if not lengths or any(length < 1 for length in lengths):
        parser.error('lengths must be positive comma-separated integers')
    ai = _loaded_ai(args.ai_index,config)
    if args.budget_scale != 1.0:
        ai.search_num3 = max(1,round(ai.search_num3 * args.budget_scale))
        ai.search3_budget_stage_playouts = tuple(
            max(1,round(value * args.budget_scale))
            for value in ai.search3_budget_stage_playouts
        )
        ai.search3_budget_rescue_playouts = round(
            ai.search3_budget_rescue_playouts * args.budget_scale
        )
    scrambles = [
        _fixture_scramble(ai.cube,length,fixture_index)
        for length in lengths for fixture_index in range(args.fixtures_per_length)
    ]
    variants = None
    if args.compare_widening:
        budget_mode = args.budget_mode or config.search3_budget_modes[args.ai_index]
        variants = (
            ('fixed_width',budget_mode,args.fixed_width,False),
            ('policy_mass',budget_mode,args.fixed_width,False,{
                'mode': 'policy_mass', 'mass': args.policy_mass,
                'minimum': args.min_width, 'maximum': args.max_width,
            }),
        )
    rows = compare_search3_variants(ai,scrambles,variants=variants) if variants else compare_search3_variants(ai,scrambles)
    print(f'AI {args.ai_index}; C={ai.search3_C}; checkpoint={ROOT / f"AIdatas{args.ai_index}"}; budget scale={args.budget_scale}')
    print('length effect variant solved stop playouts depthMean depthMax widthMean policyTop3 rootTop inverseVisit inverseRank seconds')
    for row in rows:
        print(
            f'{len(row["scramble"]):>6} {str(row["effectMovedPieceCount"]):>6} '
            f'{row["variant"]:<18} {int(row["solvedAtRoot"]):>6} '
            f'{row["endReason"]:<12} {row["consumedPlayouts"]:>8} '
            f'{row["playoutDepthMean"]:>9.2f} {row["playoutDepthMax"]:>8} '
            f'{row["wideningActionLimitMean"]:>9.2f} {row["rootTop3PolicyMass"]:>10.3f} '
            f'{row["rootTopVisitShare"] or 0:>7.3f} '
            f'{row["inverseFirstMoveVisitShare"] or 0:>12.3f} '
            f'{row["inverseFirstMovePolicyRank"]:>11} '
            f'{row["elapsedSeconds"]:>7.2f}'
        )
    if args.compare_widening:
        for label in ('fixed_width', 'policy_mass'):
            matching = [row for row in rows if row['variant'] == label]
            print(
                f'{label}: solved {sum(row["solvedAtRoot"] for row in matching)}/{len(matching)}, '
                f'mean width {np.mean([row["wideningActionLimitMean"] for row in matching]):.2f}, '
                f'total time {sum(row["elapsedSeconds"] for row in matching):.2f}s'
            )


if __name__ == '__main__':
    main()

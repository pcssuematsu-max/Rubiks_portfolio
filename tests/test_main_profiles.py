import unittest
from dataclasses import fields

from configs import build_frame_config
from main import parse_args
from ui.frame_config import FrameConfig


class MainProfileTests(unittest.TestCase):
    def test_public_profile_is_the_cli_default(self):
        args = parse_args([])
        config = build_frame_config(args.profile)

        self.assertEqual(args.profile, "public")
        self.assertEqual(config.puzzle_type, "rubiks")
        self.assertEqual(config.cube_size, 3)
        self.assertEqual(config.ai_search_modes, ("search2",))
        self.assertEqual(config.search2_max_frontiers, (5000,))
        self.assertEqual(config.use_torch, (False,))
        self.assertEqual(config.bootstrap_datas, ())
        self.assertEqual(config.control_panel_mode, "simple")

    def test_experiment_profile_preserves_the_original_configuration(self):
        config = build_frame_config("experiment")

        self.assertEqual(config.puzzle_type, "cube")
        self.assertEqual(config.cube_size, 3)
        self.assertEqual(len(config.ai_search_modes), 25)
        self.assertEqual(config.control_panel_mode, "advanced")
        self.assertEqual(config.steps_to_goal_value_loss_weights[20:], [5.0, 5.0, 3.0, 1.0, 1.0])
        self.assertEqual([config.hard_effect_direct_min_ratios[index] for index in (10,18,22)], [0.40] * 3)
        self.assertEqual([config.hard_effect_direct_min_ratios[index] for index in (11,19)], [0.40] * 2)
        self.assertEqual(config.hard_effect_direct_min_ratios[21], 0.0)
        self.assertEqual(config.steps_to_goal_states_per_bands[20:], [12, 12, 12, 12, 12])
        self.assertEqual(config.steps_to_goal_replay_stratified_ratios[20:], [0.0, 0.0, 0.75, 0.75, 0.75])
        self.assertEqual(config.steps_to_goal_long_route_min_ratios[20:], [0.0, 0.0, 0.25, 0.0, 0.0])
        self.assertEqual([config.original_long_direct_min_ratios[index] for index in (12,13,14,15,16,17)], [0.10] * 6)
        self.assertEqual(config.original_long_direct_min_ratios[20:], [0.0] * 5)
        self.assertEqual(config.search3_budget_shallow_depth_thresholds[10], 9.0)
        self.assertEqual(config.search3_budget_shallow_depth_thresholds[18], 9.0)
        self.assertEqual(config.search3_budget_shallow_depth_thresholds[11], 9.0)
        self.assertEqual(config.search3_budget_shallow_widening_initial_actions[10], 3)
        self.assertEqual(config.search3_budget_shallow_widening_options[10], (3, 5))
        self.assertEqual(config.search3_budget_shallow_widening_options[18], (3, 5))
        self.assertEqual(config.steps_to_goal_value_max_route_efficiencies[20:], [2.0, 0.0, 2.0, 0.0, 0.0])
        self.assertEqual(config.steps_to_goal_value_batch_band_max_copies[20:], [3, 1, 3, 1, 1])
        self.assertEqual(config.pairwise_fallback_max_ratios[20:], [1.0, 1.0, 1.0, 0.25, 0.25])
        self.assertEqual(config.search2_skip_differences[20:], [1.0, 1.0, 1.0, 1.0, 1000.0])
        self.assertEqual(config.search3_budget_rescue_playouts[3], 10000)
        self.assertEqual(config.search3_budget_rescue_playouts[2], 0)
        self.assertEqual(config.original_train_recent_ratios[20:], [0.0, 0.0, 0.25, 0.25, 0.25])
        self.assertEqual(
            [config.original_train_recent_ratios[index] for index in (10,11,18,19)],
            [0.5,0.5,0.5,0.25],
        )
        self.assertEqual(
            [config.original_train_batch_mix_modes[index] for index in (10,11,18,19)],
            ['ordered','mixed','mixed','mixed'],
        )
        self.assertEqual(
            [config.search3_cs[index] for index in (10,11,18,19)],
            [0.5] * 4,
        )
        self.assertEqual(
            [config.search3_budget_modes[index] for index in (10,11,18,19)],
            ['fixed'] * 4,
        )
        self.assertEqual(
            [config.transform_idx[index] for index in (10,11,18,19)],
            [0,0,24,24],
        )
        for first, second, expected_difference in (
            (10,11,'original_train_batch_mix_modes'),
            (18,19,'original_train_recent_ratios'),
        ):
            differences = {
                field.name for field in fields(config)
                if isinstance((values := getattr(config,field.name)),(list,tuple))
                and len(values) == len(config.ai_search_modes)
                and values[first] != values[second]
            }
            self.assertEqual(differences,{expected_difference})
        self.assertEqual(config.original_transformer_linear_skips, config.original_transformer_attention)
        self.assertEqual(config.original_transformer_linear_skip_variances[10], 0.125)
        self.assertEqual(config.original_transformer_linear_skip_variances[22], 0.125)
        self.assertEqual(config.original_transformer_linear_skip_variances[0], 0.0)

    def test_test_profile_is_an_alias_for_experiment(self):
        config = build_frame_config("test")

        self.assertEqual(config.puzzle_type, "cube")
        self.assertEqual(len(config.ai_search_modes), 25)

    def test_control_panel_mode_must_be_simple_or_advanced(self):
        with self.assertRaisesRegex(ValueError, "control_panel_mode"):
            FrameConfig(control_panel_mode="compact")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            build_frame_config("unknown")


if __name__ == "__main__":
    unittest.main()

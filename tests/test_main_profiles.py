import unittest

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
        self.assertEqual(config.steps_to_goal_value_loss_weights[20:], [5.0, 5.0, 2.0, 2.0, 2.0])
        self.assertEqual(config.steps_to_goal_states_per_bands[20:], [12, 12, 12, 12, 12])

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

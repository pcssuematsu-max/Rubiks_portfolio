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

        self.assertEqual(config.puzzle_type, "cto")
        self.assertEqual(config.cube_size, 7)
        self.assertEqual(len(config.ai_search_modes), 20)
        self.assertEqual(config.control_panel_mode, "advanced")

    def test_test_profile_is_an_alias_for_experiment(self):
        config = build_frame_config("test")

        self.assertEqual(config.puzzle_type, "cto")
        self.assertEqual(len(config.ai_search_modes), 20)

    def test_control_panel_mode_must_be_simple_or_advanced(self):
        with self.assertRaisesRegex(ValueError, "control_panel_mode"):
            FrameConfig(control_panel_mode="compact")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            build_frame_config("unknown")


if __name__ == "__main__":
    unittest.main()

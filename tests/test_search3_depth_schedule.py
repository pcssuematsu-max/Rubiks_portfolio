import unittest
from types import SimpleNamespace

import numpy as np

from cube.search3_engine import Node, Search3Engine
from configs.profiles import build_experiment_frame_config
from ui.frame import Frame
from ui.frame_config import FrameConfig


class Search3DepthScheduleTests(unittest.TestCase):
    def test_linearly_increases_from_root_c_to_depth_maximum(self):
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = SimpleNamespace(
            search3_C = 2.0,
            search3_C_depth_max = 4.0,
            search3_C_depth_ramp_depth = 12,
        )

        self.assertEqual(engine._exploration_c(0), 2.0)
        self.assertEqual(engine._exploration_c(6), 3.0)
        self.assertEqual(engine._exploration_c(12), 4.0)
        self.assertEqual(engine._exploration_c(30), 4.0)

    def test_disabled_schedule_keeps_the_base_coefficient(self):
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = SimpleNamespace(
            search3_C = 2.0,
            search3_C_depth_max = 4.0,
            search3_C_depth_ramp_depth = 0,
        )

        self.assertEqual(engine._exploration_c(0), 2.0)
        self.assertEqual(engine._exploration_c(20), 2.0)

    def test_node_uses_the_depth_specific_coefficient_at_selection_time(self):
        node = Node(np.array([0.9, 0.1], dtype = 'f'), value = 0.0, C = 2.0)
        node.S = 100
        node.visited[:] = (10, 0)
        node.val[:] = (8.0, 0.0)

        self.assertEqual(node.select_node(C = 0.2), 0)
        self.assertEqual(node.select_node(C = 2.0), 1)

    def test_search_diagnostics_report_actual_depth_and_coefficient_use(self):
        diagnostics = Search3Engine._search_diagnostics(
            [1, 12, 20, 30],
            [2.0, 2.5, 3.0, 4.0],
        )

        self.assertEqual(diagnostics['playoutDepthMedian'], 16.0)
        self.assertEqual(diagnostics['playoutDepthAtLeast20Count'], 2)
        self.assertEqual(diagnostics['playoutDepthAtLeast20Rate'], 0.5)
        self.assertEqual(diagnostics['selectionCMean'], 2.875)
        self.assertEqual(diagnostics['selectionCMax'], 4.0)

    def test_frame_config_validates_per_ai_depth_schedule(self):
        config = FrameConfig(
            ai_search_modes = ('search3',),
            search3_cs = (2.0,),
            search3_c_depth_maxes = (4.0,),
            search3_c_depth_ramp_depths = (12,),
            search3_budget_modes = ('progressive',),
            search3_budget_stage_playouts = ((100, 300, 600),),
            search3_budget_confidence_visit_shares = (0.7,),
            search3_budget_min_improvements = (0.05,),
            search3_budget_min_playout_depths = (3.0,),
            search3_budget_min_visit_share_gains = (0.03,),
            search3_max_node_caches = (10000,),
            search3_max_prediction_caches = (10000,),
            original_train_medium_sequence_min_steps = (20,),
            original_train_medium_sequence_max_steps = (29,),
            original_train_medium_sequence_ratios = (0.25,),
            original_train_long_sequence_max_ratios = (0.60,),
            original_train_long_sequence_min_steps = (20,),
            original_train_long_sequence_ratios = (0.25,),
        )

        self.assertEqual(config.search3_c_depth_maxes, (4.0,))
        self.assertEqual(config.search3_c_depth_ramp_depths, (12,))
        self.assertEqual(config.search3_budget_modes, ('progressive',))
        self.assertEqual(config.search3_budget_stage_playouts, ((100, 300, 600),))
        self.assertEqual(config.search3_max_node_caches, (10000,))
        self.assertEqual(config.original_train_medium_sequence_max_steps, (29,))
        self.assertEqual(config.original_train_long_sequence_min_steps, (20,))
        self.assertEqual(config.original_train_long_sequence_ratios, (0.25,))

    def test_runtime_settings_apply_the_depth_schedule(self):
        ai = SimpleNamespace(search3_C = 0.05)
        frame = SimpleNamespace(
            AInum = 1,
            AIs = [ai],
            _apply_update_scales = lambda _ai, _scales: None,
        )

        Frame._apply_ai_runtime_settings(
            frame,
            search3_cs = (2.0,),
            search3_c_depth_maxes = (4.0,),
            search3_c_depth_ramp_depths = (12,),
            search3_budget_modes = ('progressive',),
            search3_budget_stage_playouts = ((100, 300, 600),),
            search3_budget_confidence_visit_shares = (0.7,),
            search3_budget_min_improvements = (0.05,),
            search3_budget_min_playout_depths = (3.0,),
            search3_budget_min_visit_share_gains = (0.03,),
            search3_max_node_caches = (10000,),
            search3_max_prediction_caches = (10000,),
        )

        self.assertEqual(ai.search3_C, 2.0)
        self.assertEqual(ai.search3_C_depth_max, 4.0)
        self.assertEqual(ai.search3_C_depth_ramp_depth, 12)
        self.assertEqual(ai.search3_budget_mode, 'progressive')
        self.assertEqual(ai.search3_budget_stage_playouts, (100, 300, 600))
        self.assertEqual(ai.search3_budget_confidence_visit_share, 0.7)
        self.assertEqual(ai.search3_budget_min_visit_share_gain, 0.03)
        self.assertEqual(ai.search3_max_node_cache, 10000)
        self.assertEqual(ai.search3_max_prediction_cache, 10000)

    def test_experiment_profile_enables_progressive_budget_for_linear_search3(self):
        config = build_experiment_frame_config()

        self.assertEqual(
            [config.search3_budget_modes[index] for index in (3, 5, 7)],
            ['progressive', 'progressive', 'progressive'],
        )
        self.assertEqual(config.search3_max_node_caches[3], 10000)
        self.assertEqual(config.search3_max_prediction_caches[3], 10000)
        self.assertEqual(
            [config.search3_budget_min_improvements[index] for index in (3, 5, 7)],
            [0.02, 0.02, 0.02],
        )
        self.assertEqual(config.search3_budget_min_improvements[2], 0.05)
        self.assertEqual(config.original_train_medium_sequence_min_steps[10], 20)
        self.assertEqual(config.original_train_medium_sequence_max_steps[10], 29)
        self.assertEqual(config.original_train_short_sequence_max_steps[10], 19)
        self.assertEqual(config.original_train_short_sequence_min_ratios[10], 0.20)
        self.assertEqual(config.original_train_long_sequence_min_steps[10], 30)
        self.assertEqual(config.original_train_long_sequence_max_ratios[10], 0.60)
        self.assertEqual(config.search3_rank_loss_mixes[10], 0.10)
        self.assertEqual(config.search3_rank_loss_mixes[11], 0.10)
        self.assertEqual(config.search3_rank_loss_mixes[18], 0.0)
        self.assertEqual(config.search3_rank_loss_mixes[19], 0.0)
        self.assertEqual(len(config.ai_search_modes), 25)
        self.assertEqual(
            [config.search2_value_loss_types[index] for index in range(20, 25)],
            ['steps_to_goal'] * 5,
        )
        self.assertEqual(config.search2_rank_loss_mixes[20:25], [0.0] * 5)
        self.assertEqual(config.search2_rank_loss_apply_types[20:25], ['none'] * 5)
        self.assertEqual(config.search2_value_target_scales[20:25], [1.0] * 5)
        self.assertEqual(config.original_transformer_attention[20:25], [False, False, True, True, True])
        self.assertEqual(config.transform_idx[20:25], [0] * 5)
        self.assertEqual(config.flip_inside_idx[20:25], [False] * 5)
        self.assertEqual(
            config.search3_progress,
            [mode == 'progressive' for mode in config.search3_budget_modes],
        )

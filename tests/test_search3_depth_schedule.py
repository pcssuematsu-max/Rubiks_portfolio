import unittest
from types import SimpleNamespace

import numpy as np

from cube.search3_engine import Node, Search3Engine
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

    def test_frame_config_validates_per_ai_depth_schedule(self):
        config = FrameConfig(
            ai_search_modes = ('search3',),
            search3_cs = (2.0,),
            search3_c_depth_maxes = (4.0,),
            search3_c_depth_ramp_depths = (12,),
        )

        self.assertEqual(config.search3_c_depth_maxes, (4.0,))
        self.assertEqual(config.search3_c_depth_ramp_depths, (12,))

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
        )

        self.assertEqual(ai.search3_C, 2.0)
        self.assertEqual(ai.search3_C_depth_max, 4.0)
        self.assertEqual(ai.search3_C_depth_ramp_depth, 12)

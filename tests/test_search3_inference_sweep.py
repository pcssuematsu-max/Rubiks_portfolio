from types import SimpleNamespace
import unittest

from tools.sweep_search3_inference import _numbers, apply_search3_variant


class Search3InferenceSweepTests(unittest.TestCase):
    def test_variant_changes_only_c_and_shallow_width(self):
        resets = []
        ai = SimpleNamespace(
            search3_C=0.5,search3_C_depth_max=0.5,
            search3_widening_initial_actions=5,
            search3_budget_shallow_widening_initial_actions=3,
            search3_budget_shallow_widening_options=(3,5),
            search3_budget_stage_playouts=(1000,3000,6000),
            search3_engine=SimpleNamespace(reset_tree=lambda: resets.append(True)),
        )
        apply_search3_variant(ai,0.2,4)
        self.assertEqual((ai.search3_C,ai.search3_C_depth_max),(0.2,0.2))
        self.assertEqual(ai.search3_budget_shallow_widening_options,(4,))
        self.assertEqual(ai.search3_widening_initial_actions,5)
        self.assertEqual(ai.search3_budget_stage_playouts,(1000,3000,6000))
        self.assertEqual(resets,[True])

    def test_sweep_values_must_be_positive(self):
        self.assertEqual(_numbers('0.2,0.5,0.2',float,'C'),[0.2,0.5])
        with self.assertRaises(ValueError):
            _numbers('0,3',int,'width')


if __name__ == '__main__':
    unittest.main()

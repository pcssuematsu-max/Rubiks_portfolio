import unittest
from types import SimpleNamespace

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI
from ai.search3_paired_probe import compare_search3_variants


class Search3PairedProbeTests(unittest.TestCase):
    def test_root_can_keep_all_moves_while_descendants_widen(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.search3_widening_initial_actions = 5
        ai.search3_widening_root_all_actions = True
        root = SimpleNamespace(P=np.ones(18),S=0)
        child = SimpleNamespace(P=np.ones(18),S=0)
        ai.search3_engine.root_node = root
        self.assertEqual(ai.search3_engine._widening_action_limit(root),0)
        self.assertEqual(ai.search3_engine._widening_action_limit(child),5)

    def test_same_ai_state_is_compared_and_restored(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.search3_C = 0.5
        ai.search_num3 = 10
        ai.search_repeat3 = 1
        ai.search3_budget_stage_playouts = (5,5,10)
        ai.search3_budget_rescue_playouts = 0
        ai.search3_budget_mode = 'fixed'
        ai.search3_widening_initial_actions = 0
        ai.search3_budget_shallow_widening_initial_actions = 0
        ai.search3_budget_shallow_widening_options = ()
        ai.cube.scramble(2)
        original_state = ai.cube.state.copy()
        original_weights = {key:value.copy() for key,value in ai.params.items()}

        rows = compare_search3_variants(ai,[(' U ',)])

        self.assertEqual(len(rows),5)
        self.assertEqual({row['search3C'] for row in rows},{0.5})
        self.assertEqual({row['variant'] for row in rows},{
            'fixed_all','fixed_width5','progressive_width5','progressive_width3',
            'progressive_root_all',
        })
        self.assertTrue(all(row['consumedPlayouts'] > 0 for row in rows))
        np.testing.assert_array_equal(ai.cube.state,original_state)
        self.assertEqual(ai.search3_budget_mode,'fixed')
        self.assertEqual(ai.search3_widening_initial_actions,0)
        self.assertIsNone(ai.search3_engine.root_node)
        for key,value in original_weights.items():
            np.testing.assert_array_equal(ai.params[key],value)


if __name__ == '__main__':
    unittest.main()

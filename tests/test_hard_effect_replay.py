import unittest
from types import SimpleNamespace

from ai.rubiks_ai import Rubiks_3_AI
from core.experiment_log import setup_effect_metrics


class HardEffectReplayTests(unittest.TestCase):
    def test_reuses_only_selected_direct_routes_with_two_copy_cap(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.cube_size = 3
        ai.search2_value_loss_type = 'myloss2'
        ai.hard_effect_direct_min_ratio = 0.5
        hard_a = (' U2', " R'", ' D ', ' F ')
        hard_b = (' F2', ' D ', " U'", " F'", " U'", " F'")
        self.assertGreaterEqual(setup_effect_metrics('cube-3x3',3,hard_a)['movedPieceCount'],14)
        self.assertGreaterEqual(setup_effect_metrics('cube-3x3',3,hard_b)['movedPieceCount'],14)
        source = [
            SimpleNamespace(scramble=hard_a, trajectory_origin='direct_search'),
            SimpleNamespace(scramble=hard_b, trajectory_origin='direct_search'),
            SimpleNamespace(scramble=(' U ',), trajectory_origin='fallback'),
            SimpleNamespace(scramble=(' R ',), trajectory_origin='fallback'),
            SimpleNamespace(scramble=(' U ',), trajectory_origin='legacy'),
            SimpleNamespace(scramble=(' R ',), trajectory_origin='legacy'),
        ]
        batches = [[0,2],[1,3],[4,5]]
        summary = ai._enforce_hard_effect_direct_minimum(batches,source)
        selected = ai._flatten_batches(batches)
        self.assertEqual(summary['required_item_count'],3)
        self.assertEqual(summary['selected_before_item_count'],2)
        self.assertEqual(summary['selected_item_count'],3)
        self.assertEqual(summary['oversampled_item_count'],1)
        self.assertLessEqual(selected.count(0),2)
        self.assertLessEqual(selected.count(1),2)
        self.assertEqual(len(selected),6)

    def test_disabled_ratio_does_not_inspect_or_change_replay(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.cube_size = 3
        ai.hard_effect_direct_min_ratio = 0.0
        batches = [[0,1]]
        summary = ai._enforce_hard_effect_direct_minimum(batches,[object(),object()])
        self.assertEqual(batches,[[0,1]])
        self.assertEqual(summary['oversampled_item_count'],0)


if __name__ == '__main__':
    unittest.main()

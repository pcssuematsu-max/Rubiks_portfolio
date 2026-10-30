import unittest
from types import SimpleNamespace

import numpy as np

from ai.losses import MyLoss2Pairwise
from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config
from cube.search3_engine import Node, Search3Engine
from managers.search_data import SearchDataManager
from model.search_result import data_search3


class Search3PairwiseTests(unittest.TestCase):
    def test_experiment_enables_pairwise_only_for_search3(self):
        config = build_experiment_frame_config()
        for mode, objective in zip(config.ai_search_modes, config.search3_value_loss_types):
            self.assertEqual(objective, 'pairwise' if mode == 'search3' else 'absolute')

    def test_pairwise_loss_uses_each_adjacent_step_and_route_weight(self):
        loss = MyLoss2Pairwise(margin=1.0)
        values = np.array([[0.0, 0.5, 1.0, 10.0, 11.0]],dtype='f')
        weights = np.array([[2.0, 2.0, 2.0, 0.25, 0.25]],dtype='f')
        actual = loss.forward(values,[0,3,5],weights)
        expected = 4.0 * np.logaddexp(0.0,0.5) + 0.25 * np.log(2.0)
        self.assertAlmostEqual(actual,expected,places=5)
        self.assertAlmostEqual(float(np.sum(loss.backward())),0.0,places=5)

    def test_pairwise_search_uses_raw_value_and_no_depth_discount(self):
        ai = SimpleNamespace(search3_value_loss_type='pairwise',value_target_gamma=0.5,search3_C=0.5)
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = ai
        engine.node_cache = {}
        node = engine._create_node('root',np.array([0.5,0.5],dtype='f'),-3.0)
        self.assertEqual(node.value,-3.0)
        node.S = 1
        node.select_node(C=0.0)
        np.testing.assert_allclose(node.score,[-3.0,-3.0])
        self.assertEqual(engine._dead_end_value(node),-4.0)
        node.visited[0] = 1
        node.S = 1
        engine._backup_path([node],[0],-2.0)
        self.assertEqual(node.val[0],-2.0)

    def test_reused_root_refreshes_its_raw_unvisited_value(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        engine = ai.search3_engine
        engine._ensure_root_node(C=0.5)
        output = np.zeros((ai.ops,1),dtype='f')
        output[-1,0] = 2.5
        engine._predict_current_state = lambda: output
        self.assertEqual(engine._ensure_root_node(C=0.5),2.5)
        self.assertEqual(engine.root_node.value,2.5)

    def test_numpy_and_torch_pairwise_loss_match(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        out = np.zeros((ai.ops,5),dtype='f')
        out[-1] = [0.0,0.5,1.0,10.0,11.0]
        weights = np.array([[2.0,2.0,2.0,0.25,0.25]],dtype='f')
        targets = np.zeros((1,5),dtype='f')
        policy_targets = np.zeros((ai.ops - 1,5),dtype='f')
        policy_targets[0] = 1.0
        value_loss = ai._compute_search3_losses(
            out,policy_targets,targets,weights,[0,3,5],weights,
        )[1]
        self.assertAlmostEqual(value_loss,ai.search3_pairwise_loss.forward(out[-1:],[0,3,5],weights),places=5)
        try:
            import torch
        except ImportError:
            return
        value_tensor = torch.tensor(out[-1],dtype=torch.float32)
        weight_tensor = torch.tensor(weights[0],dtype=torch.float32)
        torch_loss = sum(
            torch.sum(weight_tensor[start:end - 1] * torch.nn.functional.softplus(
                1.0 - (value_tensor[start + 1:end] - value_tensor[start:end - 1]),
            ))
            for start,end in [(0,3),(3,5)]
        )
        self.assertAlmostEqual(value_loss,float(torch_loss),places=5)

    def test_connected_replay_compares_across_segments_and_keeps_policy_targets(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        first_policy = np.zeros(len(ai.cube.move_keys),dtype='f')
        second_policy = np.zeros_like(first_policy)
        first_policy[ai.cube.key_to_num[" R'"]] = 1.0
        second_policy[ai.cube.key_to_num[" U'"]] = 1.0
        first = data_search3(
            (' U ',' R '),(" R'",),np.zeros(2),0.0,[],0.0,{},
            policy_target=first_policy,search_mode='search3',sample_weight=1.0,
        )
        second = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            policy_target=second_policy,search_mode='search3',sample_weight=0.25,
        )
        manager = SearchDataManager(SimpleNamespace())
        connected = manager.build_connected_pairwise_search3_sample([first,second],ai)
        inputs = ai._build_search3_loss_inputs([connected])
        self.assertEqual(connected.moves,(" R'"," U'"))
        self.assertEqual(inputs['value_indices'],[0,3])
        np.testing.assert_allclose(inputs['sample_weights'][0],[1.0,0.25,0.25])
        np.testing.assert_allclose(inputs['policy_targets'][:,0],first_policy)
        np.testing.assert_allclose(inputs['policy_targets'][:,1],second_policy)

    def test_goal_is_reported_as_success_without_a_numeric_reward(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        ai.cube.scramble(0,(' U ',))
        engine = ai.search3_engine
        engine._ensure_root_node(C=0.5)
        correct = ai.cube.key_to_num[" U'"]
        engine.root_node.P[:] = 0.0
        engine.root_node.P[correct] = 1.0
        result = engine.search3(1,C=0.5)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.end_reason,'solved')
        self.assertEqual(result.moves,(" U'",))
        self.assertAlmostEqual(result.best_value,result.best_value_raw)


if __name__ == '__main__':
    unittest.main()

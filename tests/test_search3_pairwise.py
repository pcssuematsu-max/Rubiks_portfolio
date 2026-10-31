import unittest
from types import SimpleNamespace

import numpy as np

from ai.losses import MyLoss2Pairwise, RoutePairHuber, adjacent_mirror_pairs
from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config
from cube.search3_engine import Node, Search3Engine
from managers.search_data import SearchDataManager
from managers.solve_session import SolveSessionManager
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

    def test_adjacent_mirror_pairs_are_linear_size_and_not_duplicated(self):
        left,right = adjacent_mirror_pairs(0,6)
        self.assertEqual(list(zip(left,right)),[
            (0,1),(1,2),(2,3),(3,4),(4,5),(0,5),(1,4),
        ])

    def test_route_pair_huber_trains_noncentral_states_and_has_finite_target(self):
        loss = RoutePairHuber(margin=1.0)
        flat = np.zeros((1,6),dtype='f')
        flat_loss = loss.forward(flat,[0,6])
        gradient = loss.backward()[0]
        self.assertGreater(gradient[1],0.0)
        self.assertLess(gradient[4],0.0)
        self.assertAlmostEqual(float(np.sum(gradient)),0.0,places=6)
        self.assertAlmostEqual(flat_loss,loss.forward(flat + 123.0,[0,6]),places=5)
        self.assertAlmostEqual(loss.forward(np.arange(6,dtype='f').reshape(1,-1),[0,6]),0.0,places=6)
        self.assertGreater(loss.forward((2.0 * np.arange(6,dtype='f')).reshape(1,-1),[0,6]),0.0)

    def test_route_pair_huber_gradient_matches_finite_difference(self):
        loss = RoutePairHuber(margin=1.0)
        values = np.array([[0.2,-0.1,0.5,1.1,0.4,0.8]],dtype='d')
        weights = np.array([[1.0,0.7,1.3,0.4,1.0,0.6]],dtype='d')
        loss.forward(values,[0,4,6],weights)
        gradient = loss.backward().copy()
        epsilon = 1.0e-6
        for index in range(values.shape[1]):
            plus = values.copy()
            minus = values.copy()
            plus[0,index] += epsilon
            minus[0,index] -= epsilon
            numerical = (
                loss.forward(plus,[0,4,6],weights)
                - loss.forward(minus,[0,4,6],weights)
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[0,index]),float(numerical),places=5)

    def test_search2_pairwise_uses_the_same_huber_pairs_in_numpy_and_torch(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2',
                         search2_value_loss_type='myloss2_pairwise')
        ai.set_search2_value_loss_margin(1.0)
        self.assertIsInstance(ai.losslayer2,RoutePairHuber)
        values = np.array([[0.4,0.8,1.1,1.6,2.0,2.5]],dtype='f')
        numpy_loss = ai.losslayer2.forward(values,[0,6])
        numpy_gradient = ai.losslayer2.backward().copy()
        try:
            import torch
        except ImportError:
            return
        torch_values = torch.tensor(values[0],requires_grad=True)
        torch_loss = ai._torch_search2_value_loss_pairwise(torch_values,[0,6])
        torch_loss.backward()
        self.assertAlmostEqual(numpy_loss,float(torch_loss.detach()),places=5)
        np.testing.assert_allclose(numpy_gradient[0],torch_values.grad.numpy(),atol=1.0e-6)

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

    def test_numpy_and_torch_route_pair_huber_match(self):
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
        quality = ai._search3_quality_metrics(out,{
            'value_objective':'pairwise','pair_margin':1.0,
            'value_targets':targets,'sample_weights':weights,
            'policy_targets':policy_targets,'policy_weights':weights,
            'value_indices':[0,3,5],
        })
        self.assertAlmostEqual(quality['pair_huber_loss_sum'],value_loss,places=5)
        try:
            import torch
        except ImportError:
            return
        ai._torch_training_device = lambda: 'cpu'
        ai._torch_trainable_params = lambda device: {
            'fake': torch.nn.Parameter(torch.tensor(out,dtype=torch.float32)),
        }
        ai._torch_forward_policy_value = lambda features,params: params['fake']
        torch_losses,torch_grads,_ = ai._torch_search3_losses_and_grads({
            'x': np.zeros((ai.ips,5),dtype='f'),
            'policy_targets': policy_targets,
            'value_targets': targets,
            'sample_weights': weights,
            'policy_weights': weights,
            'value_indices': [0,3,5],
        })
        self.assertAlmostEqual(value_loss,torch_losses[1],places=5)
        np.testing.assert_allclose(
            ai.search3_pairwise_loss.backward(),torch_grads['fake'][-1:],atol=1.0e-5,
        )

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
        np.testing.assert_allclose(inputs['sample_weights'][0],[1.0,1.0,1.0])
        np.testing.assert_allclose(inputs['policy_targets'][:,0],first_policy)
        np.testing.assert_allclose(inputs['policy_targets'][:,1],second_policy)

    def test_connected_replay_removes_return_to_same_state(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        loop = (' R ',' U '," R'",' R '," U'"," R'")
        policy = np.zeros(len(ai.cube.move_keys),dtype='f')
        policy[ai.cube.key_to_num[" U'"]] = 1.0
        first = data_search3((' U ',),loop,np.zeros(7),0.0,[],0.0,{},
                             search_mode='search3',source_objective='search3')
        second = data_search3((' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
                              policy_target=policy,search_mode='search3')
        manager = SearchDataManager(SimpleNamespace())
        before = tuple(ai.cube.state)
        connected = manager.build_connected_pairwise_search3_sample([first,second],ai)
        self.assertEqual(tuple(ai.cube.state),before)
        self.assertEqual(connected.moves,(" U'",))
        self.assertEqual(connected.stats['removedRevisitedMoves'],6)
        np.testing.assert_allclose(connected.policy_targets_by_step[0],policy)

    def test_distinct_pairwise_routes_may_join_at_goal_without_a_cycle(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        manager = SearchDataManager(SimpleNamespace())
        for scramble,solution in [((' U ',),(" U'",)),((' R ',),(" R'",))]:
            kept,indices = manager.erase_revisited_states(ai.cube,scramble,solution)
            self.assertEqual(kept,solution)
            self.assertEqual(indices,(0,))

    def test_progressive_fallback_stores_only_greedy_route(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        partial = data_search3(
            (' U ',' R '),(" R'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='search3',source_objective='search3',
            trajectory_source='direct-search',
        )
        prefix = data_search3(
            (' U ',),(" R "," R'"),np.zeros(3),0.0,[],0.0,{},
            search_mode='search3',source_objective='search3',
            trajectory_source='search3-fallback-prefix',
        )
        fallback = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='myval',source_objective='search3',
            trajectory_source='greedy-fallback',
        )
        state = SimpleNamespace(
            fallback_used=True,
            search_history=[
                {'search_result':SimpleNamespace(search_mode='search3')},
                {'search_result':SimpleNamespace(search_mode='search3')},
                {'search_result':SimpleNamespace(search_mode='myval')},
            ],
            _search3_training_sample_cache={0:partial,1:prefix,2:fallback},
        )
        manager = SearchDataManager(SimpleNamespace(AI_idx=0,AIs=[ai],solve_state=state))
        manager.store_search3_data(0)
        self.assertEqual(len(ai.datas_search3),1)
        saved = ai.datas_search3[0]
        self.assertEqual(saved.scramble,(' U ',))
        self.assertEqual(saved.moves,(" U'",))
        inputs = ai._build_search3_loss_inputs([saved])
        np.testing.assert_allclose(inputs['sample_weights'],[[1.0,1.0]])
        np.testing.assert_allclose(inputs['policy_weights'],[[1.0,0.0]])

    def test_old_mixed_replay_is_excluded_but_greedy_replay_remains(self):
        for origin in ('mixed','search3-fallback-prefix','fallback'):
            item = SimpleNamespace(trajectory_source=origin,search_mode='search3')
            self.assertTrue(Rubiks_3_AI._search3_replay_contains_fallback_prefix(item))
        for origin,mode in (('greedy-fallback','myval'),('direct-search','search3')):
            item = SimpleNamespace(trajectory_source=origin,search_mode=mode)
            self.assertFalse(Rubiks_3_AI._search3_replay_contains_fallback_prefix(item))

    def test_learning_retires_saved_mixed_routes(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.datas_search3 = [
            SimpleNamespace(trajectory_source='mixed',search_mode='search3',moves=('a',)),
            SimpleNamespace(trajectory_source='greedy-fallback',search_mode='myval',moves=('b',)),
            SimpleNamespace(trajectory_source='direct-search',search_mode='search3',moves=('c',)),
        ]
        ai.indices_search3 = [0,1,2]
        seen = []
        def record_selection(**kwargs):
            seen.extend(kwargs['indices'])
            return 0.0,0.0,list(kwargs['indices']),len(kwargs['indices']),0,0.0
        ai._run_training_epochs = record_selection
        ai._record_training_metrics = lambda *args: None
        ai.learn_search3()
        self.assertEqual(seen,[1,2])
        self.assertEqual(ai.indices_search3,[1,2])

    def test_pairwise_search2_registration_erases_nonlocal_state_loop(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2')
        pairwise = SimpleNamespace(search2_value_loss_type='myloss2_pairwise',indices=[])
        frame = SimpleNamespace(
            cube=ai.cube,AIs=[ai,pairwise],AInum=2,AI_idx=0,
            solve_state=SimpleNamespace(search_TF=True,fallback_used=False,
                                        last_perfect_key=None,last_top_group=None),
        )
        frame.search_data_manager = SearchDataManager(frame)
        session = SolveSessionManager(frame)
        loop = (' R ',' U '," R'"," U'") * 6
        self.assertEqual(len(ai.cube.simplify(loop)),24)
        self.assertTrue(session._store_search2_training_sample(
            (' U ',),loop + (" U'",),target_ai_indices=[1],
        ))
        self.assertEqual(tuple(ai.datas[-1].moves),(" U'",))
        self.assertEqual(pairwise.indices,[len(ai.datas) - 1])

    def test_saved_replay_uses_uniform_fallback_weights(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        direct = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='search2',sample_weight=1.0,
            source_objective='original_search2',
        )
        fallback = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='myval',sample_weight=0.2,
        )
        bootstrap = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='bootstrap',sample_weight=0.1,
        )
        self.assertEqual(ai._search3_step_weight(direct,0),1.0)
        self.assertEqual(ai._search3_policy_step_weight(direct,0),1.0)
        self.assertEqual(ai._search3_step_weight(bootstrap,0),1.0)
        self.assertEqual(ai._search3_policy_step_weight(bootstrap,0),1.0)
        self.assertEqual(ai._search3_step_weight(fallback,0),1.0)
        self.assertEqual(ai._search3_policy_step_weight(fallback,0),1.0)

        # Previously saved connected Search3 routes retain fallback markers.
        connected = data_search3(
            (' U ',),(" U'"," R'"),np.zeros(3),0.0,[],0.0,{},
            search_mode='search3',sample_weight=1.0,
            source_objective='search3',trajectory_source='mixed',
        )
        connected.value_step_weights = (1.0,0.05,0.05)
        connected.policy_step_weights = (1.0,0.05)
        self.assertEqual(ai._search3_step_weight(connected,0),1.0)
        self.assertEqual(ai._search3_step_weight(connected,1),1.0)
        self.assertEqual(ai._search3_terminal_step_weight(connected),1.0)
        self.assertEqual(ai._search3_policy_step_weight(connected,0),1.0)
        self.assertEqual(ai._search3_policy_step_weight(connected,1),1.0)
        summary = ai._search3_training_weight_summary([0],[connected])
        totals = next(iter(summary.values()))
        self.assertAlmostEqual(totals['valueWeightSum'],3.0)
        self.assertAlmostEqual(totals['policyWeightSum'],2.0)
        connected.value_step_weights = (1.0,0.2,0.2)
        connected.policy_step_weights = (1.0,0.2)
        self.assertEqual(ai._search3_step_weight(connected,1),1.0)
        self.assertEqual(ai._search3_policy_step_weight(connected,1),1.0)

        old_search2_direct = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='search2',source_objective='original_search2',
            trajectory_source='direct-search',
        )
        old_search2_direct.value_step_weights = (0.2,0.2)
        self.assertEqual(ai._search3_step_weight(old_search2_direct,0),1.0)

        # Connected routes saved under the uniform-weight version have no
        # segment boundary. Treat the whole mixed route as fallback-derived.
        legacy_mixed = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='search3',trajectory_source='mixed',
            source_objective='search3',
        )
        self.assertEqual(ai._search3_step_weight(legacy_mixed,0),1.0)
        self.assertEqual(ai._search3_policy_step_weight(legacy_mixed,0),1.0)

    def test_new_connected_route_preserves_direct_and_fallback_weights(self):
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search3')
        ai.set_search3_value_loss_type('pairwise')
        first = data_search3(
            (' U ',' R '),(" R'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='search3',source_objective='search3',
            trajectory_source='search3-fallback-prefix',
        )
        fallback = data_search3(
            (' U ',),(" U'",),np.zeros(2),0.0,[],0.0,{},
            search_mode='myval',source_objective='search3',
            trajectory_source='greedy-fallback',
        )
        connected = SearchDataManager(SimpleNamespace()).build_connected_pairwise_search3_sample(
            [first,fallback],ai,
        )
        inputs = ai._build_search3_loss_inputs([connected])
        np.testing.assert_allclose(inputs['sample_weights'][0],[1.0,1.0,1.0])
        np.testing.assert_allclose(inputs['policy_weights'][0],[1.0,1.0,0.0])

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

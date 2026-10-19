from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path

import numpy as np

from ai.losses import Huber, MyLoss2Pairwise
from ai.rubiks_ai import Rubiks_3_AI
from configs.profiles import build_experiment_frame_config
from core.learning_history import LearningHistoryStore, completed_learning_record
from managers.search_data import SearchDataManager
from managers.solve_session import SolveSessionManager
from model.search_result import data


class StepsToGoalValueTests(unittest.TestCase):
    def test_steps_pairwise_treatment_leaves_reference_ais_unchanged(self):
        config = build_experiment_frame_config()
        self.assertEqual(config.search2_value_loss_types[22], 'steps_to_goal')
        self.assertEqual(config.search2_rank_loss_apply_types[22], 'steps_pairwise')
        self.assertEqual(config.search2_rank_loss_mixes[22], 1.0)
        self.assertEqual(config.search2_value_loss_margins[22], 1.0)
        self.assertEqual(config.steps_to_goal_plateau_fallback_min_ratios[22], 0.05)
        self.assertEqual(config.steps_to_goal_plateau_escape_rank_mixes[22], 1.0)
        self.assertEqual(config.steps_to_goal_plateau_escape_prefix_steps[22], 3)
        for index in (20,21,23,24):
            self.assertEqual(config.search2_rank_loss_mixes[index], 0.0)
            self.assertEqual(config.steps_to_goal_plateau_escape_rank_mixes[index], 0.0)

    def test_relative_loss_is_route_normalized_and_has_correct_gradient(self):
        loss = MyLoss2Pairwise(margin=1.0,reduction='trajectory_mean')
        values = np.asarray([[0.0,0.0,0.0,0.0,0.0]],dtype='f')
        result = loss.forward(values,[0,2,5])
        self.assertAlmostEqual(result,2.0 * np.logaddexp(0.0,1.0),places=6)
        gradient = loss.backward()
        self.assertAlmostEqual(float(np.sum(gradient)),0.0,places=6)
        self.assertLess(float(gradient[0,-1]),0.0)

    def test_torch_relative_loss_matches_numpy_route_normalization(self):
        from ai import rubiks_ai as module
        if module.torch is None:
            self.skipTest('Torch unavailable')
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_margin = 1.0
        values = module.torch.tensor([0.0,0.0,0.0,0.0,0.0],requires_grad=True)
        result = ai._torch_search2_value_loss_pairwise(
            values,[0,2,5],reduction='trajectory_mean',
        )
        result.backward()
        numpy_loss = MyLoss2Pairwise(margin=1.0,reduction='trajectory_mean')
        self.assertAlmostEqual(result.item(),numpy_loss.forward(
            np.zeros((1,5),dtype='f'),[0,2,5],
        ),places=6)
        np.testing.assert_allclose(values.grad.numpy(),numpy_loss.backward().reshape(-1),atol=1e-6)

    def test_steps_to_goal_keeps_absolute_target_and_adds_relative_loss(self):
        from ai import rubiks_ai as module
        if module.torch is None:
            self.skipTest('Torch unavailable')
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2',
                         search2_value_loss_type='steps_to_goal')
        ai.set_search2_value_loss_margin(1.0)
        ai.set_search2_rank_loss_apply_type('steps_pairwise')
        ai.set_search2_rank_loss_mix(1.0)
        values = module.torch.tensor([-2.0,-1.0,0.0],requires_grad=True)
        result = ai._torch_search2_value_loss(
            values,[0,3],value_steps_to_goal=[2,1,0],
            value_trajectory_origins=['direct_search'],
        )
        result.backward()
        components = ai._last_search2_value_loss_components
        self.assertEqual(components['base_name'],'steps_to_goal_huber')
        self.assertAlmostEqual(components['base'],0.0,places=6)
        self.assertTrue(components['rank_applied'])
        self.assertAlmostEqual(components['rank_raw'],np.log(2.0),places=6)
        self.assertLess(float(values.grad[-1]),0.0)

    def test_relative_treatment_and_loss_are_persisted_in_learning_history(self):
        ai = SimpleNamespace(
            search_mode='search2', search2_value_loss_type='steps_to_goal',
            search2_value_loss_margin=1.0, search2_rank_loss_mix=1.0,
            search2_rank_loss_apply_type='steps_pairwise',search3_rank_loss_mix=0.0,
            last_training_metrics={
                'updatesDuringSolve':1,'search2BaseLossPerItem':2.0,
                'search2RankLossPerItem':0.5,
                'search2PlateauEscapeLossPerItem':0.25,
            },
        )
        record = completed_learning_record(22,ai,0.1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LearningHistoryStore(Path(temporary_directory) / 'history.json')
            store.append(record)
            saved = store.records()[0]
        self.assertEqual(saved['search2RankLossApplyType'],'steps_pairwise')
        self.assertEqual(saved['search2RankLossMix'],1.0)
        self.assertEqual(saved['search2RankLossPerItem'],0.5)
        self.assertEqual(saved['search2PlateauEscapeLossPerItem'],0.25)

    def test_plateau_escape_prefix_has_independent_relative_gradient(self):
        from ai import rubiks_ai as module
        if module.torch is None:
            self.skipTest('Torch unavailable')
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_plateau_escape_rank_mix = 1.0
        ai.steps_to_goal_plateau_escape_prefix_steps = 3
        ai.search2_value_loss_margin = 1.0
        stalled = SimpleNamespace(
            plateau_escape=True, trajectory_origin='fallback', moves=('A','B','C','D'),
        )
        ordinary = SimpleNamespace(
            plateau_escape=False, trajectory_origin='direct_search', moves=('E',),
        )
        groups = ai._plateau_escape_pairs([stalled,ordinary],[0,5,7])
        self.assertEqual(groups,[[(0,1),(1,2),(2,3)]])
        values = module.torch.zeros(7,requires_grad=True)
        loss = ai._torch_plateau_escape_loss(values,groups)
        loss.backward()
        self.assertAlmostEqual(loss.item(),np.logaddexp(0.0,1.0),places=6)
        self.assertGreater(values.grad[0].item(),0.0)
        self.assertLess(values.grad[3].item(),0.0)
        self.assertEqual(values.grad[5].item(),0.0)

    def test_plateau_escape_still_trains_when_absolute_route_is_filtered(self):
        from ai import rubiks_ai as module
        if module.torch is None:
            self.skipTest('Torch unavailable')
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2',
                         search2_value_loss_type='steps_to_goal')
        ai.steps_to_goal_plateau_escape_rank_mix = 1.0
        ai.search2_value_loss_margin = 1.0
        values = module.torch.zeros(2,requires_grad=True)
        loss = ai._torch_search2_value_loss(
            values,[0],value_columns=[],plateau_escape_pairs=[[(0,1)]],
        )
        loss.backward()
        self.assertGreater(loss.item(),0.0)
        self.assertGreater(values.grad[0].item(),0.0)
        self.assertLess(values.grad[1].item(),0.0)

    def test_numpy_plateau_escape_gradient_matches_torch_direction(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.steps_to_goal_plateau_escape_rank_mix = 1.0
        ai.search2_value_loss_margin = 1.0
        values = np.zeros((1,2),dtype='f')
        loss = ai._numpy_plateau_escape_loss(values,[[(0,1)]])
        self.assertAlmostEqual(loss,np.logaddexp(0.0,1.0),places=6)
        self.assertGreater(ai._plateau_escape_gradient[0,0],0.0)
        self.assertLess(ai._plateau_escape_gradient[0,1],0.0)

    def test_plateau_escape_runs_through_search2_torch_training_batch(self):
        from ai import rubiks_ai as module
        if module.torch is None:
            self.skipTest('Torch unavailable')
        ai = Rubiks_3_AI([8],cube_size=3,search_mode='search2',
                         search2_value_loss_type='steps_to_goal')
        ai.steps_to_goal_plateau_escape_rank_mix = 1.0
        ai.steps_to_goal_plateau_escape_prefix_steps = 3
        ai.search2_value_loss_margin = 1.0
        scramble = (ai.cube.move_keys[0],ai.cube.move_keys[3])
        moves = tuple(ai.cube.invert_moves(scramble))
        sample = data(
            scramble,moves,np.zeros(len(moves),dtype='f'),
            source_search2_value_loss_type='steps_to_goal_regression',
            trajectory_origin='fallback',plateau_escape=True,
        )
        inputs = ai._build_loss_inputs([sample])
        self.assertEqual(len(inputs['plateau_escape_pairs']),1)
        losses,gradients = ai._torch_search2_losses_and_grads(inputs)
        self.assertTrue(all(np.isfinite(value) for value in losses))
        self.assertTrue(gradients)
        self.assertGreater(ai._last_search2_value_loss_components['plateau_escape_scaled'],0.0)

    def test_plateau_fallback_replay_does_not_replace_direct_routes(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_plateau_fallback_min_ratio = 0.25
        ai.cube_size = 3
        items = [
            SimpleNamespace(plateau_escape=False,trajectory_origin='direct_search',moves=('A',)),
            SimpleNamespace(plateau_escape=False,trajectory_origin='legacy',moves=('B',)),
            SimpleNamespace(plateau_escape=True,trajectory_origin='fallback',moves=('C',)),
        ]
        batches = [[0,1],[0,1]]
        summary = ai._enforce_steps_to_goal_plateau_fallback_minimum(
            batches,list(range(len(items))),items,
        )
        self.assertEqual(summary['selected_item_count'],1)
        self.assertEqual(sum(index == 0 for batch in batches for index in batch),2)
        self.assertEqual(sum(index == 2 for batch in batches for index in batch),1)

    def test_only_absolute_value_objectives_train_the_output_bias(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search_mode = 'search2'
        ai.search2_value_loss_type = 'steps_to_goal'
        self.assertTrue(ai._is_value_output_bias_trainable())
        self.assertFalse(ai._is_non_trainable_param('BO_V'))

        ai.search2_value_loss_type = 'myloss'
        self.assertFalse(ai._is_value_output_bias_trainable())
        self.assertTrue(ai._is_non_trainable_param('BO_V'))

        ai.search2_value_loss_type = 'myloss2_pairwise'
        self.assertFalse(ai._is_value_output_bias_trainable())
        ai.search_mode = 'search3'
        self.assertTrue(ai._is_value_output_bias_trainable())

    def test_steps_to_goal_bias_moves_toward_target_under_torch_grad_update(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search_mode = 'search2'
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.params = {'BO_V': np.asarray([-10.0],dtype='f')}
        ai.v = {'BO_V': np.zeros(1,dtype='f')}
        ai.h = {'BO_V': np.zeros(1,dtype='f')}
        ai.lr = 0.1
        ai.lr_v = 0.0
        ai.lr_h = 0.0
        ai.adam = False
        ai.weight_decay = False
        ai.update_scale_value = 1.0
        ai._mark_torch_params_dirty = lambda: None

        ai._apply_torch_grad_updates({'BO_V': np.asarray([-1.0],dtype='f')})

        self.assertAlmostEqual(float(ai.params['BO_V'][0]),-9.9,places=5)

    def test_huber_is_quadratic_near_target_and_clipped_for_large_errors(self):
        loss = Huber()
        prediction = np.asarray([[-3.0, 2.0]], dtype = 'f')
        target = np.asarray([[-3.0, -1.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target), 2.5)
        np.testing.assert_allclose(loss.backward(), [[0.0, 1.0]])

    def test_huber_mean_reduction_keeps_longer_routes_from_scaling_one_update(self):
        loss = Huber(reduction = 'mean')
        prediction = np.asarray([[-3.0, 2.0]], dtype = 'f')
        target = np.asarray([[-3.0, -1.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target), 1.25)
        np.testing.assert_allclose(loss.backward(), [[0.0, 0.5]])

    def test_huber_trajectory_sum_normalizes_each_route_not_whole_batch(self):
        loss = Huber(reduction = 'trajectory_sum')
        prediction = np.asarray([[2.0, 2.0, 2.0]], dtype = 'f')
        target = np.asarray([[0.0, 0.0, 0.0]], dtype = 'f')

        self.assertAlmostEqual(loss.forward(prediction, target, [0, 1, 3]), 3.0)
        np.testing.assert_allclose(loss.backward(), [[1.0, 0.5, 0.5]])

    def test_steps_to_goal_value_selection_balances_remaining_step_bands(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_states_per_band = 2
        item = SimpleNamespace(steps_to_goal = tuple(range(40,-1,-1)))

        columns = ai._steps_to_goal_value_columns_for_route(item,10,51)
        selected_steps = [item.steps_to_goal[column - 10] for column in columns]

        self.assertLessEqual(sum(step <= 10 for step in selected_steps), 2)
        self.assertLessEqual(sum(11 <= step <= 30 for step in selected_steps), 2)
        self.assertLessEqual(sum(step >= 31 for step in selected_steps), 2)
        self.assertEqual(len(columns), 6)

    def test_steps_to_goal_value_uses_only_shortest_duplicate_start_target(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        short = SimpleNamespace(scramble = ('R',), moves = ('A', 'B'))
        long = SimpleNamespace(scramble = ('R',), moves = ('A', 'B', 'C', 'D'))

        ai._prepare_steps_to_goal_shortest_targets([0,1], [short,long])

        self.assertTrue(ai._uses_shortest_steps_to_goal_target(short))
        self.assertFalse(ai._uses_shortest_steps_to_goal_target(long))

    def test_steps_to_goal_value_excludes_only_strongly_detoured_routes(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.steps_to_goal_value_max_route_efficiency = 3.0

        self.assertTrue(ai._uses_steps_to_goal_value_route(
            SimpleNamespace(scramble = ('R', 'U'), moves = tuple(range(6))),
        ))
        self.assertFalse(ai._uses_steps_to_goal_value_route(
            SimpleNamespace(scramble = ('R', 'U'), moves = tuple(range(7))),
        ))

    def test_steps_to_goal_value_balances_bands_across_one_batch(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_states_per_band = 0
        ai.steps_to_goal_value_batch_band_max_copies = 3
        ai.steps_to_goal_value_max_route_efficiency = 0.0
        ai._steps_to_goal_shortest_target_by_start = None
        long_route = SimpleNamespace(
            moves = tuple(range(40)), steps_to_goal = tuple(range(40,-1,-1)),
            trajectory_origin = 'direct_search',
        )
        short_route = SimpleNamespace(
            moves = tuple(range(10)), steps_to_goal = tuple(range(10,-1,-1)),
            trajectory_origin = 'direct_search',
        )

        columns,value_indices,_ = ai._build_search2_value_loss_selection(
            [long_route,short_route], [0,41,52],
        )

        self.assertEqual(value_indices[-1], len(columns))
        summary = ai._last_steps_to_goal_value_selection
        self.assertEqual(summary['candidate_state_bands'], {'0-10': 22, '11-30': 20, '31+': 10})
        self.assertEqual(summary['selected_state_bands'], {'0-10': 22, '11-30': 22, '31+': 22})
        self.assertEqual(summary['oversampled_state_count'], 14)

    def test_repeated_intermediate_state_keeps_shortest_value_label_only(self):
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)
        ai.search2_value_loss_type = 'steps_to_goal'
        ai.steps_to_goal_states_per_band = 0
        ai.steps_to_goal_value_batch_band_max_copies = 1
        ai.steps_to_goal_value_max_route_efficiency = 0.0
        ai._steps_to_goal_shortest_target_by_start = None
        routes = [
            SimpleNamespace(moves = ('a','b','c'), steps_to_goal = (3,2,1,0), trajectory_origin = 'direct_search'),
            SimpleNamespace(moves = ('d',), steps_to_goal = (1,0), trajectory_origin = 'direct_search'),
        ]
        states = np.asarray([[1,2,3,4,2,4]],dtype = 'f')

        columns,indices,_ = ai._build_search2_value_loss_selection(
            routes,[0,4,6],x = states,
        )

        self.assertNotIn(1,columns.tolist())
        self.assertIn(4,columns.tolist())
        self.assertEqual(indices[-1],len(columns))
        self.assertEqual(ai._last_steps_to_goal_value_selection['conflicting_intermediate_state_count'],1)
        self.assertEqual(ai._last_steps_to_goal_value_selection['discarded_nonshortest_state_count'],1)

    def test_replay_wide_shortest_target_applies_across_batches_and_restores_cube(self):
        ai = Rubiks_3_AI([8],cube_size = 3,search_mode = 'search2')
        ai.set_search2_value_loss_type('steps_to_goal')
        ai.steps_to_goal_states_per_band = 0
        ai.steps_to_goal_value_batch_band_max_copies = 1
        ai.steps_to_goal_value_max_route_efficiency = 0.0
        up,down = ai.cube.move_keys[0],ai.cube.move_keys[3]
        up_inverse = ai.cube.invert_moves((up,))[0]
        down_inverse = ai.cube.invert_moves((down,))[0]
        short = SimpleNamespace(
            scramble = (up,),moves = (up_inverse,),
            steps_to_goal = (1,0),trajectory_origin = 'direct_search',
            source_search2_value_loss_type = 'steps_to_goal_regression',
        )
        long = SimpleNamespace(
            scramble = (up,down),
            moves = (down_inverse,down,down_inverse,up_inverse),
            steps_to_goal = (4,3,2,1,0),trajectory_origin = 'direct_search',
            source_search2_value_loss_type = 'steps_to_goal_regression',
        )
        ai.cube.scramble(2)
        original_state = ai.cube.state.copy()

        ai._prepare_steps_to_goal_shortest_targets([0,1],[short,long])

        np.testing.assert_array_equal(ai.cube.state,original_state)
        x = np.empty((ai.cube.makedata().size,5),dtype = 'f')
        ai._fill_loss_tensors([long],0,False,np.empty(4,dtype = 'i'),x)
        columns,_,_ = ai._build_search2_value_loss_selection([long],[0,5],x = x)

        self.assertNotIn(1,columns.tolist())
        self.assertIn(3,columns.tolist())
        self.assertGreaterEqual(
            ai._last_steps_to_goal_value_selection['discarded_nonshortest_state_count'],1,
        )

    def test_steps_to_goal_direct_route_also_teaches_original_with_full_route(self):
        shared_data = []
        steps = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'steps_to_goal',
            datas = shared_data, indices = [],
        )
        original = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'myloss',
            datas = shared_data, indices = [],
        )
        frame = SimpleNamespace(
            solve_state = SimpleNamespace(
                s = ('scramble',), move_lis = [('A', 'B'), ('C',)],
                search_TF = True, fallback_used = False,
                last_perfect_key = '', last_top_group = None,
            ),
            cube = SimpleNamespace(
                simplify = lambda moves: tuple(moves),
                make_transformations = lambda scramble, moves: ((tuple(scramble),), (tuple(moves),)),
            ),
            AIs = [steps, original], AI_idx = 0, AInum = 2,
        )

        self.assertTrue(
            SolveSessionManager(frame)._store_connected_original_search2_training_sample(steps),
        )
        self.assertEqual(shared_data[0].moves, ('A', 'B', 'C'))
        self.assertEqual(shared_data[0].source_search2_value_loss_type, 'steps_to_goal_connected')
        self.assertEqual(original.indices, [0])

    def test_pairwise_direct_route_also_teaches_original_with_full_route(self):
        shared_data = []
        pairwise = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'myloss2_pairwise',
            datas = shared_data, indices = [],
        )
        original = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'myloss',
            datas = shared_data, indices = [],
        )
        frame = SimpleNamespace(
            solve_state = SimpleNamespace(
                s = ('scramble',), move_lis = [('A', 'B'), ('C',)],
                search_TF = True, fallback_used = False,
                last_perfect_key = '', last_top_group = None,
            ),
            cube = SimpleNamespace(
                simplify = lambda moves: tuple(moves),
                make_transformations = lambda scramble, moves: ((tuple(scramble),), (tuple(moves),)),
            ),
            AIs = [pairwise, original], AI_idx = 0, AInum = 2,
        )

        self.assertTrue(SolveSessionManager(frame)._store_connected_myloss_training_sample())
        self.assertEqual(shared_data[0].moves, ('A', 'B', 'C'))
        self.assertEqual(shared_data[0].source_search2_value_loss_type, 'myloss_connected')
        self.assertEqual(original.indices, [0])

    def test_pairwise_direct_route_registers_only_one_connected_original_sample(self):
        for segments in ((('A', 'B'),), (('A', 'B'), ('C',))):
            with self.subTest(segments = segments):
                shared_data = []
                pairwise = SimpleNamespace(
                    search_mode = 'search2', search2_value_loss_type = 'myloss2_pairwise',
                    datas = shared_data, indices = [],
                )
                original = SimpleNamespace(
                    search_mode = 'search2', search2_value_loss_type = 'myloss',
                    datas = shared_data, indices = [],
                )
                steps = SimpleNamespace(
                    search_mode = 'search2', search2_value_loss_type = 'steps_to_goal',
                    datas = shared_data, indices = [],
                )
                state = SimpleNamespace(
                    s = ('scramble',), move_lis = [()] + list(segments),
                    search_TF = True, fallback_used = False,
                    last_perfect_key = '', last_top_group = None,
                )
                frame = SimpleNamespace(
                    solve_state = state,
                    cube = SimpleNamespace(
                        simplify = lambda moves: tuple(moves),
                        make_transformations = lambda scramble, moves: (
                            (tuple(scramble),), (tuple(moves),),
                        ),
                    ),
                    AIs = [pairwise, original, steps], AI_idx = 0, AInum = 3,
                )

                self.assertFalse(SolveSessionManager(frame)._store_completed_training_samples())
                connected_moves = tuple(move for segment in segments for move in segment)
                self.assertEqual(len(shared_data), 2)
                self.assertEqual([item.moves for item in shared_data], [connected_moves] * 2)
                self.assertEqual([item.scramble for item in shared_data], [('scramble',)] * 2)
                self.assertEqual(pairwise.indices, [0])
                self.assertEqual(steps.indices, [0])
                self.assertEqual(original.indices, [1])
                self.assertEqual(shared_data[1].source_search2_value_loss_type, 'myloss_connected')
                self.assertEqual(state.s, ('scramble',))

    def test_pairwise_fallback_still_registers_only_for_pairwise(self):
        shared_data = []
        pairwise = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'myloss2_pairwise',
            datas = shared_data, indices = [],
        )
        original = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'myloss',
            datas = shared_data, indices = [],
        )
        steps = SimpleNamespace(
            search_mode = 'search2', search2_value_loss_type = 'steps_to_goal',
            datas = shared_data, indices = [],
        )
        frame = SimpleNamespace(
            solve_state = SimpleNamespace(
                s = ('scramble',), move_lis = [(), ('A',)],
                search_TF = False, fallback_used = True,
                last_perfect_key = '', last_top_group = None,
            ),
            cube = SimpleNamespace(
                simplify = lambda moves: tuple(moves),
                make_transformations = lambda scramble, moves: (
                    (tuple(scramble),), (tuple(moves),),
                ),
            ),
            AIs = [pairwise, original, steps], AI_idx = 0, AInum = 3,
        )

        self.assertFalse(SolveSessionManager(frame)._store_completed_training_samples())
        self.assertEqual(len(shared_data), 1)
        self.assertEqual(pairwise.indices, [0])
        self.assertEqual(original.indices, [])
        self.assertEqual(steps.indices, [])

    def test_connected_search2_route_has_distances_to_the_final_goal(self):
        shared_data = []
        source_ai = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'myloss',
            datas = shared_data,
            indices = [],
        )
        regression_one = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'steps_to_goal',
            datas = shared_data,
            indices = [],
        )
        regression_two = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'steps_to_goal',
            datas = shared_data,
            indices = [],
        )
        pairwise = SimpleNamespace(
            search_mode = 'search2',
            search2_value_loss_type = 'myloss2_pairwise',
            datas = shared_data,
            indices = [],
        )
        cube = SimpleNamespace(
            simplify = lambda moves: tuple(moves),
            make_transformations = lambda scramble, moves: ((tuple(scramble),), (tuple(moves),)),
        )
        state = SimpleNamespace(
            s = ('scramble',),
            move_lis = [('A', 'B'), ('C', 'D', 'E')],
            search_TF = True,
            last_perfect_key = '',
            last_top_group = None,
        )
        frame = SimpleNamespace(
            solve_state = state,
            cube = cube,
            AIs = [source_ai, regression_one, regression_two, pairwise],
            AI_idx = 0,
            AInum = 4,
        )

        stored = SolveSessionManager(frame)._store_connected_steps_to_goal_training_sample()

        self.assertTrue(stored)
        self.assertEqual(len(shared_data), 1)
        sample = shared_data[0]
        self.assertEqual(sample.moves, ('A', 'B', 'C', 'D', 'E'))
        self.assertEqual(sample.steps_to_goal, (5, 4, 3, 2, 1, 0))
        self.assertEqual(sample.source_search2_value_loss_type, 'steps_to_goal_regression')
        self.assertEqual(sample.trajectory_origin, 'direct_search')
        self.assertEqual(source_ai.indices, [])
        self.assertEqual(regression_one.indices, [0])
        self.assertEqual(regression_two.indices, [0])
        self.assertEqual(pairwise.indices, [0])

    def test_stalled_steps_fallback_keeps_value_label_and_escape_marker(self):
        shared = []
        source = SimpleNamespace(
            search_mode='search2',search2_value_loss_type='steps_to_goal',
            steps_to_goal_plateau_escape_rank_mix=1.0,datas=shared,indices=[],
        )
        target = SimpleNamespace(
            search_mode='search2',search2_value_loss_type='steps_to_goal',
            datas=shared,indices=[],
        )
        cube = SimpleNamespace(
            simplify=lambda moves: tuple(moves),
            make_transformations=lambda scramble,moves: ((tuple(scramble),),(tuple(moves),)),
        )
        state = SimpleNamespace(
            s=('stalled',),move_lis=[('escape','goal')],search_TF=False,
            fallback_used=True,direct_search_attempts=[{'returnedMoveCount':0}],
            last_perfect_key='',last_top_group=None,
        )
        frame = SimpleNamespace(solve_state=state,cube=cube,AIs=[source,target],AI_idx=0,AInum=2)

        self.assertTrue(SolveSessionManager(frame)._store_connected_steps_to_goal_training_sample([1]))
        self.assertEqual(shared[0].trajectory_origin,'fallback')
        self.assertEqual(shared[0].source_search2_value_loss_type,'steps_to_goal_regression')
        self.assertTrue(shared[0].plateau_escape)
        self.assertEqual(target.indices,[0])
        control = Rubiks_3_AI.__new__(Rubiks_3_AI)
        control.search2_value_loss_type = 'steps_to_goal'
        control.steps_to_goal_plateau_escape_rank_mix = 0.0
        self.assertFalse(control._uses_search2_value_loss_for_sample(shared[0]))
        control.steps_to_goal_plateau_escape_rank_mix = 1.0
        control.steps_to_goal_value_max_route_efficiency = 0.0
        self.assertTrue(control._uses_search2_value_loss_for_sample(shared[0]))

    def test_direct_steps_to_goal_route_creates_one_connected_sample_per_target_kind(self):
        source = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'steps_to_goal')
        original = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'myloss')
        search3 = SimpleNamespace(search_mode = 'search3', search2_value_loss_type = 'myloss')
        state = SimpleNamespace(search_TF = True, fallback_used = False, move_lis = [('A',), ('B',)])
        calls = []
        frame = SimpleNamespace(
            solve_state = state,
            AIs = [source, original, search3],
            AI_idx = 0,
            AInum = 3,
            search_data_manager = SimpleNamespace(
                store_connected_search3_data = lambda ai: calls.append(('search3_connected', ai)),
            ),
        )
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda *args: calls.append(('steps_connected', args)) or True
        manager._store_connected_original_search2_training_sample = lambda ai: calls.append(('original_connected', ai)) or True
        manager._store_search2_segment_training_samples = lambda: calls.append('segments') or True

        self.assertTrue(manager._store_completed_training_samples())
        self.assertEqual(
            calls,
            [('steps_connected', ()), ('original_connected', source), ('search3_connected', source)],
        )

    def test_fallback_steps_to_goal_route_stays_in_steps_to_goal_only(self):
        source = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'steps_to_goal')
        state = SimpleNamespace(search_TF = False, fallback_used = True, move_lis = [('A',)])
        calls = []
        frame = SimpleNamespace(
            solve_state = state,
            AIs = [source],
            AI_idx = 0,
            AInum = 1,
            search_data_manager = SimpleNamespace(
                store_connected_search3_data = lambda ai: calls.append(('search3_connected', ai)),
            ),
        )
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda *args: calls.append(('steps_connected', args)) or True
        manager._store_connected_original_search2_training_sample = lambda ai: calls.append(('original_connected', ai)) or True

        self.assertFalse(manager._store_completed_training_samples())
        self.assertEqual(calls, [('steps_connected', ([0],))])

    def test_pairwise_fallback_stays_out_of_steps_to_goal_replay(self):
        pairwise_one = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'myloss2_pairwise')
        steps = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'steps_to_goal')
        pairwise_two = SimpleNamespace(search_mode = 'search2', search2_value_loss_type = 'myloss2_pairwise')
        frame = SimpleNamespace(AIs = [pairwise_one, steps, pairwise_two])
        manager = SolveSessionManager(frame)
        targets = []
        manager._store_search2_segment_training_samples = lambda indices: targets.append(indices) or True

        self.assertTrue(manager._store_search2_fallback_training_samples('myloss2_pairwise'))
        self.assertEqual(targets, [[0, 2]])

    def test_direct_search3_route_also_creates_a_connected_steps_to_goal_sample(self):
        source = SimpleNamespace(search_mode = 'search3', search2_value_loss_type = 'myloss')
        state = SimpleNamespace(search_TF = True, fallback_used = False, move_lis = [('A',), ('B',)])
        calls = []
        frame = SimpleNamespace(solve_state = state, AIs = [source], AI_idx = 0, AInum = 1)
        manager = SolveSessionManager(frame)
        manager._store_connected_steps_to_goal_training_sample = lambda *args: calls.append(('steps_connected', args)) or True
        manager._store_connected_myloss_training_sample = lambda: calls.append('pairwise_connected') or True
        manager._store_search2_segment_training_samples = lambda *args: calls.append(('original_segments', args)) or True

        self.assertFalse(manager._store_completed_training_samples())
        self.assertEqual(calls, [('steps_connected', ()), 'pairwise_connected', ('original_segments', ([],))])

    def test_connected_steps_route_adds_one_search3_sample_per_search3_ai(self):
        source = SimpleNamespace(search_mode = 'search2')
        original = SimpleNamespace(search_mode = 'search2', datas_search3 = [], indices_search3 = [])
        search3_one = SimpleNamespace(search_mode = 'search3', datas_search3 = [], indices_search3 = [])
        search3_two = SimpleNamespace(search_mode = 'transformer', datas_search3 = [], indices_search3 = [])
        frame = SimpleNamespace(
            solve_state = SimpleNamespace(s = ('scramble',), move_lis = [('A',), ('B',)]),
            cube = SimpleNamespace(simplify = lambda moves: tuple(moves)),
            AIs = [original, search3_one, search3_two],
        )
        manager = SearchDataManager(frame)
        built = []
        manager.build_connected_search3_training_sample = lambda scramble, moves, ai: built.append(
            (scramble, moves, ai)
        ) or SimpleNamespace()

        self.assertEqual(manager.store_connected_search3_data(source), 2)
        self.assertEqual(built, [(('scramble',), ('A', 'B'), source)] * 2)
        self.assertEqual(original.datas_search3, [])
        self.assertEqual(search3_one.indices_search3, [0])
        self.assertEqual(search3_two.indices_search3, [0])


if __name__ == '__main__':
    unittest.main()

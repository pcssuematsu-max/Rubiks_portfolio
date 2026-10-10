"""Ready-to-run configuration profiles for the GUI."""

from ui.frame import build_default_bootstrap_datas
from ui.frame_config import FrameConfig

def _default_initial_scramble_groups(size,puzzle_type):
    """起動時に登録する既定の scramble 候補群を返す。"""
    if puzzle_type == "square1":
        return (
            [
                ((0, 0, "/"),),
                ((1, 0, None),),
                ((0, 1, None),),
                ((1, 1, "/"),),
                ((3, -2, "/"),),
                ((-3, 3, "/"),),
                ((-2, 0, "/"),),
                ((0, 3, "/"),),
                ((1, 1, "/"),(6,6,None)),
                ((3, -2, "/"),(6,6,None)),
                ((-3, 3, "/"),(6,6,None)),
                ((-2, 0, "/"),(6,6,None)),
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "skewb":
        return (
            [
                ("URF",),
                ("ULB",),
                ("UBR","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF"),
                ("UFL'", 'ULB', "UFL'", "URF'", "UFL'", 'URF', 'UBR', "ULB'", "UBR'", 'ULB'),
                ("DRB'", 'DBL', 'DRB', 'DFR', 'DFL', 'DBL', "DFR'", "DBL'", "DFR'", "DFL'"),
                ("UBR'","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF","UBR'"),
                ("DRB'", 'DBL', 'DRB', "DBL'", "DFR'", "DRB'", 'DFR', 'DRB', 'DBL', "DRB'", "DBL'", "DFR'", 'DRB', 'DFR'),
                ('DRB', 'UBR', "ULB'", 'DRB', "UBR'", "ULB'", "UBR'", "ULB'", 'UBR'),
                ('UBR', "ULB'", 'UBR', "ULB'", "UBR'", 'ULB', "UBR'", 'ULB'),
                ("DRB'", "DBL'", "ULB'", 'UFL', "URF'", "ULB'", "UBR'", 'ULB', "UBR'", "ULB'", "UFL'", 'ULB'),
                ("URF'", 'DFL', "DFR'", "DFL'", 'URF', "UBR'", 'UFL', "URF'", 'UFL', 'URF', 'UFL', "URF'", "UFL'", 'URF', "ULB'", 'UBR', 'ULB', "UBR'"),
                ("URF'", 'DFL', "DFR'", "DRB'", "DFL'", "ULB'", 'URF', "UBR'", "UFL'", "ULB'"),

            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "pyraminx":
      return (
            [
                ("R", "U", "R'", "U'"),
                ("L'", "U'", "L", "U"),
                ("R", "L'", "R'", "L"),
                ("U", "R", "U'", "R'"),
                ('U', 'R','U', "R'", 'U', 'R', 'U', "R'","u'"),
                ('L', 'R', 'U', "R'", "U'", "L'"),
                ('R', "L'", 'U', 'L', "U'", "R'"),
                ("R'","L","R","L'","U","L'","U'","L"),
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "master_pyraminx":
      return (
            [
                ("3L","3R","3L'","3R'"),
                ("3L","3R'","3L'","3R"),
                ('3R', '3B', 'L', '3B', "3U'", "R'", "3L'", "3B'", "3R'", 'L', "3U'", "3B'", "3L'", "U'", '3R'),



            ],
            [
                ("u",),
                ("l",),
                ("R","3U","R'","3U'"),

            ],
            [

            ],
            [],
            [],
            [],
            [],
            [],
        )


    if puzzle_type == "megaminx":
        return (
            [
                ("R2'", "U'", 'R2', 'F2', "R2'", "F2'", "U'", 'F2', 'R2', "F2'", "R2'", 'U2', 'R2', "U'"),
                ("U2'", "F'", 'U2', 'R2', "U2'", "R2'", "F'", 'R2', 'U2', "R2'", "U2'", 'F2', 'U2', "F'"),
                ('R2', "U2'", "R2'", "F2'", "U'", 'F2', "U'", 'R2', "U'", "R2'", "F2'", "U2'", 'F2', "U2'"),
                ('U2', "F2'", "U2'", "R2'", "F'", 'R2', "F'", 'U2', "F'", "U2'", "R2'", "F2'", 'R2', "F2'"),
            ],
            [
                ("R'", "L'", "U'", 'R', 'U2', "L'", 'U', 'L', "U2'", 'L'),
                ("L2'", 'U', 'L2', 'U', "L2'", "U'", 'L2', "U'"),
                ("U'", "F2'", 'U', 'F2', 'U', "F2'", "U'", 'F2'),
                ('U2', "bL2'", "sL2'", "bR2'", 'bL', 'B', "bL'", "B'", "bR'", "sL'", "B'", "bL'", 'B', 'bL', "sL2'", "bR2'", 'bL2', "U2'"),
                ("F'", "U'", "F'", 'U', 'F', "R'", 'F', 'R'),
            ],
            [
            ],
            [
            ],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "fto":
        return (
            [
                ("URF'","mUBR'","DLF'","UBR","DRB'","UFL'"),
                ("UFL","mUBR'","URF","DLF","UBR'","DRB'"),
                ("mUBR","UFL","URF","DRB'","DLF'","UBR'"),
            ],
            [
                ("DFR'", 'UFL', "DFR'", "UFL'", 'DFR', "DLF'", "DFR'", 'UFL', 'DFR', "UFL'", 'DLF', 'DFR'),
            ],
            [
                ("mULB'", "UBR'", 'DFR', 'UBR', 'mULB', "UBR'", "DFR'", 'UBR'),
            ],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "cto":
        return (
            [
                ('U', 'L', 'U', "L'", 'U', 'L', 'U', "L'", 'U', "u'", 'U', 'L', 'U', "L'", "U'", 'L', "U'", "L'", "R'", "U'", "R'", 'U', "R'", "U'", "R'", 'U', "R'", 'r'),
                ("U","F'","R2","D'","L2","B'","U2","R'","D","L'","F","B2","u","l","b","r'","d2","f'"),
            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if size == 3:
        return (
            [
                (" R2"," U'"," R "," U'"," R "," U "," R'"," U "," R2"," U "," D'"," R "," U'"," R'"," D "),
                (" D'"," R "," U "," R'"," D "," U'"," R2"," U'"," R "," U'"," R'"," U "," R'"," U "," R2"),
                (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " L'", ' R ', ' U2', ' D2', ' B2', ' D2', ' B2'),
                (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " R'", ' L ', ' B2', ' U2', ' F2'),
                (" R " ," U "," R'"," U'"," F'"," U "," F "),
                (' D ', ' F ', " D'", " L'", " F'", ' L ', ' F '),
                (' D ', ' F ', " D'", " U'", ' B2', ' U ', ' F ', " U'", ' B2', ' U ', " F'"),
                (' L ', " F'", " L'", " U'", ' F ', ' U '),
                (" U'", ' F ', " D'", ' F ', ' U ', " F'", ' D '),
                (' F2', " U'", ' F ', ' U ', ' L ', " F'", " L'"),
                (" R "," U'"," R "," U "," R'"," D "," R "," D'"," R "," U'"," D "," R2"," U "," R2"," D'"," R2"),
                (' D ', " U'", ' R ', " U'", " R'", ' U2', " F'", " U'", " R'", " F'", ' R ', ' U ', ' F ', " D'"),
                (" R "," L "," U2"," R'"," L'"),  
                (" R "," L "," U2"," R'"," L'"," U2"),
                (" R'"," U "," F'"," U "," F "," U'"," R "),  
                (" R'"," U "," F'"," U2"," F "," U'"," R "),              
            ],
            [
                (" U ",),
            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            )
    elif size == 4:
        return (
            [
                ("2R "," U ","2F'","2R'"," U ","2F "),
                (" U ","2R ","2F "," D "," R ","2B'","2L'"," F'","2U "," F'"," R ","2D'","2F2"),
                (" U "," R "," F'"," D2"," R'"," F "," B2"," R "," L'"," U2"," R "," F2"," R2"),

            ],
            [
            ],
            [
            ],
            [
            ],
            [
            ],
            [],
            [],
            [],
        )


    elif size == 7:
        return (
            [
                (" R "," L "),
                ("2U ","3U'"),
                (" F "," S2"),
                (" x2"," U'"),
                (" z2"," U2"),
                (" y "," M2"," S2"),
            ],
            [
                ("2F ","3U ","2F'"),
                ("2R ","2D'","2R2"),
                (" U "," B2"," U'"),
                (" U'","2U "," M2"," S2"),
                (" R2"," F ","2R2"," F'"," R2"),
                (" R2"," F ","2R'"," F'"," R2"),
                (" R2"," F ","2R "," F'"," R2"),
            ],
            [
            ],
            [
            ],
            [
            ],
            [],
            [],
            [],
        )


def build_public_frame_config():
    """Return a lightweight configuration suitable for a public demo."""
    return FrameConfig(
        control_panel_mode="simple",
        puzzle_type="rubiks",
        cube_size=3,
        ai_search_modes=("search2",),
        initial_scramble_groups=_default_initial_scramble_groups(3, "rubiks"),
        search3_progress=(False,),
        search2_max_frontiers=(5000,),
        use_torch=(False,),
        use_torch_predict=(False,),
        use_torch_training=(False,),
        transform_idx=(0,),
        flip_inside_idx=(False,),
        priority_list=(
            (
                "Corner",
                "MidEdge",
            ),
        ),
        # Public launches should not synthesize extra training samples.
        bootstrap_datas=(),
        bootstrap_search3_datas=(),
        max_search2_data=8000,
        max_search3_data_per_ai=2000,
    )


def build_experiment_frame_config():
    ai_search_modes = [
        'search3'
        if ai_index in [2,3,4,5,6,7,10,11,18,19]
        else 'search2'
        for ai_index in range(25)
    ]
    # AI20/21 compare the calibrated value target on Linear, while AI22-24
    # use the same target with Transformer attention.
    original_transformer_attention = [False] * 10 + [True] * 10 + [False,False,True,True,True]
    ai_count = len(ai_search_modes)
    is_search2_ai = [mode.startswith('search2') for mode in ai_search_modes]
    lrs = [
        2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,2.0e-6,
        5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,5.0e-6,
        2.0e-6,2.0e-6,5.0e-6,5.0e-6,5.0e-6,
    ]
    wdlrs = [
        1.0e-4 if original_transformer_attention[ai_index] else (1.0e-7 if is_search2_ai[ai_index] else 1.0e-5)
        for ai_index in range(ai_count)
    ]
    skip_search = [is_search2_ai[ai_index] for ai_index in range(ai_count)]
    weight_decay = [True] * ai_count
    activations = ['SiLU'] * ai_count
    residuals = [True] * ai_count
    # AI21/23/24 compare route-state pairs again.  They share the same
    # direct, connected routes as the calibrated steps_to_goal group, while
    # retaining their own pairwise Value objective.
    search2_value_loss_types = ['myloss'] * 20 + [
        'steps_to_goal', 'myloss2_pairwise', 'steps_to_goal',
        'myloss2_pairwise', 'myloss2_pairwise',
    ]
    search2_value_loss_margins = [0.0] * ai_count
    # The calibrated regression group uses only its direct Huber objective.
    search2_value_target_scales = [1.0] * ai_count
    # Keep the target at -n.  This coefficient only balances the regression
    # loss after route-wise normalization.
    steps_to_goal_value_loss_weights = [0.0] * 20 + [5.0,5.0,1.0,1.0,1.0]
    # Keep at most twelve Value states from each near/mid/far remaining-step
    # band in one route.  Policy training still sees the complete route.
    steps_to_goal_states_per_bands = [0] * 20 + [12] * 5
    # Transformer steps_to_goal replay reserves most non-recent samples for
    # an even mix of remaining-step bands and route efficiency.
    # Linear models consume their full buffer, so this is observational there.
    steps_to_goal_replay_stratified_ratios = [0.0] * 22 + [0.75] * 3
    # A direct solve that takes over three times its initial scramble length
    # is useful Policy data but a poor absolute remaining-step calibration
    # target.  Keep its policy/pairwise training while excluding it only from
    # the two calibrated steps_to_goal Value losses.
    steps_to_goal_value_max_route_efficiencies = [0.0] * 20 + [3.0,0.0,3.0,0.0,0.0]
    # Balance Huber supervision across remaining-step bands inside one batch.
    # The cap is an oversampling bound, so every selected direct trajectory
    # remains available to Policy even when 31+ Value states are scarce.
    steps_to_goal_value_batch_band_max_copies = [1] * 20 + [3,1,3,1,1]
    # Transformer pairwise buffers currently contain about one third fallback
    # routes.  Keep a direct-route majority in each sampled learning pass.
    pairwise_fallback_max_ratios = [1.0] * 23 + [0.25,0.25]
    search2_rank_loss_mixes = [0.0] * ai_count
    search2_rank_loss_apply_types = ['all'] * 20 + ['none'] * 5
    # Transformer Search3: fixed validation shows the rank-loss pair (10/11)
    # orders value states more reliably than its no-rank baseline (18/19).
    # Raise only that treatment to 0.10 while retaining the latter as control.
    search3_rank_loss_mixes = [
        0.10 if ai_index in (10,11) else 0.0
        for ai_index in range(ai_count)
    ]
    # Search3 budget allocation.  Linear Search3 also exercises the staged
    # allocator (AI 3/5/7).  AI 10/11 share the rank-loss setting, and AI
    # 18/19 are its no-rank-loss counterpart; use the first member of each
    # Transformer pair as the progressive condition.
    search3_budget_modes = ['fixed'] * ai_count
    for ai_index in (3,5,7,10,18):
        search3_budget_modes[ai_index] = 'progressive'
    search3_budget_stage_playouts = [(1000,3000,6000)] * ai_count
    search3_budget_confidence_visit_shares = [0.70] * ai_count
    # The Linear progressive variants otherwise abandon 30+ move cases at
    # tier 2 too often.  Let a smaller, still positive value improvement
    # justify the final tier; fixed peers remain the comparison baseline.
    search3_budget_min_improvements = [
        0.02 if ai_index in (3,5,7) else 0.05
        for ai_index in range(ai_count)
    ]
    search3_budget_min_playout_depths = [3.0] * ai_count
    # A growing root preference after the middle tier is another sign that
    # additional Search3 budget can be useful, even before Value rises 0.05.
    search3_budget_min_visit_share_gains = [0.03] * ai_count
    # Preserve the standard 10k allocation unless the final tier is still
    # materially concentrating at the root.  This one rescue tier targets
    # the otherwise unresolved deep branches without assuming that a large
    # scramble necessarily needs more work.
    search3_budget_rescue_playouts = [
        10000 if ai_index in (3,5,7,10,18) else 0
        for ai_index in range(ai_count)
    ]
    search3_budget_rescue_min_visit_share_gains = [
        0.10 if ai_index in (3,5,7,10,18) else 1.0
        for ai_index in range(ai_count)
    ]
    # Root concentration alone produced many unsuccessful rescue tiers.
    # Require the final tier's Value estimate not to regress as well.
    search3_budget_rescue_min_improvements = [
        0.0 if ai_index in (3,5,7,10,18) else 1.0
        for ai_index in range(ai_count)
    ]
    # Retain enough of the tree and evaluator cache to make the fixed 10k
    # allocation comparable to a continuous 10k-playout PUCT call.
    search3_max_node_caches = [10000] * ai_count
    search3_max_prediction_caches = [10000] * ai_count
    # Keep fallback-prefix collection aligned with staged budget allocation.
    # A progressive AI can therefore retain its direct-search segments and
    # the greedy recovery trajectory as one connected training path.
    search3_progress = [mode == 'progressive' for mode in search3_budget_modes]
    w1_initializers = [
        [
#        {'selector': {'correct': True, 'solve_group':'Corner'}, 'basis': [0 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'MidEdge'}, 'basis': [1 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'Wing-Layer2'}, 'basis': [2 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'Wing-Layer3'}, 'basis': [3 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'XCenter-Layer2'}, 'basis': [4 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'XCenter-Layer3'}, 'basis': [5 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'PlusCenter-Layer2'}, 'basis': [6 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'PlusCenter-Layer3'}, 'basis': [7 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'ObliqueCenter-A'}, 'basis': [8 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'ObliqueCenter-B'}, 'basis': [9 + 11 * i for i in range(5)], 'scale': -0.05},
#        {'selector': {'correct': True, 'solve_group':'CoreCenter'}, 'basis': [10 + 11 * i for i in range(5)], 'scale': -0.05},
        ],
    ] * ai_count
    # Example:
    # w1_initializers[10] = [
    #     {'selector': {'correct': True}, 'basis': 0, 'scale': 0.05},
    #     {'selector': {'piece_type': 'Center', 'colors': ['Red']}, 'basis': 1, 'scale': 0.05},
    #     {'selector': {'piece_type': 'Edge', 'position_contains': ['U:Red', 'R:Blue', '2F']}, 'basis': 2, 'scale': 0.05},
    # ]


    adam = weight_decay.copy()

    cube_size = 3
    puzzle_type = 'cube'
    if cube_size >= 6:
        transform_idx = ([0,49,50,3,52,5,54,7,24,25] * 2) + [0] * 5
        flip_inside_idx = ([False,True] * 10) + [False] * 5
    else:
        transform_idx = ([0,1,2,3,4,5,6,7,24,25] * 2) + [0] * 5
        flip_inside_idx = [False] * ai_count


    if puzzle_type == 'megaminx':
        priority_list = [['Corner', 'MidEdge']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0,1,2,3,4,5,6,7,8,9] * 2
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'pyraminx':
        priority_list = [['Corner', 'Edge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'master_pyraminx':
        priority_list = [['Corner', 'Edge', 'MidEdge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'skewb':
        priority_list = [['Corner', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'square1':
        priority_list = [['Corner', 'Edge', 'Shape']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'fto':
        priority_list = [['Corner', 'Edge', 'CenterA', 'CenterB']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'cto':
        priority_list = [['Corner', 'Edge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    else:
        priority_list = [
            ['CoreCenter','ObliqueCenter-A','PlusCenter-Layer2','XCenter-Layer2','ObliqueCenter-B','PlusCenter-Layer3','XCenter-Layer3','Wing-Layer2','Wing-Layer3','Corner','MidEdge'],
            ['Wing-Layer3','Wing-Layer2','MidEdge','Corner','XCenter-Layer2','PlusCenter-Layer2','ObliqueCenter-A','XCenter-Layer3','PlusCenter-Layer3','ObliqueCenter-B','CoreCenter'],
        ] * 13
        priority_list = priority_list[:ai_count]
        bootstrap_datas = build_default_bootstrap_datas(cube_size = cube_size)
        bootstrap_search3_datas = None

    return FrameConfig(
        control_panel_mode="advanced",
        puzzle_type = puzzle_type,
        cube_size = cube_size,
        ai_search_modes = ai_search_modes,
        initial_scramble_groups = _default_initial_scramble_groups(cube_size,puzzle_type),
        transform_random = True,
        search3_progress = search3_progress,
        lrs = lrs,
        wdlrs = wdlrs,
        skip_search = skip_search,
        weight_decay = weight_decay,
        adam = adam,
        activations = activations,
        lr_vs = [0.99] * ai_count,
        lr_hs = [0.99] * ai_count,
        out_cs = [1.0] * ai_count,
        # C=0.05 consistently left trees shallow while consuming budget.
        # AI2/3 use C=0.5, AI4/5 widen from C=1 to 5, and AI6/7 retain
        # C=5. Transformer rank/no-rank pairs all share the C=1→2 schedule.
        search3_cs = [0,0,0.5,0.5,1.0,1.0,5,5,0,0] + [1.0] * 10 + [0.0] * 5,
        search3_c_depth_maxes = [0,0,0.5,0.5,5.0,5.0,5,5,0,0] + [2.0] * 10 + [0.0] * 5,
        search3_c_depth_start_depths = [0] * 10 + [4] * 10 + [0] * 5,
        search3_c_depth_ramp_depths = [8] * 10 + [6] * 10 + [0] * 5,
        search3_budget_modes = search3_budget_modes,
        search3_budget_stage_playouts = search3_budget_stage_playouts,
        search3_budget_confidence_visit_shares = search3_budget_confidence_visit_shares,
        search3_budget_min_improvements = search3_budget_min_improvements,
        search3_budget_min_playout_depths = search3_budget_min_playout_depths,
        search3_budget_min_visit_share_gains = search3_budget_min_visit_share_gains,
        search3_budget_rescue_playouts = search3_budget_rescue_playouts,
        search3_budget_rescue_min_visit_share_gains = search3_budget_rescue_min_visit_share_gains,
        search3_budget_rescue_min_improvements = search3_budget_rescue_min_improvements,
        search3_max_node_caches = search3_max_node_caches,
        search3_max_prediction_caches = search3_max_prediction_caches,
        search2_max_frontiers = [30000] * ai_count,
        # One correct calibrated steps_to_goal move should increase Value by
        # roughly one.  A 0.75 margin accepts that progress despite ordinary
        # prediction noise; rank-based objectives retain the proven 1.0 gate.
        search2_skip_differences = [1.0,5.0] + [1.0] * 6 + [10.0,1000.0] + [1.0,1.0,1.0,1.0,10.0,10.0,1000.0,1000.0,1.0,1.0] + [1.0,1.0,1.0,1.0,1000.0],
        search2_torch_batch_sizes = [
            64 if original_transformer_attention[ai_index] else 100
            for ai_index in range(ai_count)
        ],
        search2_value_loss_types = search2_value_loss_types,
        search2_value_loss_margins = search2_value_loss_margins,
        search2_value_target_scales = search2_value_target_scales,
        steps_to_goal_value_loss_weights = steps_to_goal_value_loss_weights,
        steps_to_goal_states_per_bands = steps_to_goal_states_per_bands,
        steps_to_goal_replay_stratified_ratios = steps_to_goal_replay_stratified_ratios,
        steps_to_goal_value_max_route_efficiencies = steps_to_goal_value_max_route_efficiencies,
        steps_to_goal_value_batch_band_max_copies = steps_to_goal_value_batch_band_max_copies,
        pairwise_fallback_max_ratios = pairwise_fallback_max_ratios,
        search2_rank_loss_mixes = search2_rank_loss_mixes,
        search2_rank_loss_apply_types = search2_rank_loss_apply_types,
        search3_rank_loss_mixes = search3_rank_loss_mixes,
        torch_training_devices = [
            'cpu' if original_transformer_attention[ai_index] else 'auto'
            for ai_index in range(ai_count)
        ],
        use_torch = [False] * ai_count,
        use_torch_predict = [
            bool(original_transformer_attention[ai_index])
            for ai_index in range(ai_count)
        ],
        use_torch_training = [
            bool(original_transformer_attention[ai_index])
            for ai_index in range(ai_count)
        ],
        residuals = residuals,
        update_scales = [
            # A restored Transformer model can receive a disproportionately
            # large first Value update from its retained replay.  Use the
            # same conservative Value multiplier for every Transformer
            # Search2 model; keep the Linear comparison unchanged.
            (5.0, 1.0, 1.0)
            if original_transformer_attention[ai_index] and is_search2_ai[ai_index]
            else ((5.0, 1.0, 20.0) if is_search2_ai[ai_index] else (5.0, 1.0, 1.0))
            for ai_index in range(ai_count)
        ],
        original_transformer_attention = original_transformer_attention,
        original_transformer_attention_dims = [64] * ai_count,
        original_transformer_attention_token_modes = ['piece'] * ai_count,
        original_piece_attention_backward_chunk_sizes = [32] * ai_count,
        original_train_batch_sizes = [
            20 if original_transformer_attention[ai_index] else 100
            for ai_index in range(ai_count)
        ],
        original_train_state_batch_sizes = [
            16 if original_transformer_attention[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_max_batches = [
            100 if original_transformer_attention[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_recent_ratios = [
            (0.25 if ai_index in (22,23,24)
             else (0.5 if original_transformer_attention[ai_index] else 0.0))
            for ai_index in range(ai_count)
        ],
        # Keep a small base of reliable local trajectories while replaying
        # hard fallback paths.  Recent hard data otherwise drove short lines
        # below 10% of each Transformer Search3 learning pass.
        original_train_short_sequence_max_steps = [
            19 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_short_sequence_min_ratios = [
            0.20 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0.0
            for ai_index in range(ai_count)
        ],
        # Keep an explicit middle-distance slice while replaying long lines.
        # ``long_sequence_max`` caps the *actual* selected share, including
        # fresh/recent data; the old reserved-only setting had risen above 80%.
        # Search2 and the linear baseline retain their previous sampler.
        original_train_medium_sequence_min_steps = [
            20 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_medium_sequence_max_steps = [
            29 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_medium_sequence_ratios = [
            0.25 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0.0
            for ai_index in range(ai_count)
        ],
        original_train_long_sequence_min_steps = [
            30 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_long_sequence_ratios = [
            0.25 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0.0
            for ai_index in range(ai_count)
        ],
        original_train_long_sequence_max_ratios = [
            0.60 if original_transformer_attention[ai_index] and not is_search2_ai[ai_index] else 0.0
            for ai_index in range(ai_count)
        ],
        # Save batch-level raw-gradient aggregates for every model.  This is
        # small enough for the history file and lets us compare head balance.
        gradient_log_enableds = [True] * ai_count,
        w1_initializers = w1_initializers,
        max_search2_data = 80000,
        max_search3_data_per_ai = 80000,
        transform_idx = transform_idx,
        flip_inside_idx = flip_inside_idx,
        priority_list = priority_list,
        bootstrap_datas = bootstrap_datas,
        bootstrap_search3_datas = bootstrap_search3_datas,
    )

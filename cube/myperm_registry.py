"""Registration, expansion, and reindexing helpers for Rubik's myperms.

The helpers intentionally accept a cube instance instead of owning state.  This
keeps ``Rubiks_3`` as the public API while moving registry concerns out of the
cube-state implementation incrementally.
"""

from pathlib import Path

from core.myperm_effects import rename_myperms_by_effect
from core.myperm_keys import (
    make_myperm_key,
    normalize_myperm_registry,
    single_move_myperm_name,
)
from core.myperm_points import load_myperm_points, reindex_myperms_by_points


def group_indices_by_size():
    """Return the short myperm group indices for every supported cube size."""
    return {
        2: {'A':list(range(4)),'B':[],'C':[],'c':[],'D':[],'d':[],'E':[],'e':[],'F':[],'f':[],'G':[]},
        3: {'A':list(range(4)),'B':list(range(4,8)),'C':[],'c':[],'D':[],'d':[],'E':[],'e':[],'F':[],'f':[],'G':[8]},
        4: {'A':list(range(4)),'B':[],'C':list(range(4,12)),'c':[],'D':list(range(12,16)),'d':[],'E':[],'e':[],'F':[],'f':[],'G':[]},
        5: {'A':list(range(4)),'B':list(range(4,8)),'C':list(range(8,16)),'c':[],'D':list(range(16,20)),'d':[],'E':list(range(20,24)),'e':[],'F':[],'f':[],'G':[24]},
        6: {'A':list(range(4)),'B':[],'C':[4,5,6,7,8,9,10,11],'c':[12,13,14,15,16,17,18,19],'D':[20,21,22,23],'d':[32,33,34,35],'E':[],'e':[],'F':[24,25,26,27],'f':[28,29,30,31],'G':[]},
        7: {'A':list(range(4)),'B':list(range(4,8)),'C':[8,9,10,11,12,13,14,15],'c':[16,17,18,19,20,21,22,23],'D':[24,25,26,27],'d':[40,41,42,43],'E':[28,29,30,31],'e':[44,45,46,47],'F':[32,33,34,35],'f':[36,37,38,39],'G':[48]},
    }


def initialize_containers(cube):
    """Initialize the source and expanded myperm registries on ``cube``."""
    cube.myperms = {}
    add_single_moves(cube)
    cube.myperms2 = {}
    initialize_group_indices(cube)


def add_source(cube, name, moves):
    """Register one source myperm by its canonical source name."""
    cube.myperms2[name] = moves
    return name


def moves_available_for_size(cube, moves):
    """Drop inner-layer moves that do not exist for the current cube size."""
    return cube.simplify(
        tuple(move for move in moves if move_available_for_size(cube, move))
    )


def move_available_for_size(cube, move):
    token = str(move).strip()
    layer_digits = []
    for character in token:
        if not character.isdigit():
            break
        layer_digits.append(character)
    if not layer_digits:
        return True
    layer = int("".join(layer_digits))
    face_index = len(layer_digits)
    if face_index < len(token) and token[face_index] in "UDFBLR":
        return layer <= cube.size // 2
    return True


def add_single_moves(cube):
    """Populate the expanded registry with individual moves and rotations."""
    for move in cube.move_keys:
        cube.myperms[make_myperm_key(single_move_myperm_name(move), 0)] = (move,)

    rotate6_a = (("00", (" x ", " z ")), ("01", (" x ", " z'")),
                 ("02", (" x'", " z ")), ("03", (" x'", " z'")),
                 ("04", (" z ", " x ")), ("05", (" z ", " x'")),
                 ("06", (" z'", " x ")), ("07", (" z'", " x'")))
    rotate6_b = (("00", (" y ", " x2")), ("01", (" y ", " z2")),
                 ("02", (" x ", " y2")), ("03", (" x'", " y2")),
                 ("04", (" z ", " y2")), ("05", (" z'", " y2")))
    for suffix, moves in rotate6_a:
        cube.myperms[make_myperm_key('Rotate6A-' + suffix, 0)] = moves
    for suffix, moves in rotate6_b:
        cube.myperms[make_myperm_key('Rotate6B-' + suffix, 0)] = moves


def initialize_group_indices(cube):
    """Build short and semantic names for the cube's myperm groups."""
    short_group_indices = group_indices_by_size()[cube.size]
    group_names = cube._group_name_map()
    cube.group_indices = {}
    for short_key, indices in short_group_indices.items():
        index_list = list(indices)
        cube.group_indices[short_key] = index_list
        cube.group_indices[group_names[short_key]] = index_list


def register_foundational_algorithms(cube):
    """Register the size-independent foundational Rubik's myperms."""
    algorithms = (
        ('EAll12s', (' U2', ' D2', ' F2', ' B2', ' R2', ' L2')),
        ('EAll12[2x6][XY>YX]', (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " L'", ' R ', ' U2', ' D2', ' B2', ' D2', ' B2')),
        ('EAll12[XY>YX]', (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " R'", ' L ', ' B2', ' U2', ' F2')),
        ('C8~v01', (' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B ')),
        ('C8~v02', (" D'", ' L ', ' D ', ' R2', " D'", " L'", ' D ', ' R2', ' U2', " B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ', " R'", ' U ', ' R ', ' D2', " R'", " U'", ' R ', ' D2', ' L2', " F'", ' R ', ' F ', ' L2', " F'", " R'", ' F ', ' R ', ' B2', " R'", ' D ', ' F2', " D'", ' R ', ' B2', " R'", ' D ', ' F2', " D'")),
        ('C8[3x2]+EAll12[3x4]', (" L'", ' R ', ' U ', " D'", " F'", ' B ', " L'", ' R ')),
        ('C8s+EAll12[2x6]', (" L'", ' R ', ' U2', ' D2', " L'", ' R ', ' F2', ' B2')),
        ('EAll12[3x4]', (' F ', ' B2', " R'", ' D2', ' B ', ' R ', ' U ', " D'", ' R ', " L'", " D'", " F'", ' R2', ' D ', ' F2', " B'")),
        ('EAll12[6x2]', (' F ', ' B2', " R'", ' D2', ' B ', ' R ', ' U ', " D'", ' R ', " L'", " D'", " F'", ' R2', ' D ', ' F2', " B'", ' L2', ' R2', ' U2', ' D2', ' F2', ' B2')),
        ('C8[2x4]+EAll8[2x2]', (' L ', ' U ', ' F2', ' R ', " L'", ' U2', " B'", ' U ', ' D ', ' B2', ' L ', ' F ', " B'", " R'", ' L ', " F'", ' R ')),
        ('C8p[4x2]', (' R2', ' L2', " U'", ' R2', ' L2', ' U2', ' B2', ' F2', ' D ', ' B2', ' F2', ' U2')),
        ('C8s~v01', (' L2', ' U2', ' D2', ' F2', ' U2', ' D2', ' L2', ' R2', ' B2', ' R2')),
        ('C8[3x2]~v01', (" R'", ' F2', ' B2', ' R ', ' D ', ' F2', ' B2', " D'", " U'", ' F2', ' B2', ' D ', ' B2', ' F2', ' R ', ' B2', ' F2', " L'")),
        ('C8s~v02', (' U ', ' R2', ' U2', ' D2', ' B2', ' F2', ' L2', ' B2', ' F2', ' U ', ' D2')),
        ('C8[4x2]', (' R2', ' L2', " U'", ' R2', ' L2', ' U2', ' B2', ' F2', ' D ', ' B2', ' F2', ' U2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B ')),
        ('C8[2x4]~v01', (' L2', ' U2', ' D2', ' F2', ' U2', ' D2', ' L2', ' R2', ' B2', ' R2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B ')),
        ('C8[3x2]~v02', (" R'", ' F2', ' B2', ' R ', ' D ', ' F2', ' B2', " D'", " U'", ' F2', ' B2', ' D ', ' B2', ' F2', ' R ', ' B2', ' F2', " L'", ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B ')),
        ('C8[2x4]~v02', (' U ', ' R2', ' U2', ' D2', ' B2', ' F2', ' L2', ' B2', ' F2', ' U ', ' D2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B ')),
        ('C6[DBL>FLU>RDF;DRB>BUL>RFU]+EAll6[DB>LU>FR;DR>LB>FU]', (' F ', ' R ', ' F ', " D'", ' L ', ' D ', ' F2', ' R2', " D'", " R'", ' B ', " U'", " B'", ' R2', ' D ')),
        ('C8s+EAll4s[BR<>RF;FL<>LB]', (' F ', ' U ', ' F ', ' R ', ' L2', ' B ', " D'", ' R ', ' D2', ' L ', " D'", ' B ', ' R2', ' L ', ' F ', ' U ', ' F ')),
    )
    for name, moves in algorithms:
        add_source(cube, name, moves)


def expand_registered(cube, names = None):
    """Expand source algorithms through each available symmetry transform."""
    cube.myperms2 = normalize_myperm_registry(cube.myperms2)
    keys = tuple(cube.myperms2.keys()) if names is None else tuple(names)
    for key in keys:
        if key not in cube.myperms2:
            continue
        transformed = cube.make_transformations(cube.myperms2[key], tuple())
        transform_count = 48
        if cube.size >= 6 and any(move[0] in ['2', '3'] for move in cube.myperms2[key]):
            transform_count = 96
        for index in range(transform_count):
            cube.myperms[make_myperm_key(key, index)] = transformed[0][index]


def reindex_by_points(cube, names = None):
    """Assign the highest-point symmetry transform to index zero."""
    points_path = Path(__file__).resolve().parent.parent / "Points.txt"
    if not points_path.exists():
        cube.myperm_transform_key_aliases = {}
        cube.myperm_transform_points = {}
        return
    point_table = load_myperm_points(points_path)
    reindex_myperms_by_points(cube, point_table, names = names)


def apply_point_reindex(cube, point_table = None):
    """Reindex the current registry and refresh its derived lookup tables."""
    if point_table is None:
        points_path = Path(__file__).resolve().parent.parent / "Points.txt"
        point_table = load_myperm_points(points_path)
    reindex_myperms_by_points(cube, point_table)
    rename_myperms_by_effect(cube)
    cube._init_myperms_index()
    cube._init_single_move_and_rotate()

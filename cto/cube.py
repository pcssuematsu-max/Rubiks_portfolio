"""Corner Turning Octahedron state/move utilities."""

import itertools
import math

import numpy as np

from core.myperm_keys import make_myperm_key
from fto.cube import FTO_FACE_SIGNS, FtoCube


CTO_MOVE_AXES = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}
CTO_AXIS_TO_MOVE = {axis:move for move, axis in CTO_MOVE_AXES.items()}
CTO_TIP_THRESHOLD = 2.0 / 3.0
CTO_WIDE_THRESHOLD = 1.0 / 3.0
CTO_LAYER_EPSILON = 1.0e-8


class CtoCube(FtoCube):
    """3-layer Corner Turning Octahedron model using the shared puzzle API."""

    myperm_point_puzzle = "cto"

    def _init_move_tables(self):
        self.move = {}
        for face, axis_tuple in CTO_MOVE_AXES.items():
            axis = np.array(axis_tuple, dtype = "f")
            for base, layer in ((face.lower(), "tip"), (face, "wide")):
                clockwise = self._build_move_permutation(axis, layer = layer)
                self.move[base] = clockwise
                self.move[base + "'"] = np.argsort(clockwise)
                self.move[base + "2"] = clockwise[clockwise]

    def _build_move_permutation(self, axis, layer):
        perm = np.arange(self.sticker_num)
        normalized_axis = axis / np.linalg.norm(axis)
        angle = -math.pi / 2.0
        selected = self._selected_stickers(normalized_axis, layer)
        for source in selected:
            center = self._rotate_vector(self.stickers[source]["center"], normalized_axis, angle)
            normal = self._rotate_vector(self.stickers[source]["normal"], normalized_axis, angle)
            target = self._nearest_sticker(center, normal)
            perm[target] = source
        return perm

    def _selected_stickers(self, axis, layer):
        threshold = CTO_TIP_THRESHOLD if layer == "tip" else CTO_WIDE_THRESHOLD
        return [
            index for index, sticker in enumerate(self.stickers)
            if np.dot(axis, sticker["center"]) > (threshold + CTO_LAYER_EPSILON)
        ]

    def _init_move_keys(self):
        self.move_keys = tuple(self.move.keys())
        self.move_len = len(self.move_keys)
        self.key_to_num = {key:index for index, key in enumerate(self.move_keys)}
        self.inverse = {"":"'","'":"","2":"2"}
        self.mult = {
            ("", ""):"2",
            ("", "2"):"'",
            ("", "'"):0,
            ("2", ""):"'",
            ("2", "2"):0,
            ("2", "'"):"",
            ("'", ""):0,
            ("'", "2"):"",
            ("'", "'"):"2",
        }

    def _init_transformation_tables(self):
        self.transformation_keys = []
        axis_set = {tuple(axis) for axis in CTO_MOVE_AXES.values()}
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat = 3):
                matrix = np.zeros((3, 3), dtype = int)
                for row, source in enumerate(perm):
                    matrix[row, source] = signs[row]
                mapped_axes = {tuple(matrix @ np.array(axis, dtype = int)) for axis in axis_set}
                if mapped_axes == axis_set:
                    self.transformation_keys.append({
                        "matrix": matrix,
                        "mirror": round(np.linalg.det(matrix)) < 0,
                    })
        self._move_identity_transformation_to_front()
        self.tf_invert = {
            index:self._inverse_transformation_index(index)
            for index in range(len(self.transformation_keys))
        }

    def _register_myperms2(self):
        self._add_myperm2('E3[FR>UB>RU]', ('U', 'R', "U'", "R'"), add_inverse = False)
        self._add_myperm2('E3[FR>RU>UB]', ('R', 'U', "R'", "U'"), add_inverse = False)
        self._add_myperm2('E3[FL>FU>LU]', ('U', "L'", "U'", 'L'), add_inverse = False)
        self._add_myperm2('E3[FL>LU>FU]', ("L'", 'U', 'L', "U'"), add_inverse = False)
        self._add_myperm2('E3[FL>LU>RF]', ('F2', 'L', 'F2', "L'"), add_inverse = False)
        self._add_myperm2('E3[FL>RF>LU]', ('L', 'F2', "L'", 'F2'), add_inverse = False)

        self._add_myperm2('E2[FL>LF;UF>FU]~v01', ("L'", 'U', 'L', "U'", 'F', "U'", "F'", 'U'), add_inverse = False)
        self._add_myperm2('E2[FL>LF;UF>FU]~v02', ("U'", 'F', 'U', "F'", 'U', "L'", "U'", 'L'), add_inverse = False)
        self._add_myperm2('E2[FL>LF;UR>RU]~v01', ('F', "R'", 'F', 'R', "F'", 'U', "F'", "U'"), add_inverse = False)
        self._add_myperm2('E2[FL>LF;UR>RU]~v02', ('U', 'F', "U'", 'F', "R'", "F'", 'R', "F'"), add_inverse = False)
        self._add_myperm2('E2[FL>LF;FR>RF]~v01', ('F2', "L'", 'F', "U'", 'F2', 'U', "F'", 'L'), add_inverse = False)
        self._add_myperm2('E2[FL>LF;FR>RF]~v02', ("L'", 'F', "U'", 'F2', 'U', "F'", 'L', 'F2'), add_inverse = False)
        self._add_myperm2('E2[BL>LB;FR>RF]~v01', ('F', 'L', "D'", 'L2', 'D', "L'", 'F', 'L2', 'F2'), add_inverse = False)
        self._add_myperm2('E2[BL>LB;FR>RF]~v02', ('F2', 'L2', "F'", 'L', "D'", 'L2', 'D', "L'", "F'"), add_inverse = False)

        self._add_myperm2('E3[FL>FU>FR]', ("L'", "U'", "F'", 'U', 'F', 'L'), add_inverse = False)
        self._add_myperm2('E3[FL>FR>FU]', ("L'", "F'", "U'", 'F', 'U', 'L'), add_inverse = False)
        self._add_myperm2('E3[FL>UR>FU]', ('U2', "L'", "U'", 'L', "U'"), add_inverse = False)
        self._add_myperm2('E3[FL>FU>UR]', ('U', "L'", 'U', 'L', 'U2'), add_inverse = False)
        self._add_myperm2('E3[FR>UL>UB]', ('U2', 'R', "U'", "R'", "U'"), add_inverse = False)
        self._add_myperm2('E3[FR>UB>UL]', ('U', 'R', 'U', "R'", 'U2'), add_inverse = False)
        self._add_myperm2('E3[FL>RU>RF]', ("F'", 'U', 'F2', "U'", "F'"), add_inverse = False)
        self._add_myperm2('E3[FL>RF>RU]', ('F', 'U', 'F2', "U'", 'F'), add_inverse = False)
        self._add_myperm2('E3[FL>RF>UB]', ('F', 'U2', 'F2', 'U2', 'F'), add_inverse = False)
        self._add_myperm2('E3[FL>UB>RF]', ("F'", 'U2', 'F2', 'U2', "F'"), add_inverse = False)
        self._add_myperm2('E3[BR>FL>UB]', ('B', 'L2', 'B', 'L2', 'B2'), add_inverse = False)
        self._add_myperm2('E3[BR>UB>FL]', ('B2', 'L2', "B'", 'L2', "B'"), add_inverse = False)
        self._add_myperm2('E3[BL>FR>LF]', ('F2', 'L2', 'F2', 'L2'), add_inverse = False)
        self._add_myperm2('E3[BL>LF>FR]', ('L2', 'F2', 'L2', 'F2'), add_inverse = False)
        self._add_myperm2('E3[FL>UF>RF]', ("L'", "F'", 'L', "F'", "L'", 'F2', 'L'), add_inverse = False)
        self._add_myperm2('E3[FL>RF>UF]', ("L'", 'F2', 'L', 'F', "L'", 'F', 'L'), add_inverse = False)
        self._add_myperm2('E3[FL>RU>FU]', ('F', "U'", 'R', 'U', "R'", "F'"), add_inverse = False)
        self._add_myperm2('E3[FL>FU>RU]', ('F', 'R', "U'", "R'", 'U', "F'"), add_inverse = False)
        self._add_myperm2('E3[BR>FU>LF]', ("R'", 'U', 'F', "U'", "F'", 'R'), add_inverse = False)
        self._add_myperm2('E3[BR>LF>FU]', ("R'", 'F', 'U', "F'", "U'", 'R'), add_inverse = False)
        self._add_myperm2('E3[FL>UR>UB]', ('U2', "F'", 'L', 'F', "L'", 'U2'), add_inverse = False)
        self._add_myperm2('E3[FL>UB>UR]', ('U2', 'L', "F'", "L'", 'F', 'U2'), add_inverse = False)
        self._add_myperm2('E3[FL>BU>RF]', ("U'", 'F2', 'L', 'F2', "L'", 'U'), add_inverse = False)
        self._add_myperm2('E3[FL>RF>BU]', ("U'", 'L', 'F2', "L'", 'F2', 'U'), add_inverse = False)
        self._add_myperm2('E3[FL>FU>RF]', ('U', 'F2', 'L', 'F2', "L'", "U'"), add_inverse = False)
        self._add_myperm2('E3[FL>RF>FU]', ('U', 'L', 'F2', "L'", 'F2', "U'"), add_inverse = False)

        self._add_myperm2('Ctr1[U>U-]+E2s[FR<>UB]', ('R', "U'", "R'", 'u', "U'", 'R', "U'", "R'", 'U2'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2s[FR<>UB]', ('U2', 'R', 'U', "R'", 'U', "u'", 'R', 'U', "R'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2s[UB<>UL]', ("u'", 'R2', 'U', 'R2', 'U2', 'R2', 'U', 'R2', 'U'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2s[UB<>UL]', ("U'", 'R2', "U'", 'R2', 'U2', 'R2', "U'", 'R2', 'u'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2s[FR<>UF]', ("u'", 'R', 'U', "R'", 'U', 'R', 'U', "R'", 'U2'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2s[FR<>UF]', ('U2', 'R', "U'", "R'", "U'", 'R', "U'", "R'", 'u'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2[UL>RU]', ('U2', "F'", 'U', 'F', "U'", "F'", "U'", 'F', "U'", "F'", 'U', 'F', "u'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2[UL>RU]', ('u', "F'", "U'", 'F', 'U', "F'", 'U', 'F', 'U', "F'", "U'", 'F', 'U2'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2s[FL<>UR]', ('U', "L'", "U'", 'L', "U'", "L'", "U'", 'L', 'U', 'u'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2s[FL<>UR]', ("u'", "U'", "L'", 'U', 'L', 'U', "L'", 'U', 'L', "U'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2[FL>LU]', ('U', "L'", 'U', 'L', 'U', "L'", 'U', 'L', 'U', "u'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2[FL>LU]', ('u', "U'", "L'", "U'", 'L', "U'", "L'", "U'", 'L', "U'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2[DL>BU]', ('L2', 'U', 'L2', "u'", 'U', 'L2', 'U', 'L2', 'U2'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2[DL>BU]', ('U2', 'L2', "U'", 'L2', "U'", 'u', 'L2', "U'", 'L2'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2s[DL<>UR]', ('U', "u'", 'B', 'L2', 'U', 'L2', "B'", 'U2', 'B', 'U', "B'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2s[DL<>UR]', ('B', "U'", "B'", 'U2', 'B', 'L2', "U'", 'L2', "B'", 'u', "U'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U-]+E2[DF>RF]', ('U', "F'", 'R', "U'", "R'", 'F', "U'", 'u', "F'", "U'", 'F', 'U'), add_inverse = False)
        self._add_myperm2('Ctr1[U>U+]+E2[DF>RF]', ("U'", "F'", 'U', 'F', "u'", 'U', "F'", 'R', 'U', "R'", 'F', "U'"), add_inverse = False)

        self._add_myperm2('Ctr1[U>U2]~v01', ('U2', 'R', 'U2', "R'", 'U', 'R', 'U2', "R'", 'U2', 'u2', 'R', 'U', "R'"), add_inverse = False)
        self._add_myperm2('Ctr1[U>U2]~v02', ('R', "U'", "R'", 'u2', 'U2', 'R', 'U2', "R'", "U'", 'R', 'U2', "R'", 'U2'), add_inverse = False)
        self._add_myperm2('Ctr2[F>F+;U>U+]', ('U', 'B', 'U', "B'", 'U2', 'B', 'U', "B'", "u'", 'F', 'U', 'F', "U'", 'F', 'U', 'F', "U'", 'F', "f'"), add_inverse = False)
        self._add_myperm2('Ctr2[F>F-;U>U-]', ('f', "F'", 'U', "F'", "U'", "F'", 'U', "F'", "U'", "F'", 'u', 'B', "U'", "B'", 'U2', 'B', "U'", "B'", "U'"), add_inverse = False)

    def _init_groups(self):
        self.corner_index = self._piece_indices_at_vertices()
        self.edge_index = self._piece_indices_at_edges()
        self.center_index = self._center_piece_orbits()
        self.group_pieces = {
            "Corner":self.corner_index,
            "Edge":self.edge_index,
            "Center":self.center_index,
        }
        self.group_indices = {
            group_name:list(range(len(pieces)))
            for group_name, pieces in self.group_pieces.items()
        }

    def _center_piece_orbits(self):
        used = {
            index
            for pieces in (self.corner_index, self.edge_index)
            for piece in pieces
            for index in piece
        }
        center_indices = set(range(self.sticker_num)) - used
        adjacency = {index:set() for index in center_indices}
        for permutation in self.move.values():
            for target, source in enumerate(permutation):
                source = int(source)
                if source in adjacency and target in adjacency:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

        seen = set()
        orbits = []
        for index in sorted(center_indices):
            if index in seen:
                continue
            orbit = []
            stack = [index]
            seen.add(index)
            while stack:
                current = stack.pop()
                orbit.append(current)
                for next_index in adjacency[current]:
                    if next_index not in seen:
                        seen.add(next_index)
                        stack.append(next_index)
            orbits.append(tuple(sorted(orbit)))
        return tuple(sorted(orbits, key = lambda orbit: orbit[0]))

    def invert_str(self, move):
        base, suffix = self._split_move(self.normalize_move_key(move))
        if suffix == "2":
            return base + "2"
        if suffix == "'":
            return base
        return base + "'"

    def _split_move(self, move):
        if move.endswith("'"):
            return move[:-1], "'"
        if move.endswith("2"):
            return move[:-1], "2"
        return move, ""

    def _transform_move(self, move, transformation):
        base, suffix = self._split_move(move)
        is_tip = base.islower()
        axis = self._move_axis(base)
        mapped_axis = tuple(int(value) for value in transformation["matrix"] @ axis.astype(int))
        mapped_base = CTO_AXIS_TO_MOVE[mapped_axis]
        if is_tip:
            mapped_base = mapped_base.lower()
        if transformation["mirror"] and suffix != "2":
            suffix = "" if suffix == "'" else "'"
        return self.normalize_move_key(mapped_base + suffix)

    def _move_axis(self, base):
        return np.array(CTO_MOVE_AXES[base.upper()], dtype = "f")

    def piece_display_name(self, piece_type, piece):
        labels = "-".join(self.index_to_face[index] + str(index % self.face_sticker_count) for index in piece)
        return f"{piece_type}-{labels}"

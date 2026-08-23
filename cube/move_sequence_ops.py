"""Move sequence transformation helpers for Rubik's cube operations."""


class MoveSequenceOps:
    """手順列の対称変換・反転・展開をまとめて扱う。"""

    def __init__(self, cube):
        self.cube = cube

    def flip_moves(self, moves, axis = None):
        """指定軸の鏡映ルールで手順列を変換する。"""
        if axis in self.cube.flip.keys():
            return tuple([x[0] + self.cube.flip[axis][x[1:]] for x in moves])
        return tuple(moves)

    def rotate_moves(self, moves, axis = None):
        """指定回転ルールで手順列を回転変換する。"""
        if axis in self.cube.rotate.keys():
            return tuple([x[0] + self.cube.rotate[axis][x[1:]] for x in moves])
        return tuple(moves)

    def diag_flip_moves(self, moves):
        """対角反転ルールで手順列を変換する。"""
        return tuple([x[0] + self.cube.diag_flip[x[1:]] for x in moves])

    def invert_str(self, move):
        """1手だけ逆回転に変換する。"""
        return move[:2] + self.cube.inverse[move[2]]

    def invert_moves(self, moves):
        """手順列を逆順・逆回転にした列を返す。"""
        return tuple([self.invert_str(x) for x in moves[::-1]])

    def swap_moves(self, moves):
        """2層・3層の手を入れ替える補助変換を適用する。"""
        return tuple([self.cube.swap_2_3(x) for x in moves])

    def flip_inside(self, move):
        """1手だけ内外反転ルールで変換する。"""
        if move[0] == ' ':
            return move
        return move[0] + self.cube.opposite[move[1]] + self.cube.inverse[move[2]]

    def flip_inside_moves(self, moves):
        """内外反転ルールで手順列を変換する。"""
        return tuple(self.flip_inside(x) for x in moves)

    def transform(self, moves, transform_index, flip_inside = False, invert = False):
        """変換indexに対応する対称変換を手順列へ適用する。"""
        transform_key = self._transformation_key(transform_index, invert = invert)
        transformed_moves = tuple(moves)
        for transform_step in transform_key:
            transformed_moves = self._apply_transform_step(transformed_moves, transform_step)

        if flip_inside:
            transformed_moves = self.flip_inside_moves(transformed_moves)

        return transformed_moves

    def _transformation_key(self, transform_index, invert = False):
        """変換indexから、実際に適用する変換手順列を取り出す。"""
        transform_key = self.cube.transformation_keys[transform_index]
        if not invert:
            return transform_key
        return tuple([self.cube.tf_invert[step] for step in transform_key[::-1]])

    def _apply_transform_step(self, moves, transform_step):
        """変換手順1つ分だけ手順列へ反映する。"""
        if transform_step in ['UD','FB','LR']:
            return self.flip_moves(moves, axis = transform_step)
        if transform_step == 'S':
            return self.swap_moves(moves)
        if transform_step in ['120','240']:
            return self.rotate_moves(moves, axis = transform_step)
        return self.diag_flip_moves(moves)

    def make_transformations(self, scramble_moves, solution_moves):
        """全ての対称変換について、scramble列とmove列の組を作る。"""
        scramble_list = []
        move_list = []
        for transform_index in range(len(self.cube.transformation_keys)):
            transformed_scramble = self.transform(scramble_moves, transform_index, invert = True)
            transformed_moves = self.transform(solution_moves, transform_index, invert = True)
            scramble_list.append(transformed_scramble)
            move_list.append(transformed_moves)

        return scramble_list, move_list

    def simplify(self, move_lis):
        """同じ面・同じ層の連続手をまとめて手順列を簡約する。"""
        simplified_moves = ()
        for move in move_lis:
            if len(simplified_moves) > 0 and simplified_moves[-1][:2] == move[:2]:
                combined_turn = self.cube.mult[simplified_moves[-1][2],move[2]]
                simplified_moves = simplified_moves[:-1]
                if combined_turn != 0:
                    simplified_moves += (move[:2] + combined_turn,)
            else:
                simplified_moves += (move,)

        return simplified_moves

    def conjugate(self, A, B):
        """共役 A B A^-1 を作って簡約した手順列を返す。"""
        return self.simplify(A + B + self.invert_moves(A))

    def commutator(self, A, B):
        """交換子 A B A^-1 B^-1 を作って簡約した手順列を返す。"""
        return self.simplify(A + B + self.invert_moves(A) + self.invert_moves(B))

"""Square-1 state and move utilities using the shared puzzle interface."""

import random
import re
from dataclasses import dataclass
from functools import reduce

import numpy as np

from core.scramble_selector import ScrambleSelector
from model.search_result import SearchResult


MOVE_RE = re.compile(r"^\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*(/?)$")
PIECE_CHARS = "ABCDEFGHIJKLMNOP"
SOLVED_SIDE_SLOTS = ("B", "R", "R", "R", "F", "F", "F", "L", "L", "L", "B", "B")


@dataclass(frozen = True)
class Square1Piece:
    id: int
    kind: str
    width: int
    color: str


class Square1Cube:
    """Square-1 model compatible with the existing AI/search managers."""

    def __init__(self, S = "", size = 3, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        self.size = 3
        self.order = 3
        self.F2L = F2L
        self.OLL = OLL
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.surface_num = 24
        self.sticker_num = 24
        self.colors = list(PIECE_CHARS)
        self.color_to_num = {color:index for index, color in enumerate(self.colors)}
        self.pieces = self._build_pieces()
        self.top_0 = tuple(self.pieces[:8])
        self.bottom_0 = tuple(self.pieces[8:])
        self.top = self.top_0
        self.bottom = self.bottom_0
        self.top_side = SOLVED_SIDE_SLOTS
        self.bottom_side = SOLVED_SIDE_SLOTS
        self.top_offset = 0
        self.bottom_offset = 0
        self._init_middle_regions()
        self._init_move_keys()
        self._init_myperms()
        self._init_scramble_registry()
        self._init_groups()
        self._sync_state()
        self.state_0 = self.state.copy()
        self.ips = 24 * len(self.pieces) + 24 + 2
        self.perfect_data = self.makedata()
        self.scramble_selector = ScrambleSelector(self)
        if S:
            self.set_state(S)

    def _build_pieces(self):
        pieces = []
        for index in range(8):
            pieces.append(Square1Piece(index, "Corner" if index % 2 == 0 else "Edge", 2 if index % 2 == 0 else 1, PIECE_CHARS[index]))
        for index in range(8, 16):
            pieces.append(Square1Piece(index, "Corner" if index % 2 == 0 else "Edge", 2 if index % 2 == 0 else 1, PIECE_CHARS[index]))
        return tuple(pieces)

    def _init_move_keys(self):
        slash_moves = [(u, d, "/") for u in range(-5, 7) for d in range(-5, 7)]
        rotation_moves = [(u, d, None) for u in range(-5, 7) for d in range(-5, 7) if (u, d) != (0, 0)]
        self.move_keys = tuple(slash_moves + rotation_moves)
        self.move_len = len(self.move_keys)
        self.key_to_num = {key:index for index, key in enumerate(self.move_keys)}
        self.inverse = {}
        self.mult = {}

    def _init_myperms(self):
        self.myperms = {}
        self.myperms2 = {}

    def _init_middle_regions(self):
        self.middle_regions = {
            "left":tuple(SOLVED_SIDE_SLOTS[6:12]),
            "right":tuple(SOLVED_SIDE_SLOTS[0:6]),
        }

    def _init_scramble_registry(self):
        self.my_scrambles = []
        self.my_scrambles2 = {0:{move_key:set() for move_key in self.move_keys}}
        self.my_scramble_changed_piece_keys = {0:{}}
        self.counter = {0:{}}

    def _init_groups(self):
        self.group_pieces = {
            "Corner": tuple(piece.id for piece in self.pieces if piece.kind == "Corner"),
            "Edge": tuple(piece.id for piece in self.pieces if piece.kind == "Edge"),
            "Shape": tuple(range(24)),
            "Center": tuple(),
            "MidEdge": tuple(),
        }
        self.piece_color_counter = {}

    def create_new_set(self):
        index = len(self.my_scrambles2)
        self.my_scrambles2[index] = {move_key:set() for move_key in self.move_keys}
        self.my_scramble_changed_piece_keys[index] = {}
        self.counter[index] = {}

    def normalize_move_key(self, move):
        if isinstance(move, str):
            move = move.strip()
            if move == "/":
                return (0, 0, "/")
            match = MOVE_RE.match(move)
            if not match:
                raise KeyError(move)
            u = int(match.group(1)) % 12
            d = int(match.group(2)) % 12
            slash = "/" if match.group(3) else None
            return (self._signed_turn(u), self._signed_turn(d), slash)
        if isinstance(move, (tuple, list)):
            if len(move) == 1:
                return self.normalize_move_key(move[0])
            if len(move) == 2:
                u, d = move
                slash = None
            elif len(move) == 3:
                u, d, slash = move
            else:
                raise KeyError(move)
            slash = "/" if slash == "/" else None
            return (self._signed_turn(int(u)), self._signed_turn(int(d)), slash)
        raise KeyError(move)

    def normalize_move_sequence(self, moves):
        return tuple(self.normalize_move_key(move) for move in moves)

    def _signed_turn(self, value):
        value = int(value) % 12
        if value > 6:
            value -= 12
        return value

    def format_move(self, move):
        u, d, slash = self.normalize_move_key(move)
        suffix = "/" if slash == "/" else ""
        return f"({u},{d}){suffix}"

    def format_moves(self, moves):
        return tuple(self.format_move(move) for move in self.normalize_move_sequence(moves))

    def _sync_state(self):
        slots = self._layer_slots(self.top, self.top_offset) + self._layer_slots(self.bottom, self.bottom_offset)
        self.state = np.array(slots)
        self.top_side_state = np.array(self._unit_slots(self.top_side, self.top_offset))
        self.bottom_side_state = np.array(self._unit_slots(self.bottom_side, self.bottom_offset))

    def _layer_slots(self, layer, offset):
        slots = ["?"] * 12
        position = offset % 12
        for piece in layer:
            for delta in range(piece.width):
                slots[(position + delta) % 12] = piece.color
            position = (position + piece.width) % 12
        return slots

    def _unit_slots(self, unit_values, offset):
        slots = ["?"] * 12
        for index, value in enumerate(unit_values):
            slots[(offset + index) % 12] = value
        return slots

    def _cut_positions(self, layer, offset):
        positions = {offset % 12}
        position = offset % 12
        for piece in layer:
            position = (position + piece.width) % 12
            positions.add(position)
        return positions

    def _can_slash_current(self):
        top_cuts = self._cut_positions(self.top, self.top_offset)
        bottom_cuts = self._cut_positions(self.bottom, self.bottom_offset)
        return 0 in top_cuts and 6 in top_cuts and 0 in bottom_cuts and 6 in bottom_cuts

    def is_legal_move(self, move):
        u, d, slash = self.normalize_move_key(move)
        if slash != "/":
            return True
        top_offset = (self.top_offset + u) % 12
        bottom_offset = (self.bottom_offset + d) % 12
        top_cuts = self._cut_positions(self.top, top_offset)
        bottom_cuts = self._cut_positions(self.bottom, bottom_offset)
        return 0 in top_cuts and 6 in top_cuts and 0 in bottom_cuts and 6 in bottom_cuts

    def legal_move_mask(self):
        return np.array([self.is_legal_move(move) for move in self.move_keys], dtype = bool)

    def illegal_move_indices(self):
        return np.where(~self.legal_move_mask())[0]

    def make_move(self, key):
        u, d, slash = self.normalize_move_key(key)
        self.top_offset = (self.top_offset + u) % 12
        self.bottom_offset = (self.bottom_offset + d) % 12
        if slash == "/":
            if not self._can_slash_current():
                raise ValueError("Illegal Square-1 slash move: " + self.format_move((u, d, slash)))
            self._slash()
        self._sync_state()

    def _slash(self):
        top_front, top_back = self._split_layer_at_cuts(self.top, self.top_offset)
        bottom_front, bottom_back = self._split_layer_at_cuts(self.bottom, self.bottom_offset)
        top_side_front, top_side_back = self._split_units_at_cuts(self.top_side, self.top_offset)
        bottom_side_front, bottom_side_back = self._split_units_at_cuts(self.bottom_side, self.bottom_offset)
        self.top = tuple(list(reversed(bottom_front)) + top_back)
        self.bottom = tuple(list(reversed(top_front)) + bottom_back)
        self.top_side = tuple(list(reversed(bottom_side_front)) + bottom_side_back)
        self.bottom_side = tuple(list(reversed(top_side_front)) + top_side_back)
        self._slash_middle_regions()
        self.top_offset = 0
        self.bottom_offset = 0

    def _slash_middle_regions(self):
        self.middle_regions["right"] = tuple(reversed(self.middle_regions["right"]))

    def _split_layer_at_cuts(self, layer, offset):
        ordered = self._rotate_layer_to_boundary(layer, offset, 0)
        front = []
        back = []
        position = 0
        target = front
        for piece in ordered:
            if position == 6:
                target = back
            target.append(piece)
            position += piece.width
        if position != 12:
            raise ValueError("Invalid Square-1 layer width.")
        return front, back

    def _split_units_at_cuts(self, unit_values, offset):
        ordered = self._rotate_units_to_boundary(unit_values, offset, 0)
        return list(ordered[:6]), list(ordered[6:])

    def _rotate_units_to_boundary(self, unit_values, offset, boundary):
        start = (boundary - offset) % 12
        return tuple(unit_values[start:]) + tuple(unit_values[:start])

    def _rotate_layer_to_boundary(self, layer, offset, boundary):
        offset = offset % 12
        boundary = boundary % 12
        position = offset
        for index, piece in enumerate(layer):
            if position == boundary:
                return list(layer[index:]) + list(layer[:index])
            position = (position + piece.width) % 12
        if position == boundary:
            return list(layer)
        raise ValueError("Boundary does not align with a Square-1 piece edge.")

    def scramble(self, N, Move = None, difficult_mode = False, scramble_mode = None, flip = None, rotate = None, swap = False, add_moves = False, transform_N = None, flip_inside = None, move_count_policy = 'prefer_rare'):
        if Move is not None:
            moves = self.normalize_move_sequence(Move)
            for move in moves:
                self.make_move(move)
            return moves
        moves = []
        for _ in range(max(0, int(N))):
            legal_moves = [move for move in self.move_keys if self.is_legal_move(move)]
            if moves:
                inverse_prefix = tuple(self.invert_moves((moves[-1],)))
                legal_moves = [move for move in legal_moves if (move,) != inverse_prefix]
            move = random.choice(legal_moves)
            self.make_move(move)
            moves.append(move)
        return tuple(moves)

    def select_fallback_search_result(self, ai, search_result, max_candidates = 160):
        """Return a Square-1-specific fallback SearchResult when it improves the search result."""
        if search_result.succeeded:
            return None
        candidates = self.fallback_move_candidates(max_candidates = max_candidates)
        if not candidates:
            return None
        root_value_raw = self._predict_ai_value(ai)
        scored = self._score_fallback_candidates(ai, candidates)
        if not scored:
            return None
        best_score, best_value, best_moves = max(scored, key = lambda item:item[0])
        if best_value <= root_value_raw + 1.0e-4:
            return None
        current_best = getattr(search_result, "best_value_raw", search_result.best_value)
        if len(search_result.moves) > 0 and best_value <= current_best + 1.0e-4:
            return None
        value_trace_raw = [root_value_raw, best_value]
        if search_result.search_mode == "search3":
            value_trace = [self._sigmoid(value) for value in value_trace_raw]
            root_value = value_trace[0]
            best_display_value = value_trace[-1]
        else:
            value_trace = value_trace_raw
            root_value = root_value_raw
            best_display_value = best_value
        return SearchResult(
            False,
            best_moves,
            root_value,
            value_trace,
            best_display_value,
            np.array([1, len(candidates)], dtype = "i"),
            search_mode = search_result.search_mode,
            end_reason = "square1-fallback",
            root_value_raw = root_value_raw,
            value_trace_raw = value_trace_raw,
            best_value_raw = best_value,
        )

    def fallback_move_candidates(self, max_candidates = 160):
        """Collect compact legal fallback candidates that can keep Square-1 search moving."""
        candidates = []
        seen = set()
        for move in self.move_keys:
            if move[2] == "/" and self.is_legal_move(move):
                self._append_unique_candidate(candidates, seen, (move,))
        for move in self.move_keys:
            if move[2] is not None or (move[0], move[1]) == (0, 0):
                continue
            old = self._snapshot()
            try:
                self.make_move(move)
                if self.is_legal_move((0, 0, "/")):
                    self._append_unique_candidate(candidates, seen, (move,))
            finally:
                self._restore(old)
        if len(candidates) < max_candidates:
            for move in self.move_keys:
                if self.is_legal_move(move):
                    self._append_unique_candidate(candidates, seen, (move,))
                if len(candidates) >= max_candidates:
                    break
        return candidates[:max_candidates]

    def _append_unique_candidate(self, candidates, seen, moves):
        normalized = self.normalize_move_sequence(moves)
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    def _score_fallback_candidates(self, ai, candidates):
        old = self._snapshot()
        features = []
        valid_candidates = []
        shape_scores = []
        slash_scores = []
        try:
            for moves in candidates:
                self._restore(old)
                try:
                    for move in moves:
                        if not self.is_legal_move(move):
                            raise ValueError(move)
                        self.make_move(move)
                except Exception:
                    continue
                features.append(self.makedata().reshape(-1))
                valid_candidates.append(moves)
                shape_scores.append(self._fallback_shape_score())
                slash_scores.append(1.0 if self.is_legal_move((0, 0, "/")) else 0.0)
            if not valid_candidates:
                return []
            predictions = ai._predict_search2(np.array(features, dtype = "f").T)
            values = predictions[-1].reshape(-1)
            scored = []
            for index, moves in enumerate(valid_candidates):
                score = (
                    values[index]
                    + 0.15 * slash_scores[index]
                    + 0.02 * shape_scores[index]
                    - 0.01 * len(moves)
                )
                scored.append((score, values[index], moves))
            return scored
        finally:
            self._restore(old)

    def _predict_ai_value(self, ai):
        prediction = ai._predict_search2(self.makedata().reshape(-1, 1))
        return float(prediction[-1, 0])

    def _sigmoid(self, value):
        return float(1.0 / (1.0 + np.exp(-value)))

    def _fallback_shape_score(self):
        return float(sum(self.state[index] == self.state_0[index] for index in range(24))) / 24.0

    def register_scramble_sequence(self, level, moves):
        normalized_moves = self.normalize_move_sequence(moves)
        if not normalized_moves:
            return
        while level not in self.my_scrambles2:
            self.create_new_set()
        last_move = normalized_moves[-1]
        if last_move not in self.my_scrambles2[level]:
            self.my_scrambles2[level][last_move] = set()
        self.my_scrambles2[level][last_move].add(normalized_moves)
        self.my_scramble_changed_piece_keys[level][normalized_moves] = tuple(
            self.get_chenged_pieces_keys_from_moves(normalized_moves)
        )

    def get_registered_scramble_changed_piece_keys(self, level, moves):
        return self.my_scramble_changed_piece_keys[level].get(self.normalize_move_sequence(moves))

    def get_chenged_pieces_keys_from_moves(self, moves):
        old = self._snapshot()
        self.reset()
        for move in self.normalize_move_sequence(moves):
            self.make_move(move)
        changed = []
        solved_slots = self._layer_slots(self.top_0, 0) + self._layer_slots(self.bottom_0, 0)
        for index, color in enumerate(self.state):
            if color != solved_slots[index]:
                changed.append(("Shape", index))
        for piece in self.pieces:
            if self._piece_slot_positions(piece.id) != self._solved_piece_slot_positions(piece.id):
                changed.append((piece.kind, piece.id))
        self._restore(old)
        return changed

    def _piece_slot_positions(self, piece_id):
        color = PIECE_CHARS[piece_id]
        return tuple(index for index, value in enumerate(self.state) if value == color)

    def _solved_piece_slot_positions(self, piece_id):
        color = PIECE_CHARS[piece_id]
        return tuple(index for index, value in enumerate(self.state_0) if value == color)

    def makedata(self):
        x = np.zeros(self.ips, dtype = "f")
        for slot_index, color in enumerate(self.state):
            piece_index = PIECE_CHARS.index(color)
            x[slot_index * len(self.pieces) + piece_index] = 1.0
        shape_offset = 24 * len(self.pieces)
        for slot_index, color in enumerate(self.state):
            piece = self.pieces[PIECE_CHARS.index(color)]
            if piece.kind == "Corner":
                x[shape_offset + slot_index] = 1.0
        x[shape_offset + 24] = self.top_offset / 11.0
        x[shape_offset + 25] = self.bottom_offset / 11.0
        return x

    def is_perfect(self):
        return bool((self.state == self.state_0).all())

    def reset(self):
        self.top = self.top_0
        self.bottom = self.bottom_0
        self.top_side = SOLVED_SIDE_SLOTS
        self.bottom_side = SOLVED_SIDE_SLOTS
        self.top_offset = 0
        self.bottom_offset = 0
        self._init_middle_regions()
        self._sync_state()

    def state_to_str(self):
        return reduce(lambda x, y: x + y, self.state)

    def set_state(self, S):
        if len(S) != self.sticker_num:
            return
        raise NotImplementedError("Square-1 set_state from slot string is not implemented yet.")

    def _snapshot(self):
        return (
            self.top,
            self.bottom,
            self.top_side,
            self.bottom_side,
            self.top_offset,
            self.bottom_offset,
            {key:tuple(value) for key, value in self.middle_regions.items()},
            self.state.copy(),
        )

    def _restore(self, snapshot):
        self.top, self.bottom, self.top_side, self.bottom_side, self.top_offset, self.bottom_offset, middle_regions, _state = snapshot
        self.middle_regions = {key:tuple(value) for key, value in middle_regions.items()}
        self._sync_state()

    def invert_str(self, move):
        return tuple(self.invert_moves((move,)))

    def invert_moves(self, Moves):
        inverse = []
        for move in self.normalize_move_sequence(Moves)[::-1]:
            u, d, slash = move
            if slash == "/":
                inverse.append((0, 0, "/"))
            if u != 0 or d != 0:
                inverse.append((-u, -d, None))
        return tuple(inverse)

    def simplify(self, move_lis):
        return self.normalize_move_sequence(move_lis)

    def reduce(self, move_lis):
        return (self.normalize_move_sequence(move_lis), tuple(range(len(move_lis))))

    def conjugate(self, A, B):
        return self.simplify(tuple(A) + tuple(B) + self.invert_moves(A))

    def commutator(self, A, B):
        return self.simplify(tuple(A) + tuple(B) + self.invert_moves(A) + self.invert_moves(B))

    def transform(self, s, i, flip_inside = False, invert = False):
        return self.normalize_move_sequence(s)

    def make_transformations(self, s, Moves):
        return [self.normalize_move_sequence(s)], [self.normalize_move_sequence(Moves)]

    def flip_inside_moves(self, Moves):
        return self.normalize_move_sequence(Moves)

    def collect_single_moves_and_rotate(self):
        return tuple((move,) for move in self.move_keys)

    def collect_single_move_and_rotate(self):
        return self.collect_single_moves_and_rotate()

    def get_correct_group_count(self, group_name):
        if group_name == "Corner":
            return sum(self._piece_slot_positions(piece.id) == self._solved_piece_slot_positions(piece.id) for piece in self.pieces if piece.kind == "Corner")
        if group_name == "Edge":
            return sum(self._piece_slot_positions(piece.id) == self._solved_piece_slot_positions(piece.id) for piece in self.pieces if piece.kind == "Edge")
        if group_name == "Shape":
            return int(sum(self.state[index] == self.state_0[index] for index in range(24)))
        return 0

    def get_correct_group_index(self, group_name):
        if group_name not in ("Corner", "Edge"):
            return []
        return [
            piece.id for piece in self.pieces
            if piece.kind == group_name and self._piece_slot_positions(piece.id) == self._solved_piece_slot_positions(piece.id)
        ]

    def _evaluate_piece_color_value(self, changed_piece_keys):
        return 0 if not changed_piece_keys else len(changed_piece_keys)

    def _update_piece_color_counter(self, changed_piece_keys):
        return None

    def _update_count(self, move_count, moves):
        for move in moves:
            move_count[move] = move_count.get(move, 0) + 1

    def _update_counter_stats(self, level, moves):
        return None

    def piece_display_name(self, piece_type, piece):
        return f"{piece_type}-{piece}"


Rubiks_3 = Square1Cube

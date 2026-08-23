"""Shared registered-scramble selection utilities."""


class ScrambleSelector:
    """Select registered scramble candidates while balancing move and piece usage."""

    def __init__(self, puzzle):
        self.puzzle = puzzle

    def init_move_count(self):
        return {move_key:0 for move_key in self.puzzle.move_keys}

    def select(self, level, move_count, move_count_policy = 'prefer_rare', add_moves = None):
        level = self.resolve_level(level)
        candidates = self.collect_candidates(level)
        if not candidates:
            return ()

        move_count_policy = self.resolve_move_count_policy(move_count_policy, add_moves)
        if move_count_policy == 'prefer_frequent':
            selected_moves = self._select_candidate_max(candidates, move_count, level)
        else:
            selected_moves = self._select_candidate_min(candidates, move_count, level)

        self.update_count(move_count, selected_moves)
        self.update_counter_stats(level, selected_moves)
        changed_piece_keys = self.puzzle.get_registered_scramble_changed_piece_keys(level, selected_moves)
        self.update_piece_color_counter(changed_piece_keys)
        return selected_moves

    def resolve_move_count_policy(self, move_count_policy, add_moves = None):
        if add_moves is not None:
            return 'prefer_frequent' if add_moves else 'prefer_rare'
        if isinstance(move_count_policy, bool):
            return 'prefer_frequent' if move_count_policy else 'prefer_rare'
        if move_count_policy in ('prefer_frequent', 'prefer_rare'):
            return move_count_policy
        raise ValueError(f'unknown move_count_policy: {move_count_policy}')

    def resolve_level(self, level):
        if level not in self.puzzle.my_scrambles2:
            return 0
        return level

    def collect_candidates(self, level):
        sort_level = 1 if 1 in self.puzzle.my_scrambles2 else 0
        move_keys = sorted(
            self.puzzle.move_keys,
            key = lambda move_key:len(self.puzzle.my_scrambles2[sort_level][move_key]),
        )
        candidates = []
        for move_key in move_keys:
            candidates += list(self.puzzle.my_scrambles2[level][move_key])
        return candidates

    def _select_candidate_max(self, candidates, move_count, level):
        top_score = -1
        top_piece_score = 0
        selected = ()
        for candidate in candidates:
            normalized_candidate = self.normalize_move_sequence(candidate)
            if len(self.puzzle.get_registered_scramble_changed_piece_keys(level, normalized_candidate)) == 0:
                score = -2
                piece_score = 0
            else:
                score = sum(move_count[move] for move in normalized_candidate) / len(normalized_candidate)
                piece_score = self.piece_color_value(level, normalized_candidate)

            if top_score < score or (top_score == score and top_piece_score > piece_score):
                top_score = score
                top_piece_score = piece_score
                selected = normalized_candidate
        return selected

    def _select_candidate_min(self, candidates, move_count, level):
        top_score = 1.0e+8
        top_piece_score = 0
        selected = ()
        for candidate in candidates:
            normalized_candidate = self.normalize_move_sequence(candidate)
            if len(self.puzzle.get_registered_scramble_changed_piece_keys(level, normalized_candidate)) == 0:
                score = 1.1e+8
                piece_score = 0
            else:
                score = sum(move_count[move] for move in normalized_candidate) / len(normalized_candidate)
                piece_score = self.piece_color_value(level, normalized_candidate)

            if top_score > score or (top_score == score and top_piece_score > piece_score):
                top_score = score
                top_piece_score = piece_score
                selected = normalized_candidate
        return selected

    def piece_color_value(self, level, moves):
        changed_piece_keys = self.puzzle.get_registered_scramble_changed_piece_keys(level, moves)
        if not changed_piece_keys:
            return 0
        return sum(self.puzzle.piece_color_counter.get(key, 0) for key in changed_piece_keys) / len(changed_piece_keys)

    def normalize_move_sequence(self, moves):
        if hasattr(self.puzzle, "normalize_move_sequence"):
            return self.puzzle.normalize_move_sequence(moves)
        return tuple(moves)

    def update_piece_color_counter(self, changed_piece_keys):
        if not changed_piece_keys:
            return
        for key in changed_piece_keys:
            self.puzzle.piece_color_counter[key] = self.puzzle.piece_color_counter.get(key, 0) + 1

    def update_count(self, move_count, moves):
        for move in moves:
            move_count[move] += 1

    def update_counter_stats(self, level, moves):
        if level == 0 or level >= 8:
            return
        self.puzzle.counter.setdefault(level, {})
        self.puzzle.counter[level][moves] = self.puzzle.counter[level].get(moves, 0) + 1

"""Deterministic, held-out probes for comparing AI checkpoints.

The online training loss changes its meaning as newly solved positions enter the
replay data.  These fixtures deliberately never enter that data: every run
uses the same seeded scrambles and their inverse move sequence as a canonical
teacher.  They are an inexpensive model-quality signal, not a replacement for
the direct-search benchmark.
"""

from __future__ import annotations

import random

import numpy as np


FIXTURE_ID = "canonical-inverse-v2"
# Keep the original short and medium probes, then add two long held-out
# trajectories.  They are reported separately so a gain on easy positions
# cannot conceal a plateau at 40+ moves.
FIXTURE_LENGTHS = (12, 24, 36, 48, 60)
FIXTURES_PER_LENGTH = 3
_FIXTURE_SEED = 20260926


def evaluate_fixed_validation(ai) -> dict[str, object]:
    """Evaluate one model on fixed, never-trained-on scramble trajectories.

    Policy labels are the next moves of one canonical inverse solution.  A
    Rubik's state can have multiple equally valid moves, so top-k is a stable
    proxy rather than a literal solve-rate.  Value quality is compared against
    the known remaining distance.  Standard Search2 values are ordinal within
    a path, while ``steps_to_goal`` is calibrated directly in move units and
    Search3 values are sigmoid probabilities; their calibration metrics are
    kept separate.
    """
    cube = ai.cube
    original_state = cube.state.copy()
    try:
        states, policy_targets, value_targets, path_slices, length_slices = _fixture_tensors(ai)
        outputs = ai.predict(states, policy = True, value = True, loss = False, retain_cache = False)
    finally:
        cube.state[:] = original_state

    policy_logits = np.asarray(outputs[:-1], dtype = "f")
    raw_values = np.asarray(outputs[-1], dtype = "f").reshape(-1)

    metrics: dict[str, object] = {
        "fixtureId": FIXTURE_ID,
        "fixtureLengths": list(FIXTURE_LENGTHS),
        "fixtureCount": FIXTURES_PER_LENGTH * len(FIXTURE_LENGTHS),
        "stateCount": int(policy_targets.size),
    }
    metrics.update(_quality_metrics(
        ai,
        policy_logits,
        raw_values,
        policy_targets,
        value_targets,
        path_slices,
    ))
    metrics["byLength"] = {
        str(length): {
            "fixtureCount": FIXTURES_PER_LENGTH,
            "stateCount": int(length_slice.stop - length_slice.start),
            **_quality_metrics(
                ai,
                policy_logits[:, length_slice],
                raw_values[length_slice],
                policy_targets[length_slice],
                value_targets[length_slice],
                _relative_path_slices(path_slices, length_slice),
            ),
        }
        for length, length_slice in length_slices.items()
    }
    return metrics


def _quality_metrics(ai, policy_logits, raw_values, policy_targets, value_targets, path_slices):
    """Return comparable policy/value quality metrics for one fixture group."""
    probabilities = _softmax_columns(policy_logits)
    columns = np.arange(policy_targets.size)
    ranks = np.argsort(np.argsort(-policy_logits, axis = 0), axis = 0)
    target_probabilities = probabilities[policy_targets, columns]
    top1 = ranks[policy_targets, columns] == 0
    top3 = ranks[policy_targets, columns] < min(3, policy_logits.shape[0])
    metrics = {
        "policyTop1Accuracy": float(np.mean(top1)),
        "policyTop3Accuracy": float(np.mean(top3)),
        "policyTargetProbability": float(np.mean(target_probabilities)),
        "policyCrossEntropy": float(np.mean(-np.log(target_probabilities + 1.0e-7))),
        "valueRankCorrelation": _spearman(raw_values, value_targets),
        "valuePearsonCorrelation": _pearson(raw_values, value_targets),
        "valuePredictionMean": float(np.mean(raw_values)),
        "valuePredictionStd": float(np.std(raw_values)),
        "valueTargetMean": float(np.mean(value_targets)),
        "valueTargetStd": float(np.std(value_targets)),
    }
    if getattr(ai, "search_mode", "") == "search3":
        predicted_probability = _sigmoid(raw_values)
        metrics["valueMae"] = float(np.mean(np.abs(predicted_probability - value_targets)))
        metrics["valueBce"] = _binary_cross_entropy(predicted_probability, value_targets)
    elif getattr(ai, "search2_value_loss_type", "") == "steps_to_goal":
        metrics["valueMae"] = float(np.mean(np.abs(raw_values - value_targets)))
        metrics["valuePathCrossEntropy"] = None
    else:
        metrics["valuePathCrossEntropy"] = _search2_path_cross_entropy(
            raw_values,
            value_targets,
            path_slices,
        )
    return metrics


def _fixture_tensors(ai):
    """Build fixture state columns and canonical next-move/value targets."""
    cube = ai.cube
    state_columns = []
    policy_targets = []
    value_targets = []
    path_slices = []
    length_slices = {}
    gamma = float(getattr(ai, "value_target_gamma", (1 / 2) ** (1 / 20)))
    steps_to_goal = getattr(ai, "search2_value_loss_type", "") == "steps_to_goal"
    target_scale = float(getattr(ai, "search2_value_target_scale", 1.0))

    for length in FIXTURE_LENGTHS:
        length_start = len(policy_targets)
        for fixture_index in range(FIXTURES_PER_LENGTH):
            scramble = _fixture_scramble(cube, length, fixture_index)
            solution = tuple(cube.invert_moves(scramble))
            start = len(policy_targets)
            cube.reset()
            cube.scramble(0, scramble)
            for move_index, move in enumerate(solution):
                state_columns.append(cube.makedata())
                policy_targets.append(cube.key_to_num[move])
                remaining_steps = len(solution) - move_index
                value_targets.append(
                    -target_scale * remaining_steps
                    if steps_to_goal else gamma ** remaining_steps
                )
                cube.make_move(move)
            path_slices.append(slice(start, len(policy_targets)))
        length_slices[length] = slice(length_start, len(policy_targets))

    return (
        np.asarray(state_columns, dtype = "f").T,
        np.asarray(policy_targets, dtype = int),
        np.asarray(value_targets, dtype = "f"),
        tuple(path_slices),
        length_slices,
    )


def _relative_path_slices(path_slices, outer_slice):
    """Translate trajectory slices into the coordinates of a fixture group."""
    start = outer_slice.start
    stop = outer_slice.stop
    return tuple(
        slice(path_slice.start - start, path_slice.stop - start)
        for path_slice in path_slices
        if path_slice.start >= start and path_slice.stop <= stop
    )


def _fixture_scramble(cube, length: int, fixture_index: int) -> tuple[str, ...]:
    """Build a deterministic outer-face scramble without global RNG effects."""
    candidates = [
        move for move in cube.move_keys
        if move.startswith(" ") and move.strip()[:1] in {"R", "L", "U", "D", "F", "B"}
    ]
    if not candidates:
        candidates = [move for move in cube.move_keys if move.strip()[:1] not in {"x", "y", "z"}]
    if not candidates:
        raise ValueError("Fixed validation requires at least one non-rotation move")

    generator = random.Random(_FIXTURE_SEED + length * 101 + fixture_index)
    moves = []
    previous_face = None
    for _ in range(length):
        available = [move for move in candidates if _face(move) != previous_face]
        move = generator.choice(available or candidates)
        moves.append(move)
        previous_face = _face(move)
    return tuple(moves)


def _face(move: str) -> str:
    return move.strip()[-1:].replace("'", "").replace("2", "") or move.strip()[:1]


def _softmax_columns(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis = 0, keepdims = True)
    exponentials = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponentials / np.sum(exponentials, axis = 0, keepdims = True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _binary_cross_entropy(predicted: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(predicted, 1.0e-7, 1.0 - 1.0e-7)
    return float(np.mean(-(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))))


def _search2_path_cross_entropy(values, targets, path_slices) -> float:
    """Evaluate the Search2 ordinal value distribution without assuming scale."""
    losses = []
    for path_slice in path_slices:
        path_values = values[path_slice]
        path_targets = targets[path_slice]
        if path_values.size == 0:
            continue
        predicted = _softmax_columns(path_values.reshape(-1, 1)).reshape(-1)
        target_distribution = path_targets / np.sum(path_targets)
        losses.append(float(-np.sum(target_distribution * np.log(predicted + 1.0e-7))))
    return float(np.mean(losses)) if losses else 0.0


def _pearson(values, targets) -> float | None:
    if values.size < 2 or np.std(values) == 0.0 or np.std(targets) == 0.0:
        return None
    return float(np.corrcoef(values, targets)[0, 1])


def _spearman(values, targets) -> float | None:
    return _pearson(_average_ranks(values), _average_ranks(targets))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending average ranks while giving tied values the same rank."""
    order = np.argsort(values, kind = "mergesort")
    ranks = np.empty(values.size, dtype = "f")
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks

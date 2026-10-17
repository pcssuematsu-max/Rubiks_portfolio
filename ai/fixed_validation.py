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

from core.experiment_log import SETUP_EFFECT_BANDS, setup_effect_metrics


FIXTURE_ID = "canonical-inverse-v3"
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
    the canonical inverse-route length, not necessarily the shortest distance.
    Standard Search2 values are ordinal within
    a path, while ``steps_to_goal`` is calibrated directly in move units and
    Search3 values are sigmoid probabilities; their calibration metrics are
    kept separate.
    """
    cube = ai.cube
    original_state = cube.state.copy()
    try:
        states, policy_targets, value_targets, remaining_steps, path_slices, length_slices, effects, snapshots = _fixture_tensors(ai)
        outputs = ai.predict(states, policy = True, value = True, loss = False, retain_cache = False)
        branch_probe = _transformer_branch_probe(ai, states, outputs, policy_targets, value_targets, path_slices)
        choice_probe = _effect_choice_metrics(ai, snapshots, policy_targets, effects)
        exact_distance = (
            _exact_distance_anchor_metrics(ai)
            if getattr(ai, "search2_value_loss_type", "") == "steps_to_goal" else None
        )
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
    # A 60-move probe alone is sparse and noisy.  Keep the individual lengths
    # for diagnosis, and also expose a stable held-out 40+ bucket (48/60) for
    # deciding whether Search2 is learning genuinely difficult positions.
    long_start = length_slices[48].start
    long_stop = length_slices[60].stop
    long_slice = slice(long_start, long_stop)
    metrics["long40Plus"] = {
        "fixtureCount": FIXTURES_PER_LENGTH * 2,
        "stateCount": int(long_stop - long_start),
        **_quality_metrics(
            ai,
            policy_logits[:, long_slice],
            raw_values[long_slice],
            policy_targets[long_slice],
            value_targets[long_slice],
            _relative_path_slices(path_slices, long_slice),
        ),
    }
    if effects is not None:
        metrics["byEffect"] = _effect_decision_metrics(
            policy_logits, raw_values, policy_targets, path_slices, effects,
        )
    if choice_probe is not None:
        metrics["byEffectChoices"] = choice_probe
    if branch_probe is not None:
        metrics["transformerBranches"] = branch_probe
    if getattr(ai, "search2_value_loss_type", "") == "steps_to_goal":
        metrics["stepsToGoalExactDistance"] = exact_distance
        metrics["stepsToGoalByRemainingSteps"] = _steps_to_goal_remaining_step_metrics(
            ai, policy_logits, raw_values, policy_targets, value_targets, remaining_steps,
        )
    return metrics


def _exact_distance_anchor_metrics(ai):
    """Evaluate states with provable distance zero or one from the goal."""
    cube = ai.cube
    cube.reset()
    columns = [cube.makedata()]
    for move in cube.move_keys:
        cube.reset()
        cube.make_move(move)
        if not cube.is_perfect():
            columns.append(cube.makedata())
    outputs = ai.predict(
        np.asarray(columns,dtype = "f").T,
        policy = False,value = True,loss = False,retain_cache = False,
    )
    predicted = np.asarray(outputs[-1],dtype = "f").reshape(-1)
    scale = float(getattr(ai,"search2_value_target_scale",1.0))
    one_move_error = predicted[1:] + scale
    return {
        "stateCount": int(predicted.size),
        "goalValueError": float(predicted[0]),
        "goalValueMae": float(abs(predicted[0])),
        "oneMoveValueBias": float(np.mean(one_move_error)),
        "oneMoveValueMae": float(np.mean(np.abs(one_move_error))),
    }


def _steps_to_goal_remaining_step_metrics(ai, policy_logits, raw_values, policy_targets, value_targets, remaining_steps):
    """Expose calibration separately near, mid, and far from the goal."""
    bands = (("0-10",0,10),("11-30",11,30),("31+",31,None))
    result = {}
    for label,minimum,maximum in bands:
        mask = remaining_steps >= minimum
        if maximum is not None:
            mask &= remaining_steps <= maximum
        count = int(np.sum(mask))
        if count == 0:
            continue
        result[label] = {
            "stateCount": count,
            **_quality_metrics(
                ai,policy_logits[:,mask],raw_values[mask],policy_targets[mask],
                value_targets[mask],(),
            ),
        }
    return result


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
        errors = raw_values - value_targets
        metrics["valueMae"] = float(np.mean(np.abs(errors)))
        metrics["valueSignedError"] = float(np.mean(errors))
        metrics["valueRmse"] = float(np.sqrt(np.mean(errors ** 2)))
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
    remaining_steps_list = []
    effect_counts = []
    snapshots = []
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
                snapshots.append(cube.state.copy())
                policy_targets.append(cube.key_to_num[move])
                # The inverse of the remaining solution reconstructs this
                # state from the goal; it never enters replay or training.
                effect = setup_effect_metrics(
                    "cube-3x3", getattr(ai, "cube_size", 3),
                    cube.invert_moves(solution[move_index:]),
                )
                effect_counts.append(None if effect is None else effect["movedPieceCount"])
                remaining_steps = len(solution) - move_index
                remaining_steps_list.append(remaining_steps)
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
        np.asarray(remaining_steps_list, dtype = int),
        tuple(path_slices),
        length_slices,
        None if any(effect is None for effect in effect_counts) else np.asarray(effect_counts, dtype = int),
        tuple(snapshots),
    )


def _effect_decision_metrics(policy_logits, raw_values, policy_targets, path_slices, effects):
    """Measure nonterminal raw-Value progress, grouped by net piece effect."""
    columns = np.arange(policy_targets.size)
    ranks = np.argsort(np.argsort(-policy_logits, axis = 0), axis = 0)
    target_ranks = ranks[policy_targets, columns] + 1
    next_values = np.empty_like(raw_values)
    nonterminal = np.zeros(raw_values.size,dtype = bool)
    for path_slice in path_slices:
        next_values[path_slice.start:path_slice.stop - 1] = raw_values[path_slice.start + 1:path_slice.stop]
        nonterminal[path_slice.start:path_slice.stop - 1] = True
    deltas = next_values - raw_values
    result = {}
    for minimum, maximum, label in SETUP_EFFECT_BANDS:
        mask = effects >= minimum
        if maximum is not None:
            mask &= effects <= maximum
        count = int(np.count_nonzero(mask))
        if count:
            value_mask = mask & nonterminal
            value_count = int(np.count_nonzero(value_mask))
            result[label] = {
                "stateCount": count,
                "valueStepCount": value_count,
                "policyTargetRankMean": float(np.mean(target_ranks[mask])),
                "policyTop1Accuracy": float(np.mean(target_ranks[mask] == 1)),
                "valueDeltaMean": float(np.mean(deltas[value_mask])) if value_count else None,
                "valueImprovementRate": float(np.mean(deltas[value_mask] > 0.0)) if value_count else None,
            }
    return result


def _effect_choice_metrics(ai, snapshots, policy_targets, effects):
    """Check whether Value prefers the held-out correct successor to siblings.

    Four deterministic positions per Effect band keep this much cheaper than
    expanding all 540 fixture states.  No successor enters replay or training.
    """
    if effects is None or getattr(ai, 'cube_size', 3) != 3:
        return None
    selected = {}
    for minimum, maximum, label in SETUP_EFFECT_BANDS:
        mask = effects >= minimum
        if maximum is not None:
            mask &= effects <= maximum
        candidates = np.flatnonzero(mask)
        if candidates.size:
            positions = np.linspace(0,candidates.size - 1,min(4,candidates.size),dtype=int)
            selected[label] = [int(candidates[position]) for position in positions]
    if not selected:
        return None
    cube = ai.cube
    move_keys = tuple(cube.move_keys)
    columns = []
    targets = []
    labels = []
    for label, indices in selected.items():
        for index in indices:
            snapshot = snapshots[index]
            for move in move_keys:
                cube.state[:] = snapshot
                cube.make_move(move)
                columns.append(cube.makedata())
            targets.append(int(policy_targets[index]))
            labels.append(label)
    raw = []
    for start in range(0,len(columns),64):
        batch = np.asarray(columns[start:start + 64],dtype='f').T
        raw.extend(np.asarray(ai.predict(
            batch,policy=False,value=True,loss=False,retain_cache=False,
        ),dtype='f').reshape(-1).tolist())
    values = np.asarray(raw,dtype='f').reshape(len(targets),len(move_keys))
    result = {}
    for label in selected:
        ranks = []
        margins = []
        for row_index, row_label in enumerate(labels):
            if row_label != label:
                continue
            target = targets[row_index]
            correct = values[row_index,target]
            other = np.delete(values[row_index],target)
            ranks.append(int(np.count_nonzero(other > correct) + 1))
            margins.append(float(correct - np.max(other)))
        result[label] = {
            'sampleCount': len(ranks),
            'valueTargetRankMean': float(np.mean(ranks)),
            'valueTop1Accuracy': float(np.mean(np.asarray(ranks) == 1)),
            'valueCorrectVsBestOtherDeltaMean': float(np.mean(margins)),
        }
    return result


def _transformer_branch_probe(ai, states, full_outputs, policy_targets, value_targets, path_slices):
    """Small, read-only ablations of the attention and direct Linear branches."""
    evaluator = getattr(ai, "diagnose_first_layer_branches", None)
    if not callable(evaluator):
        return None
    selected = sorted({index for path in path_slices for index in (
        path.start, (path.start + path.stop - 1) // 2, path.stop - 1,
    )})
    indices = np.asarray(selected, dtype = int)
    probe = evaluator(states[:, indices])
    if probe is None:
        return None
    result = {key: probe[key] for key in (
        "attentionRms", "linearRms", "linearToAttentionRms", "preactivationCosine",
    )}
    result["sampleCount"] = len(selected)
    for label, outputs in (
        ("full", np.asarray(full_outputs)[:, indices]),
        ("withoutAttention", probe["withoutAttention"]),
        ("withoutLinear", probe["withoutLinear"]),
    ):
        logits = np.asarray(outputs[:-1], dtype = "f")
        values = np.asarray(outputs[-1], dtype = "f").reshape(-1)
        ranks = np.argsort(np.argsort(-logits, axis = 0), axis = 0)
        target_ranks = ranks[policy_targets[indices], np.arange(len(indices))] + 1
        result[label] = {
            "policyTargetRankMean": float(np.mean(target_ranks)),
            "policyTop1Accuracy": float(np.mean(target_ranks == 1)),
            "valueRankCorrelation": _spearman(values, value_targets[indices]),
        }
    return result


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

"""Persist per-AI learning metrics for later inspection in the GUI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


LEARNING_HISTORY_SCHEMA_VERSION = 1
LEARNING_HISTORY_FILE_NAME = "ai-learning-history.json"
# The history chart displays the latest 60 learning events; retain two full
# chart windows per AI so the file stays quick to parse and rewrite.
LEARNING_HISTORY_RECORDS_PER_AI = 120
_PAYLOAD_FIELDS = frozenset({"schemaVersion", "updatedAt", "records"})


def default_learning_history_path() -> Path:
    """Return the local JSON history path, optionally overridden for experiments."""
    configured_path = os.environ.get("TWISTY_LEARNING_HISTORY_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parents[1] / "exports" / LEARNING_HISTORY_FILE_NAME


class LearningHistoryStore:
    """Append validated learning records to one readable JSON document."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_learning_history_path()

    def append(self, record: dict[str, Any]) -> None:
        """Persist one completed learning operation without discarding older points."""
        self.append_many((record,))

    def append_many(self, records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
        """Persist a completed learning pass with one read/write cycle."""
        records = list(records)
        if len(records) == 0:
            return
        payload = self._read()
        for record in records:
            _validate_record(record)
        payload["records"].extend(records)
        payload["records"] = _compact_records(payload["records"])
        payload["updatedAt"] = records[-1]["timestamp"]
        self._write(payload)

    def compact(self) -> tuple[int, int]:
        """Keep the recent, graph-visible history window for each AI."""
        payload = self._read()
        original_count = len(payload["records"])
        payload["records"] = _compact_records(payload["records"])
        if payload["records"]:
            payload["updatedAt"] = payload["records"][-1]["timestamp"]
        self._write(payload)
        return original_count, len(payload["records"])

    def records(self) -> list[dict[str, Any]]:
        """Return a copy of the complete, chronological history."""
        return list(self._read()["records"])

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _new_payload()
        try:
            with self.path.open(encoding = "utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid learning history: {self.path}") from error
        _validate_payload(payload)
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        _validate_payload(payload)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding = "utf-8") as stream:
            json.dump(payload, stream, ensure_ascii = False, separators = (",", ":"))
            stream.write("\n")
        temporary_path.replace(self.path)


def _compact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain a bounded chronological window without mixing AI histories."""
    retained_by_ai: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        ai_index = int(record["aiIndex"])
        bucket = retained_by_ai.setdefault(ai_index, [])
        bucket.append(record)
        if len(bucket) > LEARNING_HISTORY_RECORDS_PER_AI:
            del bucket[0]
    retained = [record for bucket in retained_by_ai.values() for record in bucket]
    retained.sort(key = lambda record: record["timestamp"])
    return retained


def completed_learning_record(ai_index: int, ai, elapsed_seconds: float) -> dict[str, Any]:
    """Capture one AI's completed learning metrics in JSON-friendly form."""
    metrics = dict(getattr(ai, "last_training_metrics", None) or {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
        "aiIndex": int(ai_index),
        "searchMode": str(getattr(ai, "search_mode", "unknown")),
        "policyLoss": _finite_number(metrics.get("policyLoss")),
        "valueLoss": _finite_number(metrics.get("valueLoss")),
        "policyCePerState": _finite_number(metrics.get("policyCePerState")),
        "policyEffectiveStateCount": _finite_number(metrics.get("policyEffectiveStateCount")),
        "valueBcePerState": _finite_number(metrics.get("valueBcePerState")),
        "valueMae": _finite_number(metrics.get("valueMae")),
        "valueStartToEndDelta": _finite_number(metrics.get("valueStartToEndDelta")),
        "valueTargetStartToEndDelta": _finite_number(metrics.get("valueTargetStartToEndDelta")),
        "valueEffectiveStateCount": _finite_number(metrics.get("valueEffectiveStateCount")),
        "valueSequenceCount": _safe_optional_int(metrics.get("valueSequenceCount")),
        "fixedValidation": _fixed_validation(metrics.get("fixedValidation")),
        "trainingSample": _training_sample(metrics.get("trainingSample")),
        "trainingDataSources": _training_data_sources(metrics.get("trainingDataSources")),
        "search3TrainingWeights": _search3_training_weights(metrics.get("search3TrainingWeights")),
        "gradientMetrics": _gradient_metrics(metrics.get("gradientMetrics")),
        "stepsToGoalRegression": _steps_to_goal_regression(metrics.get("stepsToGoalRegression")),
        "search2BaseLossPerItem": _finite_number(metrics.get("search2BaseLossPerItem")),
        "search2RankLossPerItem": _finite_number(metrics.get("search2RankLossPerItem")),
        "search2PlateauEscapeLossPerItem": _finite_number(metrics.get("search2PlateauEscapeLossPerItem")),
        "updatesDuringSolve": int(metrics.get("updatesDuringSolve", 0) or 0),
        "trainingDataCount": int(metrics.get("trainingDataCount", 0) or 0),
        "retainedDataCount": int(metrics.get("retainedDataCount", 0) or 0),
        "learningSeconds": round(max(0.0, float(elapsed_seconds)), 6),
        "learningRate": {
            "base": _finite_number(getattr(ai, "lr", None)),
            "lrC": _finite_number(getattr(ai, "lr_C", None)),
            "momentumV": _finite_number(getattr(ai, "lr_v", None)),
            "momentumH": _finite_number(getattr(ai, "lr_h", None)),
        },
        "updateScales": {
            "shared": _finite_number(getattr(ai, "update_scale_shared", None)),
            "policy": _finite_number(getattr(ai, "update_scale_policy", None)),
            "value": _finite_number(getattr(ai, "update_scale_value", None)),
        },
        "search2ValueLossType": str(getattr(ai, "search2_value_loss_type", "")),
        "search2ValueLossMargin": _finite_number(getattr(ai, "search2_value_loss_margin", None)),
        "search2RankLossMix": _finite_number(getattr(ai, "search2_rank_loss_mix", None)),
        "search2RankLossApplyType": str(getattr(ai, "search2_rank_loss_apply_type", "")),
        "search3RankLossMix": _finite_number(getattr(ai, "search3_rank_loss_mix", None)),
    }


def _new_payload() -> dict[str, Any]:
    return {
        "schemaVersion": LEARNING_HISTORY_SCHEMA_VERSION,
        "updatedAt": None,
        "records": [],
    }


def _validate_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("Invalid learning history payload")
    if payload["schemaVersion"] != LEARNING_HISTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported learning history schemaVersion")
    if payload["updatedAt"] is not None and not isinstance(payload["updatedAt"], str):
        raise ValueError("Invalid learning history updatedAt")
    if not isinstance(payload["records"], list):
        raise ValueError("Invalid learning history records")
    for record in payload["records"]:
        _validate_record(record)


def _validate_record(record: Any) -> None:
    required = {
        "timestamp", "aiIndex", "searchMode", "policyLoss", "valueLoss",
        "updatesDuringSolve", "trainingDataCount", "retainedDataCount",
        "learningSeconds", "learningRate", "updateScales", "search2ValueLossType",
    }
    optional = {
        # Added after the first history format shipped.  These remain optional
        # so existing local history files stay readable.
        "policyCePerState", "policyEffectiveStateCount",
        "valueBcePerState", "valueMae", "valueStartToEndDelta",
        "valueTargetStartToEndDelta", "valueEffectiveStateCount",
        "valueSequenceCount", "fixedValidation", "trainingSample", "trainingDataSources", "search3TrainingWeights", "gradientMetrics", "stepsToGoalRegression", "search3RankLossMix",
        "search2BaseLossPerItem", "search2RankLossPerItem", "search2PlateauEscapeLossPerItem", "search2ValueLossMargin", "search2RankLossMix", "search2RankLossApplyType",
    }
    if not isinstance(record, dict) or not required.issubset(record) or set(record) - required - optional:
        raise ValueError("Invalid learning history record")
    if not isinstance(record["timestamp"], str) or not record["timestamp"]:
        raise ValueError("Invalid learning history timestamp")
    for field in ("aiIndex", "updatesDuringSolve", "trainingDataCount", "retainedDataCount"):
        if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
            raise ValueError(f"Invalid learning history {field}")
    if not isinstance(record["searchMode"], str) or not isinstance(record["search2ValueLossType"], str):
        raise ValueError("Invalid learning history search mode")
    _validate_optional_number(record["policyLoss"])
    _validate_optional_number(record["valueLoss"])
    for field in (
        "policyCePerState", "policyEffectiveStateCount",
        "valueBcePerState", "valueMae", "valueStartToEndDelta",
        "valueTargetStartToEndDelta", "valueEffectiveStateCount",
    ):
        _validate_optional_number(record.get(field))
    value_sequence_count = record.get("valueSequenceCount")
    if value_sequence_count is not None and (
        isinstance(value_sequence_count,bool)
        or not isinstance(value_sequence_count,int)
        or value_sequence_count < 0
    ):
        raise ValueError("Invalid learning history valueSequenceCount")
    _validate_fixed_validation(record.get("fixedValidation"))
    _validate_training_sample(record.get("trainingSample"))
    _validate_training_data_sources(record.get("trainingDataSources"))
    _validate_search3_training_weights(record.get("search3TrainingWeights"))
    _validate_gradient_metrics(record.get("gradientMetrics"))
    _validate_steps_to_goal_regression(record.get("stepsToGoalRegression"))
    for field in ("search2BaseLossPerItem", "search2RankLossPerItem", "search2PlateauEscapeLossPerItem", "search2ValueLossMargin", "search2RankLossMix"):
        _validate_optional_number(record.get(field))
    if "search2RankLossApplyType" in record and not isinstance(record["search2RankLossApplyType"], str):
        raise ValueError("Invalid learning history search2RankLossApplyType")
    _validate_optional_number(record.get("search3RankLossMix"))
    _validate_optional_number(record["learningSeconds"])
    _validate_number_mapping(record["learningRate"], {"base", "lrC", "momentumV", "momentumH"})
    _validate_number_mapping(record["updateScales"], {"shared", "policy", "value"})


def _validate_number_mapping(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Invalid learning history numeric mapping")
    for number in value.values():
        _validate_optional_number(number)


def _fixed_validation(value: Any) -> dict[str, Any] | None:
    """Copy the compact held-out metrics while dropping invalid numeric values."""
    if not isinstance(value, dict):
        return None
    fixture_id = value.get("fixtureId")
    fixture_lengths = value.get("fixtureLengths")
    if not isinstance(fixture_id, str) or not fixture_id or not isinstance(fixture_lengths, (tuple, list)):
        return None
    if any(isinstance(length, bool) or not isinstance(length, int) or length < 1 for length in fixture_lengths):
        return None
    result: dict[str, Any] = {
        "fixtureId": fixture_id,
        "fixtureLengths": list(fixture_lengths),
        "fixtureCount": _safe_optional_int(value.get("fixtureCount")),
        "stateCount": _safe_optional_int(value.get("stateCount")),
    }
    for key in _FIXED_VALIDATION_NUMBERS:
        result[key] = _finite_number(value.get(key))
    for key in _FIXED_VALIDATION_OPTIONAL_NUMBERS:
        if key in value:
            result[key] = _finite_number(value[key])
    by_length = _fixed_validation_by_length(value.get("byLength"))
    if by_length is not None:
        result["byLength"] = by_length
    long_40_plus = _fixed_validation_bucket(value.get("long40Plus"))
    if long_40_plus is not None:
        result["long40Plus"] = long_40_plus
    remaining_steps = _fixed_validation_remaining_steps(value.get("stepsToGoalByRemainingSteps"))
    if remaining_steps is not None:
        result["stepsToGoalByRemainingSteps"] = remaining_steps
    exact_distance = _fixed_validation_exact_distance(value.get("stepsToGoalExactDistance"))
    if exact_distance is not None:
        result["stepsToGoalExactDistance"] = exact_distance
    by_effect = _fixed_validation_by_effect(value.get("byEffect"))
    if by_effect is not None:
        result["byEffect"] = by_effect
    by_effect_choices = _fixed_validation_by_effect_choices(value.get("byEffectChoices"))
    if by_effect_choices is not None:
        result["byEffectChoices"] = by_effect_choices
    branches = _fixed_validation_transformer_branches(value.get("transformerBranches"))
    if branches is not None:
        result["transformerBranches"] = branches
    return result


_FIXED_VALIDATION_NUMBERS = frozenset({
    "policyTop1Accuracy", "policyTop3Accuracy", "policyTargetProbability",
    "policyCrossEntropy", "valueRankCorrelation", "valuePearsonCorrelation",
    "valuePredictionMean", "valuePredictionStd", "valueTargetMean",
    "valueTargetStd", "valueMae", "valueBce", "valuePathCrossEntropy",
})

# Regression-only metrics were added after the first fixed-validation records.
# Keep them optional so older histories and non-regression AIs remain readable.
_FIXED_VALIDATION_OPTIONAL_NUMBERS = frozenset({
    "valueSignedError", "valueRmse",
})


_TRAINING_SAMPLE_INT_FIELDS = frozenset({
    "originalBatchCount", "selectedBatchCount", "recentBatchCount",
    "longBatchCount", "randomBatchCount", "originalItemCount",
    "selectedItemCount", "selectedStateCount", "remainderItemCount",
    "longSequenceMinSteps", "longEligibleItemCount", "longReservedItemCount",
    "longSelectedItemCount", "longSelectedStepMax",
})

# These fields were introduced after sampled replay was first persisted.
# They must stay optional: users can keep opening existing local histories
# whose records only contain the original long-replay counters.
_TRAINING_SAMPLE_OPTIONAL_INT_FIELDS = frozenset({
    "shortSequenceMaxSteps", "shortEligibleItemCount", "shortSelectedItemCount",
    "mediumBatchCount", "mediumSequenceMinSteps", "mediumSequenceMaxSteps",
    "mediumEligibleItemCount", "mediumReservedItemCount", "mediumSelectedItemCount",
    "directSearchSelectedItemCount", "fallbackSelectedItemCount",
    "stepsToGoalStratifiedBatchCount", "stepsToGoalStratifiedItemCount",
    "stepsToGoalLongRouteEligibleItemCount", "stepsToGoalLongRouteRequiredItemCount",
    "stepsToGoalLongRouteSelectedItemCount", "stepsToGoalLongRouteOversampledItemCount",
    "originalLongDirectEligibleItemCount", "originalLongDirectRequiredItemCount",
    "originalLongDirectSelectedItemCount", "originalLongDirectOversampledItemCount",
    "hardEffectDirectEligibleItemCount", "hardEffectDirectRequiredItemCount",
    "hardEffectDirectSelectedBeforeItemCount", "hardEffectDirectSelectedItemCount",
    "hardEffectDirectOversampledItemCount",
    "plateauFallbackEligibleItemCount", "plateauFallbackRequiredItemCount",
    "plateauFallbackSelectedItemCount", "plateauFallbackOversampledItemCount",
    "recentWindowSelectedItemCount", "mixedBatchCount", "selectedUniqueItemCount",
})

_TRAINING_SAMPLE_NUMBER_FIELDS = frozenset({
    "shortReplayMinRatio", "shortSelectedRatio",
    "longReplayRatio", "longReplayMaxRatio", "longSelectedRatio",
    "stepsToGoalStratifiedRatio",
    "stepsToGoalLongRouteMinRatio",
    "originalLongDirectMinRatio", "fallbackSelectedRatio", "pairwiseFallbackMaxRatio",
    "hardEffectDirectMinRatio",
    "recentReplayRatio",
})


def _training_data_sources(value: Any) -> dict[str, dict[str, dict[str, int]]] | None:
    """Copy source-family counts for all candidates and selected batches."""
    if not isinstance(value, dict) or set(value) != {"available", "selected"}:
        return None
    result = {}
    for population in ("available", "selected"):
        rows = value.get(population)
        if not isinstance(rows, dict):
            return None
        copied = {}
        for source, metrics in rows.items():
            if not isinstance(source, str) or not source or not isinstance(metrics, dict):
                return None
            trajectories = _safe_optional_int(metrics.get("trajectoryCount"))
            states = _safe_optional_int(metrics.get("stateCount"))
            if trajectories is None or states is None or set(metrics) != {"trajectoryCount", "stateCount"}:
                return None
            copied[source] = {"trajectoryCount": trajectories, "stateCount": states}
        result[population] = copied
    return result


def _validate_training_data_sources(value: Any) -> None:
    if value is None:
        return
    normalized = _training_data_sources(value)
    if normalized is None or normalized != value:
        raise ValueError("Invalid learning history trainingDataSources")


def _search3_training_weights(value: Any) -> dict[str, dict[str, int | float]] | None:
    """Keep only aggregate selected-route weights; no individual states."""
    if not isinstance(value, dict):
        return None
    result = {}
    for source, metrics in value.items():
        if not isinstance(source, str) or not source or not isinstance(metrics, dict):
            return None
        if set(metrics) != {"trajectoryCount", "stateCount", "policyWeightSum", "valueWeightSum"}:
            return None
        trajectories = _safe_optional_int(metrics["trajectoryCount"])
        states = _safe_optional_int(metrics["stateCount"])
        policy_weight = _finite_number(metrics["policyWeightSum"])
        value_weight = _finite_number(metrics["valueWeightSum"])
        if (trajectories is None or states is None or policy_weight is None
                or value_weight is None or policy_weight < 0 or value_weight < 0):
            return None
        result[source] = {
            "trajectoryCount": trajectories,
            "stateCount": states,
            "policyWeightSum": policy_weight,
            "valueWeightSum": value_weight,
        }
    return result


def _validate_search3_training_weights(value: Any) -> None:
    if value is None:
        return
    normalized = _search3_training_weights(value)
    if normalized is None or normalized != value:
        raise ValueError("Invalid learning history search3TrainingWeights")


_GRADIENT_METRIC_NUMBER_FIELDS = frozenset({
    "globalL2Mean", "globalL2Max", "globalRmsMean",
    "policyHeadL2Mean", "valueHeadL2Mean", "trunkL2Mean",
    "policyGradientShareMean", "valueGradientShareMean",
})


_STEPS_TO_GOAL_REGRESSION_NUMBER_FIELDS = frozenset({
    "targetMin", "targetMean", "targetMax",
    "predictionMin", "predictionMean", "predictionMax",
    "signedErrorMean", "maePerState", "huberPerState",
})


_STEPS_TO_GOAL_REPLAY_POPULATION_REQUIRED_INT_FIELDS = frozenset({
    "routeCount", "duplicateStartStateCount", "duplicateStartStateRouteCount",
    "conflictingStartStateCount", "conflictingStartStateRouteCount",
})

_STEPS_TO_GOAL_REPLAY_POPULATION_OPTIONAL_INT_FIELDS = frozenset({
    "shortestTargetRouteCount", "longerDuplicateRouteCount",
})

_STEPS_TO_GOAL_VALUE_SELECTION_INT_FIELDS = frozenset({
    "candidateRouteCount", "selectedRouteCount", "sourceExcludedRouteCount",
    "detourExcludedRouteCount", "longerDuplicateExcludedRouteCount",
    "oversampledStateCount",
})

_STEPS_TO_GOAL_VALUE_SELECTION_OPTIONAL_INT_FIELDS = frozenset({
    "conflictingIntermediateStateCount", "discardedNonshortestStateCount",
    "excludedAmbiguousValueStateCount",
})


def _steps_to_goal_replay_population(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    required = set(_STEPS_TO_GOAL_REPLAY_POPULATION_REQUIRED_INT_FIELDS) | {
        "remainingStepBands", "efficiencyBands", "routeEfficiencyMean",
        "duplicateTargetStdMean",
    }
    allowed = required | set(_STEPS_TO_GOAL_REPLAY_POPULATION_OPTIONAL_INT_FIELDS)
    if not required.issubset(value) or set(value) - allowed:
        return None
    result = {}
    for key in _STEPS_TO_GOAL_REPLAY_POPULATION_REQUIRED_INT_FIELDS:
        count = _safe_optional_int(value.get(key))
        if count is None or count < 0:
            return None
        result[key] = count
    for key in _STEPS_TO_GOAL_REPLAY_POPULATION_OPTIONAL_INT_FIELDS:
        if key not in value:
            continue
        count = _safe_optional_int(value.get(key))
        if count is None or count < 0:
            return None
        result[key] = count
    bands = value.get("remainingStepBands")
    expected_bands = {"0-10", "11-30", "31+"}
    if not isinstance(bands, dict) or set(bands) != expected_bands:
        return None
    result["remainingStepBands"] = {label: _safe_optional_int(bands[label]) for label in expected_bands}
    if any(count is None or count < 0 for count in result["remainingStepBands"].values()):
        return None
    efficiency = value.get("efficiencyBands")
    expected_efficiency = {"efficient", "medium", "detour", "unknown"}
    if not isinstance(efficiency, dict) or set(efficiency) != expected_efficiency:
        return None
    result["efficiencyBands"] = {label: _safe_optional_int(efficiency[label]) for label in expected_efficiency}
    if any(count is None or count < 0 for count in result["efficiencyBands"].values()):
        return None
    for key in ("routeEfficiencyMean", "duplicateTargetStdMean"):
        number = _finite_number(value.get(key))
        if value.get(key) is not None and (number is None or number < 0.0):
            return None
        result[key] = number
    return result


def _steps_to_goal_replay(value: Any) -> dict[str, Any] | None:
    required_fields = {
        "stratifiedRatio", "stratifiedBatchCount", "stratifiedItemCount",
        "available", "selected",
    }
    long_route_fields = {
        "longRouteMinRatio", "longRouteEligibleItemCount",
        "longRouteRequiredItemCount", "longRouteSelectedItemCount",
        "longRouteOversampledItemCount",
    }
    if (
        not isinstance(value, dict)
        or not required_fields.issubset(value)
        or set(value) - required_fields - long_route_fields
        or bool(set(value) & long_route_fields) != (set(value) >= long_route_fields)
    ):
        return None
    ratio = _finite_number(value.get("stratifiedRatio"))
    batches = _safe_optional_int(value.get("stratifiedBatchCount"))
    items = _safe_optional_int(value.get("stratifiedItemCount"))
    available = _steps_to_goal_replay_population(value.get("available"))
    selected = _steps_to_goal_replay_population(value.get("selected"))
    if (
        ratio is None or not 0.0 <= ratio <= 1.0
        or batches is None or batches < 0 or items is None or items < 0
        or available is None or selected is None
    ):
        return None
    result = {
        "stratifiedRatio": ratio,
        "stratifiedBatchCount": batches,
        "stratifiedItemCount": items,
        "available": available,
        "selected": selected,
    }
    if set(value) >= long_route_fields:
        long_ratio = _finite_number(value.get("longRouteMinRatio"))
        counts = {
            key: _safe_optional_int(value.get(key))
            for key in long_route_fields - {"longRouteMinRatio"}
        }
        if (
            long_ratio is None or not 0.0 <= long_ratio <= 1.0
            or any(count is None or count < 0 for count in counts.values())
        ):
            return None
        result["longRouteMinRatio"] = long_ratio
        result.update(counts)
    return result


def _steps_to_goal_value_selection(value: Any) -> dict[str, Any] | None:
    """Validate compact Value-teacher filtering diagnostics."""
    required = set(_STEPS_TO_GOAL_VALUE_SELECTION_INT_FIELDS) | {
        "candidateStateBands", "selectedStateBands",
    }
    if (
        not isinstance(value, dict) or not required.issubset(value)
        or set(value) - required - _STEPS_TO_GOAL_VALUE_SELECTION_OPTIONAL_INT_FIELDS
    ):
        return None
    result = {}
    for key in _STEPS_TO_GOAL_VALUE_SELECTION_INT_FIELDS:
        count = _safe_optional_int(value.get(key))
        if count is None or count < 0:
            return None
        result[key] = count
    for key in _STEPS_TO_GOAL_VALUE_SELECTION_OPTIONAL_INT_FIELDS:
        if key not in value:
            continue
        count = _safe_optional_int(value.get(key))
        if count is None or count < 0:
            return None
        result[key] = count
    expected_bands = {"0-10", "11-30", "31+"}
    for key in ("candidateStateBands", "selectedStateBands"):
        bands = value.get(key)
        if not isinstance(bands, dict) or set(bands) != expected_bands:
            return None
        copied = {label: _safe_optional_int(bands[label]) for label in expected_bands}
        if any(count is None or count < 0 for count in copied.values()):
            return None
        result[key] = copied
    return result


def _steps_to_goal_regression(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    state_count = _safe_optional_int(value.get("stateCount"))
    trajectory_count = _safe_optional_int(value.get("trajectoryCount"))
    if state_count is None or trajectory_count is None:
        return None
    result = {"stateCount": state_count, "trajectoryCount": trajectory_count}
    for key in _STEPS_TO_GOAL_REGRESSION_NUMBER_FIELDS:
        number = _finite_number(value.get(key))
        if number is None:
            return None
        result[key] = number
    if "lossWeight" in value:
        weight = _finite_number(value.get("lossWeight"))
        if weight is None or weight < 0.0:
            return None
        result["lossWeight"] = weight
    if "origins" in value:
        source_rows = value.get("origins")
        if not isinstance(source_rows, dict):
            return None
        origins = {}
        for origin, metrics in source_rows.items():
            if not isinstance(origin, str) or not origin:
                return None
            copied = _steps_to_goal_regression(metrics)
            if copied is None or "lossWeight" in copied or "origins" in copied or "remainingStepBands" in copied or "valueSelection" in copied or "replay" in copied:
                return None
            origins[origin] = copied
        result["origins"] = origins
    if "remainingStepBands" in value:
        bands = value.get("remainingStepBands")
        expected_bands = {"0-10", "11-30", "31+"}
        if not isinstance(bands, dict) or set(bands) != expected_bands:
            return None
        copied_bands = {label: _safe_optional_int(bands.get(label)) for label in expected_bands}
        if any(count is None for count in copied_bands.values()):
            return None
        result["remainingStepBands"] = copied_bands
    if "valueSelection" in value:
        selection = _steps_to_goal_value_selection(value.get("valueSelection"))
        if selection is None:
            return None
        result["valueSelection"] = selection
    if "replay" in value:
        replay = _steps_to_goal_replay(value.get("replay"))
        if replay is None:
            return None
        result["replay"] = replay
    return result


def _validate_steps_to_goal_regression(value: Any) -> None:
    if value is None:
        return
    required = {"stateCount", "trajectoryCount", *_STEPS_TO_GOAL_REGRESSION_NUMBER_FIELDS}
    allowed = required | {"lossWeight", "origins", "remainingStepBands", "valueSelection", "replay"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - allowed or _steps_to_goal_regression(value) != value:
        raise ValueError("Invalid learning history stepsToGoalRegression")


def _gradient_metrics(value: Any) -> dict[str, Any] | None:
    """Copy optional per-learning raw-gradient aggregates."""
    if not isinstance(value, dict):
        return None
    update_count = _safe_optional_int(value.get("updateCount"))
    if update_count is None:
        return None
    result = {"updateCount": update_count}
    for key in _GRADIENT_METRIC_NUMBER_FIELDS:
        number = _finite_number(value.get(key))
        if number is None or number < 0.0:
            return None
        result[key] = number
    for key in ("policyGradientShareMean", "valueGradientShareMean"):
        if result[key] > 1.0:
            return None
    return result


def _validate_gradient_metrics(value: Any) -> None:
    if value is None:
        return
    expected = {"updateCount", *_GRADIENT_METRIC_NUMBER_FIELDS}
    if not isinstance(value, dict) or set(value) != expected or _gradient_metrics(value) != value:
        raise ValueError("Invalid learning history gradientMetrics")


def _training_sample(value: Any) -> dict[str, Any] | None:
    """Copy the optional sampled-replay mix recorded by Transformer learning."""
    if not isinstance(value, dict):
        return None
    result = {}
    if "batchMixMode" in value:
        if value["batchMixMode"] not in ("ordered", "mixed"):
            return None
        result["batchMixMode"] = value["batchMixMode"]
    for key in _TRAINING_SAMPLE_INT_FIELDS:
        number = _safe_optional_int(value.get(key))
        if number is None:
            return None
        result[key] = number
    result["longSelectedStepMean"] = _finite_number(value.get("longSelectedStepMean"))
    for key in _TRAINING_SAMPLE_OPTIONAL_INT_FIELDS:
        if key not in value:
            continue
        number = _safe_optional_int(value.get(key))
        if number is None or number < 0:
            return None
        result[key] = number
    # Replay ratios were added after the first sampled-replay histories.
    # Preserve those existing local records while including them going forward.
    for key in _TRAINING_SAMPLE_NUMBER_FIELDS:
        if key not in value:
            continue
        ratio = _finite_number(value.get(key))
        if ratio is None or not 0.0 <= ratio <= 1.0:
            return None
        result[key] = ratio
    return result


def _validate_training_sample(value: Any) -> None:
    if value is None:
        return
    required = set(_TRAINING_SAMPLE_INT_FIELDS) | {"longSelectedStepMean"}
    allowed = required | set(_TRAINING_SAMPLE_OPTIONAL_INT_FIELDS) | set(_TRAINING_SAMPLE_NUMBER_FIELDS) | {"batchMixMode"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - allowed:
        raise ValueError("Invalid learning history trainingSample")
    if _training_sample(value) != value:
        raise ValueError("Invalid learning history trainingSample")


def _fixed_validation_by_length(value: Any) -> dict[str, dict[str, Any]] | None:
    """Copy optional per-scramble-length probes without invalidating v1 data."""
    if not isinstance(value, dict) or len(value) == 0:
        return None
    result = {}
    for length, metrics in value.items():
        if not isinstance(length, str) or not length.isdecimal() or int(length) < 1:
            return None
        if not isinstance(metrics, dict):
            return None
        count = _safe_optional_int(metrics.get("fixtureCount"))
        state_count = _safe_optional_int(metrics.get("stateCount"))
        if count is None or state_count is None:
            return None
        copied = {"fixtureCount": count, "stateCount": state_count}
        for key in _FIXED_VALIDATION_NUMBERS:
            copied[key] = _finite_number(metrics.get(key))
        for key in _FIXED_VALIDATION_OPTIONAL_NUMBERS:
            if key in metrics:
                copied[key] = _finite_number(metrics[key])
        result[length] = copied
    return result


def _fixed_validation_bucket(value: Any) -> dict[str, Any] | None:
    """Copy an optional aggregate held-out bucket such as the 40+ probe."""
    if not isinstance(value, dict):
        return None
    count = _safe_optional_int(value.get("fixtureCount"))
    state_count = _safe_optional_int(value.get("stateCount"))
    if count is None or state_count is None:
        return None
    copied = {"fixtureCount": count, "stateCount": state_count}
    for key in _FIXED_VALIDATION_NUMBERS:
        copied[key] = _finite_number(value.get(key))
    for key in _FIXED_VALIDATION_OPTIONAL_NUMBERS:
        if key in value:
            copied[key] = _finite_number(value[key])
    return copied


def _fixed_validation_remaining_steps(value: Any) -> dict[str, dict[str, Any]] | None:
    """Copy optional calibrated-regression metrics by remaining-step band."""
    if not isinstance(value, dict) or len(value) == 0:
        return None
    result = {}
    for label,metrics in value.items():
        if label not in {"0-10", "11-30", "31+"} or not isinstance(metrics, dict):
            return None
        state_count = _safe_optional_int(metrics.get("stateCount"))
        if state_count is None:
            return None
        copied = {"stateCount": state_count}
        for key in _FIXED_VALIDATION_NUMBERS:
            copied[key] = _finite_number(metrics.get(key))
        for key in _FIXED_VALIDATION_OPTIONAL_NUMBERS:
            if key in metrics:
                copied[key] = _finite_number(metrics[key])
        result[label] = copied
    return result


def _fixed_validation_exact_distance(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    count = _safe_optional_int(value.get("stateCount"))
    if count is None or count < 2:
        return None
    result = {"stateCount": count}
    for key in ("goalValueError", "goalValueMae", "oneMoveValueBias", "oneMoveValueMae"):
        number = _finite_number(value.get(key))
        if number is None:
            return None
        result[key] = number
    return result


def _fixed_validation_by_effect(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    result = {}
    for label, metrics in value.items():
        if label not in {"0-7", "8-13", "14-19", "20+"} or not isinstance(metrics, dict):
            return None
        count = _safe_optional_int(metrics.get("stateCount"))
        value_count = _safe_optional_int(metrics.get("valueStepCount"))
        if count is None or value_count is None or value_count > count:
            return None
        result[label] = {"stateCount": count, "valueStepCount": value_count}
        for key in ("policyTargetRankMean", "policyTop1Accuracy", "valueDeltaMean", "valueImprovementRate"):
            result[label][key] = _finite_number(metrics.get(key))
    return result


def _fixed_validation_by_effect_choices(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    result = {}
    for label, metrics in value.items():
        if label not in {"0-7", "8-13", "14-19", "20+"} or not isinstance(metrics, dict):
            return None
        count = _safe_optional_int(metrics.get("sampleCount"))
        if count is None or count < 1 or count > 4:
            return None
        copied = {"sampleCount": count}
        for key in ("valueTargetRankMean", "valueTop1Accuracy", "valueCorrectVsBestOtherDeltaMean"):
            number = _finite_number(metrics.get(key))
            if number is None:
                return None
            copied[key] = number
        result[label] = copied
    return result


def _fixed_validation_transformer_branches(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    count = _safe_optional_int(value.get("sampleCount"))
    if count is None:
        return None
    result = {"sampleCount": count}
    for key in ("attentionRms", "linearRms", "linearToAttentionRms", "preactivationCosine"):
        result[key] = _finite_number(value.get(key))
    for label in ("full", "withoutAttention", "withoutLinear"):
        metrics = value.get(label)
        if not isinstance(metrics, dict):
            return None
        result[label] = {
            key: _finite_number(metrics.get(key))
            for key in ("policyTargetRankMean", "policyTop1Accuracy", "valueRankCorrelation")
        }
    return result


def _validate_fixed_validation(value: Any) -> None:
    if value is None:
        return
    expected = {
        "fixtureId", "fixtureLengths", "fixtureCount", "stateCount",
        *_FIXED_VALIDATION_NUMBERS,
    }
    if not isinstance(value, dict) or set(value) - expected - _FIXED_VALIDATION_OPTIONAL_NUMBERS - {"byLength", "long40Plus", "stepsToGoalByRemainingSteps", "stepsToGoalExactDistance", "byEffect", "byEffectChoices", "transformerBranches"}:
        raise ValueError("Invalid learning history fixedValidation")
    if not isinstance(value["fixtureId"], str) or not value["fixtureId"]:
        raise ValueError("Invalid learning history fixedValidation fixture")
    if not isinstance(value["fixtureLengths"], list) or any(
        isinstance(length, bool) or not isinstance(length, int) or length < 1
        for length in value["fixtureLengths"]
    ):
        raise ValueError("Invalid learning history fixedValidation fixture lengths")
    for key in ("fixtureCount", "stateCount"):
        if value[key] is None or isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise ValueError("Invalid learning history fixedValidation count")
    for key in _FIXED_VALIDATION_NUMBERS:
        _validate_optional_number(value[key])
    for key in _FIXED_VALIDATION_OPTIONAL_NUMBERS:
        if key in value:
            _validate_optional_number(value[key])
    if "byLength" in value:
        _validate_fixed_validation_by_length(value["byLength"])
    if "long40Plus" in value:
        normalized = _fixed_validation_bucket(value["long40Plus"])
        if normalized is None or normalized != value["long40Plus"]:
            raise ValueError("Invalid learning history fixedValidation 40+ bucket")
    if "stepsToGoalByRemainingSteps" in value:
        normalized = _fixed_validation_remaining_steps(value["stepsToGoalByRemainingSteps"])
        if normalized is None or normalized != value["stepsToGoalByRemainingSteps"]:
            raise ValueError("Invalid learning history fixedValidation remaining-step bands")
    if "stepsToGoalExactDistance" in value:
        normalized = _fixed_validation_exact_distance(value["stepsToGoalExactDistance"])
        if normalized is None or normalized != value["stepsToGoalExactDistance"]:
            raise ValueError("Invalid learning history fixedValidation exact-distance anchors")
    if "byEffect" in value:
        normalized = _fixed_validation_by_effect(value["byEffect"])
        if normalized is None or normalized != value["byEffect"]:
            raise ValueError("Invalid learning history fixedValidation effect bands")
    if "byEffectChoices" in value:
        normalized = _fixed_validation_by_effect_choices(value["byEffectChoices"])
        if normalized is None or normalized != value["byEffectChoices"]:
            raise ValueError("Invalid learning history fixedValidation Effect choices")
    if "transformerBranches" in value:
        normalized = _fixed_validation_transformer_branches(value["transformerBranches"])
        if normalized is None or normalized != value["transformerBranches"]:
            raise ValueError("Invalid learning history fixedValidation branches")


def _validate_fixed_validation_by_length(value: Any) -> None:
    """Validate v2's per-length metrics while retaining v1 compatibility."""
    normalized = _fixed_validation_by_length(value)
    if normalized is None or normalized != value:
        raise ValueError("Invalid learning history fixedValidation byLength")


def _validate_optional_number(value: Any) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError("Invalid learning history numeric value")


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _safe_optional_int(value: Any) -> int | None:
    """Convert a non-negative integral metric without accepting booleans."""
    if value is None or isinstance(value,bool):
        return None
    try:
        number = int(value)
    except (TypeError,ValueError):
        return None
    return number if number >= 0 else None

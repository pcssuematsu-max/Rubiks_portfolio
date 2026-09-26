"""Persist per-AI learning metrics for later inspection in the GUI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


LEARNING_HISTORY_SCHEMA_VERSION = 1
LEARNING_HISTORY_FILE_NAME = "ai-learning-history.json"
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
        payload = self._read()
        _validate_record(record)
        payload["records"].append(record)
        payload["updatedAt"] = record["timestamp"]
        self._write(payload)

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
            json.dump(payload, stream, ensure_ascii = False, indent = 2)
            stream.write("\n")
        temporary_path.replace(self.path)


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
        "valueSequenceCount", "fixedValidation",
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
    return result


_FIXED_VALIDATION_NUMBERS = frozenset({
    "policyTop1Accuracy", "policyTop3Accuracy", "policyTargetProbability",
    "policyCrossEntropy", "valueRankCorrelation", "valuePearsonCorrelation",
    "valuePredictionMean", "valuePredictionStd", "valueTargetMean",
    "valueTargetStd", "valueMae", "valueBce", "valuePathCrossEntropy",
})


def _validate_fixed_validation(value: Any) -> None:
    if value is None:
        return
    expected = {
        "fixtureId", "fixtureLengths", "fixtureCount", "stateCount",
        *_FIXED_VALIDATION_NUMBERS,
    }
    if not isinstance(value, dict) or set(value) != expected:
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

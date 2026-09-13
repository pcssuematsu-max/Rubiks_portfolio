"""Append-only experiment logs for comparing solve runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
from typing import Any


EXPERIMENT_LOG_SCHEMA_VERSION = 3
EXPERIMENT_LOG_FILE_NAME = "ai-experiments.jsonl"
EXPERIMENT_CSV_FILE_NAME = "ai-experiments.csv"
EXPERIMENT_SUMMARY_FILE_NAME = "ai-experiment-summary.json"
EXPERIMENT_SUMMARY_CSV_FILE_NAME = "ai-experiment-summary.csv"
SUMMARY_CSV_FIELDS = (
    "puzzle",
    "searchMode",
    "aiIndex",
    "aiSettings",
    "runCount",
    "classifiedRunCount",
    "directSearchSuccessCount",
    "directSearchSuccessRate",
    "fallbackCompletedCount",
    "fallbackCompletionRate",
    "failedCount",
    "legacyUnknownCount",
    "elapsedMinimum",
    "elapsedMedian",
    "elapsedMean",
    "directMovesMinimum",
    "directMovesMedian",
    "directMovesMean",
    "completedMovesMinimum",
    "completedMovesMedian",
    "completedMovesMean",
    "interestingDiscoveries",
)
CSV_FIELDS = (
    "schemaVersion", "timestamp", "puzzleType", "cubeSize", "puzzle",
    "searchMode", "aiIndex", "solveIndex", "stage", "succeeded",
    "searchSucceeded", "fallbackUsed", "outcome",
    "elapsedSeconds", "setupMoveCount", "moveCount", "score", "rootScore",
    "bestScore", "endReason", "stats", "setup", "moves", "aiSettings",
)


def default_experiment_log_path() -> Path:
    """Return the local path for append-only benchmark data."""
    configured_path = os.environ.get("TWISTY_EXPERIMENT_LOG_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parents[1] / "exports" / EXPERIMENT_LOG_FILE_NAME


@dataclass(frozen = True)
class ExperimentLogRecord:
    """One completed solve, with stable JSON-friendly benchmark fields."""

    timestamp: str
    puzzle_type: str
    cube_size: int
    puzzle: str
    search_mode: str
    ai_index: int
    solve_index: int
    stage: int
    succeeded: bool
    search_succeeded: bool
    fallback_used: bool
    outcome: str
    elapsed_seconds: float
    setup: tuple[str, ...]
    moves: tuple[str, ...]
    score: float | None
    root_score: float | None
    best_score: float | None
    end_reason: str | None
    stats: tuple[Any, ...]
    ai_settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the camelCase record written to both output formats."""
        return {
            "schemaVersion": EXPERIMENT_LOG_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "puzzleType": self.puzzle_type,
            "cubeSize": self.cube_size,
            "puzzle": self.puzzle,
            "searchMode": self.search_mode,
            "aiIndex": self.ai_index,
            "solveIndex": self.solve_index,
            "stage": self.stage,
            "succeeded": self.succeeded,
            "searchSucceeded": self.search_succeeded,
            "fallbackUsed": self.fallback_used,
            "outcome": self.outcome,
            "elapsedSeconds": self.elapsed_seconds,
            "setupMoveCount": len(self.setup),
            "moveCount": len(self.moves),
            # score is the default comparison value; the two originals retain
            # enough context for a later Web benchmark.
            "score": self.score,
            "rootScore": self.root_score,
            "bestScore": self.best_score,
            "endReason": self.end_reason,
            "stats": list(self.stats),
            "setup": list(self.setup),
            "moves": list(self.moves),
            "aiSettings": self.ai_settings,
        }


class ExperimentLogStore:
    """Write a run to JSONL and an analysis-friendly CSV companion."""

    def __init__(self, path: Path | None = None, csv_path: Path | None = None):
        self.path = Path(path) if path is not None else default_experiment_log_path()
        self.csv_path = (
            Path(csv_path)
            if csv_path is not None
            else self.path.with_name(EXPERIMENT_CSV_FILE_NAME)
        )

    def append(self, record: ExperimentLogRecord) -> None:
        """Append one completed solve without rewriting prior experiment data."""
        payload = record.to_dict()
        self.path.parent.mkdir(parents = True, exist_ok = True)
        with self.path.open("a", encoding = "utf-8", newline = "\n") as stream:
            json.dump(payload, stream, ensure_ascii = False, separators = (",", ":"))
            stream.write("\n")
        self._append_csv(payload)

    def _append_csv(self, payload: dict[str, Any]) -> None:
        self.csv_path.parent.mkdir(parents = True, exist_ok = True)
        self._ensure_csv_schema()
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = {field: self._csv_value(payload[field]) for field in CSV_FIELDS}
        with self.csv_path.open("a", encoding = "utf-8", newline = "") as stream:
            writer = csv.DictWriter(stream, fieldnames = CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_csv_schema(self) -> None:
        """Extend a legacy CSV header before appending a newer schema row."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        with self.csv_path.open(encoding = "utf-8", newline = "") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == list(CSV_FIELDS):
                return
            rows = list(reader)
        temporary_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temporary_path.open("w", encoding = "utf-8", newline = "") as stream:
            writer = csv.DictWriter(stream, fieldnames = CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        temporary_path.replace(self.csv_path)

    def summarize(self, source = "jsonl") -> dict[str, Any]:
        """Aggregate runs by puzzle and search mode for comparison or Web use."""
        records = self._read_records(source)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        ai_groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = {}
        for record in records:
            key = (record["puzzle"], record["searchMode"])
            groups.setdefault(key, []).append(record)
            ai_key = (
                record["puzzle"],
                record["searchMode"],
                record["aiIndex"],
                _settings_fingerprint(record["aiSettings"]),
            )
            ai_groups.setdefault(ai_key, []).append(record)
        return {
            "schemaVersion": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
            "source": source,
            "sourceSchemaVersions": sorted({record["schemaVersion"] for record in records}),
            "totalRuns": len(records),
            "groups": [
                _summarize_group(puzzle, search_mode, rows)
                for (puzzle, search_mode), rows in sorted(groups.items())
            ],
            "aiGroups": [
                _summarize_group(
                    puzzle,
                    search_mode,
                    rows,
                    ai_index = ai_index,
                    ai_settings = rows[0]["aiSettings"],
                )
                for (puzzle, search_mode, ai_index, _settings), rows in sorted(
                    ai_groups.items(),
                    key = _ai_group_sort_key,
                )
            ],
        }

    def export_summary(self, path: Path | None = None, source = "jsonl") -> Path:
        """Write compact JSON and CSV summaries ready for Web or spreadsheet use."""
        destination = (
            Path(path)
            if path is not None
            else self.path.with_name(EXPERIMENT_SUMMARY_FILE_NAME)
        )
        destination.parent.mkdir(parents = True, exist_ok = True)
        payload = self.summarize(source)
        temporary_path = destination.with_suffix(destination.suffix + ".tmp")
        with temporary_path.open("w", encoding = "utf-8") as stream:
            json.dump(payload, stream, ensure_ascii = False, indent = 2)
            stream.write("\n")
        temporary_path.replace(destination)
        self._write_summary_csv(
            payload,
            destination.with_name(EXPERIMENT_SUMMARY_CSV_FILE_NAME),
        )
        return destination

    def _read_records(self, source: str) -> list[dict[str, Any]]:
        if source == "csv":
            return self._read_csv_records()
        if source != "jsonl":
            raise ValueError("Experiment summary source must be 'jsonl' or 'csv'")
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding = "utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid experiment log at line {line_number}"
                    ) from error
                records.append(_normalize_experiment_record(payload, line_number))
        return records

    def _read_csv_records(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        records = []
        with self.csv_path.open(encoding = "utf-8", newline = "") as stream:
            for line_number, row in enumerate(csv.DictReader(stream), 2):
                records.append(_normalize_experiment_record(
                    _csv_row_to_payload(row),
                    line_number,
                ))
        return records

    @staticmethod
    def _write_summary_csv(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents = True, exist_ok = True)
        with path.open("w", encoding = "utf-8", newline = "") as stream:
            writer = csv.DictWriter(stream, fieldnames = SUMMARY_CSV_FIELDS)
            writer.writeheader()
            for group in payload["aiGroups"]:
                writer.writerow(_summary_csv_row(group))

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii = False, separators = (",", ":"))
        return value


def completed_experiment_record(
    *, puzzle_type: str, cube_size: int, search_mode: str, ai_index: int,
    solve_index: int, stage: int, succeeded: bool, search_succeeded: bool,
    fallback_used: bool, elapsed_seconds: float,
    setup, moves, root_score: float | None, best_score: float | None,
    end_reason: str | None, stats = (), ai_settings: dict[str, Any] | None = None,
) -> ExperimentLogRecord:
    """Create a normalized record at the instant a solve completes."""
    normalized_type = str(puzzle_type).strip().lower() or "unknown"
    normalized_size = int(cube_size)
    normalized_stats = () if stats is None else stats
    return ExperimentLogRecord(
        timestamp = datetime.now(timezone.utc).isoformat(timespec = "seconds"),
        puzzle_type = normalized_type,
        cube_size = normalized_size,
        puzzle = f"{normalized_type}-{normalized_size}x{normalized_size}",
        search_mode = str(search_mode).strip().lower() or "unknown",
        ai_index = int(ai_index),
        solve_index = int(solve_index),
        stage = int(stage),
        succeeded = bool(succeeded),
        search_succeeded = bool(search_succeeded),
        fallback_used = bool(fallback_used),
        outcome = experiment_outcome(
            search_succeeded = search_succeeded,
            succeeded = succeeded,
            fallback_used = fallback_used,
        ),
        elapsed_seconds = round(max(0.0, float(elapsed_seconds)), 6),
        setup = tuple(str(move).strip() for move in setup if str(move).strip()),
        moves = tuple(str(move).strip() for move in moves if str(move).strip()),
        score = best_score,
        root_score = root_score,
        best_score = best_score,
        end_reason = None if end_reason is None else str(end_reason),
        stats = tuple(_json_scalar(value) for value in normalized_stats),
        ai_settings = _normalize_ai_settings(ai_settings),
    )


def experiment_outcome(*, search_succeeded: bool, succeeded: bool, fallback_used: bool) -> str:
    """Classify direct search results separately from greedy fallback results."""
    if search_succeeded:
        return "search_success"
    if fallback_used:
        return "greedy_fallback_success" if succeeded else "greedy_fallback_failed"
    return "completed_without_search" if succeeded else "search_failed"


def _json_scalar(value: Any) -> Any:
    """Convert NumPy-like scalar values before passing them to json.dump."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_experiment_record(payload: Any, line_number: int) -> dict[str, Any]:
    """Normalize older rows while preserving their unavailable settings."""
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid experiment log at line {line_number}: expected object")
    required_fields = ("puzzle", "searchMode", "succeeded", "elapsedSeconds", "moveCount")
    if any(field not in payload for field in required_fields):
        raise ValueError(f"Invalid experiment log at line {line_number}: missing required field")
    schema_version = int(payload.get("schemaVersion", 1))
    search_succeeded = payload.get("searchSucceeded") if schema_version >= 2 else None
    fallback_used = payload.get("fallbackUsed") if schema_version >= 2 else None
    outcome = payload.get("outcome") if schema_version >= 2 else "legacy_unknown"
    if not outcome:
        outcome = "legacy_unknown"
    return {
        "schemaVersion": schema_version,
        "timestamp": str(payload.get("timestamp", "")),
        "puzzle": str(payload["puzzle"]),
        "searchMode": str(payload["searchMode"]),
        "aiIndex": _safe_int(payload.get("aiIndex")),
        "solveIndex": _safe_int(payload.get("solveIndex")),
        "succeeded": _safe_bool(payload["succeeded"]),
        "searchSucceeded": _safe_bool(search_succeeded),
        "fallbackUsed": _safe_bool(fallback_used),
        "outcome": str(outcome),
        "elapsedSeconds": _safe_float(payload["elapsedSeconds"]),
        "moveCount": _safe_int(payload["moveCount"]),
        "score": _safe_float(payload.get("score")),
        "setup": list(payload.get("setup", [])),
        "moves": list(payload.get("moves", [])),
        "aiSettings": _normalize_ai_settings(
            payload.get("aiSettings") if schema_version >= 3 else {}
        ),
    }


def _summarize_group(
    puzzle: str,
    search_mode: str,
    rows: list[dict[str, Any]],
    *,
    ai_index: int | None = None,
    ai_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build comparison metrics and the shortest direct-search discoveries."""
    classified_rows = [row for row in rows if row["outcome"] != "legacy_unknown"]
    direct_rows = [row for row in rows if row["outcome"] == "search_success"]
    fallback_rows = [
        row for row in rows if row["outcome"] == "greedy_fallback_success"
    ]
    failed_rows = [row for row in rows if not row["succeeded"]]
    elapsed = [row["elapsedSeconds"] for row in rows if row["elapsedSeconds"] is not None]
    direct_moves = [row["moveCount"] for row in direct_rows if row["moveCount"] is not None]
    completed_moves = [
        row["moveCount"] for row in rows
        if row["succeeded"] and row["moveCount"] is not None
    ]
    interesting_rows = sorted(
        direct_rows,
        key = lambda row: (
            row["moveCount"] if row["moveCount"] is not None else float("inf"),
            row["elapsedSeconds"] if row["elapsedSeconds"] is not None else float("inf"),
            row["timestamp"],
        ),
    )[:5]
    summary = {
        "puzzle": puzzle,
        "searchMode": search_mode,
        "runCount": len(rows),
        "classifiedRunCount": len(classified_rows),
        "directSearchSuccessCount": len(direct_rows),
        "directSearchSuccessRate": _rate(len(direct_rows), len(classified_rows)),
        "fallbackCompletedCount": len(fallback_rows),
        "fallbackCompletionRate": _rate(len(fallback_rows), len(classified_rows)),
        "failedCount": len(failed_rows),
        "legacyUnknownCount": len(rows) - len(classified_rows),
        "elapsedSeconds": _number_summary(elapsed),
        "directSolutionMoves": _number_summary(direct_moves),
        "completedSolutionMoves": _number_summary(completed_moves),
        "interestingDiscoveries": [
            {
                "timestamp": row["timestamp"],
                "aiIndex": row["aiIndex"],
                "solveIndex": row["solveIndex"],
                "moveCount": row["moveCount"],
                "elapsedSeconds": row["elapsedSeconds"],
                "score": row["score"],
                "setup": row["setup"],
                "moves": row["moves"],
            }
            for row in interesting_rows
        ],
    }
    if ai_index is not None:
        summary["aiIndex"] = ai_index
        summary["aiSettings"] = ai_settings or {}
    return summary


def _number_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": round(float(median(values)), 6),
        "mean": round(float(sum(values) / len(values)), 6),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        return None
    return bool(value)


def _csv_row_to_payload(row: dict[str, str]) -> dict[str, Any]:
    """Restore typed fields and JSON columns from the raw experiment CSV."""
    return {
        **row,
        "setup": _parse_csv_json_list(row.get("setup")),
        "moves": _parse_csv_json_list(row.get("moves")),
        "stats": _parse_csv_json_list(row.get("stats")),
        "aiSettings": _parse_csv_json_object(row.get("aiSettings")),
    }


def _parse_csv_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_csv_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _summary_csv_row(group: dict[str, Any]) -> dict[str, Any]:
    """Flatten one Search2/Search3 comparison row for spreadsheet tools."""
    elapsed = group["elapsedSeconds"]
    direct_moves = group["directSolutionMoves"]
    completed_moves = group["completedSolutionMoves"]
    return {
        "puzzle": group["puzzle"],
        "searchMode": group["searchMode"],
        "aiIndex": group.get("aiIndex"),
        "aiSettings": json.dumps(
            group.get("aiSettings", {}),
            ensure_ascii = False,
            separators = (",", ":"),
            sort_keys = True,
        ),
        "runCount": group["runCount"],
        "classifiedRunCount": group["classifiedRunCount"],
        "directSearchSuccessCount": group["directSearchSuccessCount"],
        "directSearchSuccessRate": group["directSearchSuccessRate"],
        "fallbackCompletedCount": group["fallbackCompletedCount"],
        "fallbackCompletionRate": group["fallbackCompletionRate"],
        "failedCount": group["failedCount"],
        "legacyUnknownCount": group["legacyUnknownCount"],
        "elapsedMinimum": elapsed["minimum"],
        "elapsedMedian": elapsed["median"],
        "elapsedMean": elapsed["mean"],
        "directMovesMinimum": direct_moves["minimum"],
        "directMovesMedian": direct_moves["median"],
        "directMovesMean": direct_moves["mean"],
        "completedMovesMinimum": completed_moves["minimum"],
        "completedMovesMedian": completed_moves["median"],
        "completedMovesMean": completed_moves["mean"],
        "interestingDiscoveries": json.dumps(
            group["interestingDiscoveries"],
            ensure_ascii = False,
            separators = (",", ":"),
        ),
    }


def rank_ai_metric_differences(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank outcome metrics by their relative spread across comparable AIs.

    This describes observed differences, not the cause.  The supplied groups
    should therefore all share one puzzle and one search mode.
    """
    specifications = (
        ("directSearchSuccessRate", "直接探索成功率", "rate", "higher"),
        ("directSolutionMoves", "直接探索の中央値手数", "median", "lower"),
        ("elapsedSeconds", "探索時間の中央値", "median", "lower"),
        ("completedSolutionMoves", "完了結果の中央値手数", "median", "lower"),
    )
    differences = []
    for key, label, field, preference in specifications:
        values = []
        for group in groups:
            value = group.get(key) if field == "rate" else group.get(key, {}).get(field)
            if value is not None:
                values.append((group, float(value)))
        if len(values) < 2:
            continue
        low_group, low_value = min(values, key = lambda item: item[1])
        high_group, high_value = max(values, key = lambda item: item[1])
        midpoint = (abs(low_value) + abs(high_value)) / 2
        differences.append({
            "key": key,
            "label": label,
            "field": field,
            "preference": preference,
            "lowGroup": low_group,
            "lowValue": low_value,
            "highGroup": high_group,
            "highValue": high_value,
            "relativeSpread": (high_value - low_value) / (midpoint or 1.0),
        })
    return sorted(differences, key = lambda item: item["relativeSpread"], reverse = True)


def _normalize_ai_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for key, item in value.items():
        if isinstance(item, dict):
            normalized[str(key)] = _normalize_ai_settings(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            normalized[str(key)] = _json_scalar(item)
    return normalized


def _settings_fingerprint(settings: dict[str, Any]) -> str:
    return json.dumps(settings, ensure_ascii = False, sort_keys = True, separators = (",", ":"))


def _ai_group_sort_key(item):
    (puzzle, search_mode, ai_index, settings), _rows = item
    return (puzzle, search_mode, -1 if ai_index is None else ai_index, settings)

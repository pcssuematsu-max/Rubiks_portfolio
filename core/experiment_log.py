"""Append-only experiment logs for comparing solve runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
from statistics import median
from typing import Any


EXPERIMENT_LOG_SCHEMA_VERSION = 5
SETUP_LENGTH_BANDS = (
    (0, 19, "0-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, None, "50+"),
)
SEARCH3_PLAYOUT_DEPTH_BANDS = (
    (0, 2, "0-2"),
    (3, 5, "3-5"),
    (6, 9, "6-9"),
    (10, 19, "10-19"),
    (20, None, "20+"),
)
# The physical effect after normalizing the cube's whole orientation.  These
# bands complement, rather than replace, the nominal scramble move count.
SETUP_EFFECT_BANDS = (
    (0, 7, "0-7"),
    (8, 13, "8-13"),
    (14, 19, "14-19"),
    (20, None, "20+"),
)
# The GUI consumes a recent solve window, not an unbounded archive.  Keep the
# setting snapshot on each row, while bounding the history per AI/search mode
# so tiny incidental setting changes cannot defeat compaction.
EXPERIMENT_RECORDS_PER_AI_MODE = 200
EXPERIMENT_COMPACT_THRESHOLD_BYTES = 32 * 1024 * 1024
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
    "bestScore", "endReason", "stats", "directSearch", "setup", "moves", "setupEffect", "aiSettings",
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
    direct_search: dict[str, Any]
    setup_effect: dict[str, Any] | None
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
            # Preserve the direct-search result before a greedy fallback
            # replaces the session's display/result state.
            "directSearch": self.direct_search,
            # The raw routes are saved in the discovery feed when useful.
            # Benchmark logs retain only their scalar effect summaries.
            "setup": [],
            "moves": [],
            "setupEffect": self.setup_effect,
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
        """Append one compact solve record and bound retained benchmark history."""
        payload = record.to_dict()
        if self.path.exists() and self.path.stat().st_size >= EXPERIMENT_COMPACT_THRESHOLD_BYTES:
            self.compact()
        self.path.parent.mkdir(parents = True, exist_ok = True)
        with self.path.open("a", encoding = "utf-8", newline = "\n") as stream:
            json.dump(payload, stream, ensure_ascii = False, separators = (",", ":"))
            stream.write("\n")
        self._append_csv(payload)

    def compact(self) -> tuple[int, int]:
        """Keep a compact recent window per AI/search mode and rebuild both logs.

        Full route sequences and per-attempt arrays are intentionally omitted:
        the experiment log is for outcome analysis, while reproducible notable
        routes belong in the discovery archive.
        """
        if not self.path.exists():
            return (0, 0)
        retained: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
        total = 0
        with self.path.open(encoding = "utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = _normalize_experiment_record(payload, line_number)
                effect = record.get("setupEffect")
                if effect is None:
                    effect = setup_effect_metrics(
                        record["puzzle"], payload.get("cubeSize"), record.get("setup", ()),
                    )
                compacted = _compact_experiment_payload(record, effect)
                key = (record["puzzle"], record["searchMode"], record["aiIndex"])
                rows = retained.setdefault(key, [])
                rows.append(compacted)
                if len(rows) > EXPERIMENT_RECORDS_PER_AI_MODE:
                    del rows[0]
                total += 1
        records = [record for rows in retained.values() for record in rows]
        records.sort(key = lambda record: record["timestamp"])
        self._write_compacted_records(records)
        return (total, len(records))

    def _write_compacted_records(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents = True, exist_ok = True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding = "utf-8", newline = "\n") as stream:
            for payload in records:
                json.dump(payload, stream, ensure_ascii = False, separators = (",", ":"))
                stream.write("\n")
        temporary_path.replace(self.path)
        with self.csv_path.open("w", encoding = "utf-8", newline = "") as stream:
            writer = csv.DictWriter(stream, fieldnames = CSV_FIELDS)
            writer.writeheader()
            for payload in records:
                writer.writerow({field: self._csv_value(payload.get(field)) for field in CSV_FIELDS})

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

    def recent_ai_results(
        self,
        ai_index: int,
        *,
        puzzle: str | None = None,
        search_mode: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return a compact, current-window benchmark for one AI.

        The learning-history dialog uses this as its outcome metric.  It is
        deliberately separate from loss because direct-search success is the
        user-visible check that learned estimates actually improve search.
        """
        rows = [
            row for row in self._read_records("jsonl")
            if row["aiIndex"] == int(ai_index)
            and (puzzle is None or row["puzzle"] == puzzle)
            and (search_mode is None or row["searchMode"] == search_mode)
        ]
        window = rows[-max(1,int(limit)):]
        classified = [row for row in window if row["outcome"] != "legacy_unknown"]
        direct = [row for row in classified if row["outcome"] == "search_success"]
        fallback = [
            row for row in classified
            if row["outcome"] in ("greedy_fallback_success", "greedy_fallback_failed")
        ]
        probed_failures = [
            (row.get("directSearch") or {}).get("finalAttempt") or {}
            for row in classified if row["outcome"] != "search_success"
        ]
        probed_failures = [
            attempt for attempt in probed_failures
            if isinstance(attempt.get("terminalImprovingSuccessorCount"), int)
        ]
        improving_failures = sum(
            attempt["terminalImprovingSuccessorCount"] > 0
            for attempt in probed_failures
        )
        return {
            "runCount": len(window),
            "classifiedRunCount": len(classified),
            "directSearchSuccessCount": len(direct),
            "directSearchSuccessRate": _rate(len(direct),len(classified)),
            "fallbackCount": len(fallback),
            "terminalProbedFailureCount": len(probed_failures),
            "terminalImprovingFailureCount": improving_failures,
            "terminalImprovingFailureRate": _rate(improving_failures, len(probed_failures)),
        }

    def recent_ai_search3_depth_bands(
        self,
        ai_index: int,
        *,
        puzzle: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return recent Search3 outcomes binned by observed playout depth."""
        rows = [
            row for row in self._read_records("jsonl")
            if row["aiIndex"] == int(ai_index)
            and row["searchMode"] in ("search3", "transformer")
            and (puzzle is None or row["puzzle"] == puzzle)
        ]
        return _search3_depth_bands(rows[-max(1, int(limit)):])

    def recent_ai_setup_effect_bands(
        self,
        ai_index: int,
        *,
        puzzle: str | None = None,
        search_mode: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return direct-search outcomes by orientation-normalized effect band."""
        rows = [
            row for row in self._read_records("jsonl")
            if row["aiIndex"] == int(ai_index)
            and (puzzle is None or row["puzzle"] == puzzle)
            and (search_mode is None or row["searchMode"] == search_mode)
        ]
        return _setup_effect_bands(rows[-max(1, int(limit)):])

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
    end_reason: str | None, stats = (), direct_search: dict[str, Any] | None = None,
    ai_settings: dict[str, Any] | None = None,
) -> ExperimentLogRecord:
    """Create a normalized record at the instant a solve completes."""
    normalized_type = str(puzzle_type).strip().lower() or "unknown"
    normalized_size = int(cube_size)
    normalized_stats = () if stats is None else stats
    normalized_setup = tuple(str(move).strip() for move in setup if str(move).strip())
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
        setup = normalized_setup,
        moves = tuple(str(move).strip() for move in moves if str(move).strip()),
        score = best_score,
        root_score = root_score,
        best_score = best_score,
        end_reason = None if end_reason is None else str(end_reason),
        stats = tuple(_json_scalar(value) for value in normalized_stats),
        direct_search = _normalize_json_object(direct_search),
        setup_effect = setup_effect_metrics(
            f"{normalized_type}-{normalized_size}x{normalized_size}",
            normalized_size,
            normalized_setup,
        ),
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


def _normalize_json_object(value: Any) -> dict[str, Any]:
    """Return a JSON-safe object while retaining nested search diagnostics."""
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _normalize_json_value(item)
        for key,item in value.items()
    }


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _normalize_json_object(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return _json_scalar(value)


def setup_effect_metrics(puzzle: str, cube_size: Any, setup) -> dict[str, Any] | None:
    """Return compact physical difficulty metrics for a 3x3 setup.

    A whole-cube rotation changes notation but not puzzle difficulty.  We
    append the shortest orientation-restoring rotation before analysing the
    net permutation, then count only visible piece effects.
    """
    if str(puzzle) not in {"cube-3x3", "rubiks-3x3"}:
        return None
    try:
        if int(cube_size) != 3:
            return None
    except (TypeError, ValueError):
        return None
    try:
        moves = tuple(str(move).strip() for move in setup if str(move).strip())
    except TypeError:
        return None
    return _cube_setup_effect_metrics(moves)


@lru_cache(maxsize = 8192)
def _cube_setup_effect_metrics(moves: tuple[str, ...]) -> dict[str, Any] | None:
    """Analyse one normalized 3x3 route once, even when many AIs share it."""
    try:
        cube, analyzer, display_to_key, orientation_suffixes = _rubiks_effect_context()
        moves = tuple(display_to_key[move] for move in moves)
        suffix = _orientation_restoring_suffix(cube, moves, orientation_suffixes)
        effect = analyzer.analyze(moves + suffix)
        visible = tuple(
            component for component in effect.components
            if not component.is_internal_center_permutation()
        )
        simplified = cube.simplify(moves)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return {
        "version": 1,
        "movedPieceCount": sum(component.moved_count for component in visible),
        "orientationCount": sum(component.orientation_count for component in visible),
        "componentCount": len(visible),
        "reducedMoveCount": len(simplified),
        "orientationNormalizationMoveCount": len(suffix),
    }


@lru_cache(maxsize = 1)
def _rubiks_effect_context():
    """Build one immutable 3x3 effect-analysis context for experiment logs."""
    from cube.rubiks_cube import Rubiks_3
    from core.myperm_effects import MypermEffectAnalyzer

    cube = Rubiks_3()
    display_to_key = {str(key).strip(): key for key in cube.move_keys}
    orientation_keys = tuple(
        key for key in cube.move_keys if str(key).strip().lstrip("1234567890")[:1] in {"x", "y", "z"}
    )
    center_positions = tuple(int(piece[0]) for piece in cube.center_index)
    identity = tuple(range(len(cube.state_0)))
    queue = [()]
    seen = {tuple(identity[position] for position in center_positions)}
    suffixes = []
    while queue:
        sequence = queue.pop(0)
        suffixes.append(sequence)
        for move in orientation_keys:
            candidate = sequence + (move,)
            state = _move_permutation(cube, candidate)
            signature = tuple(state[position] for position in center_positions)
            if signature not in seen:
                seen.add(signature)
                queue.append(candidate)
    return cube, MypermEffectAnalyzer(cube), display_to_key, tuple(suffixes)


def _move_permutation(cube, moves):
    state = list(range(len(cube.state_0)))
    for move in moves:
        state = [state[int(index)] for index in cube.move[move]]
    return state


def _orientation_restoring_suffix(cube, moves, suffixes):
    center_positions = tuple(int(piece[0]) for piece in cube.center_index)
    for suffix in suffixes:
        state = _move_permutation(cube, moves + suffix)
        if all(state[position] == position for position in center_positions):
            return suffix
    return ()


def _setup_effect(value: Any) -> dict[str, Any] | None:
    required = {
        "version", "movedPieceCount", "orientationCount", "componentCount",
        "reducedMoveCount", "orientationNormalizationMoveCount",
    }
    if not isinstance(value, dict) or set(value) != required:
        return None
    normalized = {key: _safe_int(value.get(key)) for key in required}
    if any(number is None or number < 0 for number in normalized.values()):
        return None
    if normalized["version"] != 1:
        return None
    return normalized


def _compact_direct_search(value: Any) -> dict[str, Any]:
    """Drop repeated attempt arrays while keeping diagnostics used by the GUI."""
    source = _normalize_json_object(value)
    result = {}
    for key in ("attemptCount", "terminalEndReason", "endReasonCounts", "totals", "finalAttempt"):
        if key in source:
            result[key] = source[key]
    final_attempt = result.get("finalAttempt")
    if isinstance(final_attempt, dict):
        final_attempt.pop("budgetStages", None)
    return result


def _compact_experiment_payload(record: dict[str, Any], effect: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a normalized row into the bounded on-disk experiment schema."""
    return {
        "schemaVersion": EXPERIMENT_LOG_SCHEMA_VERSION,
        "timestamp": record["timestamp"],
        "puzzle": record["puzzle"],
        "searchMode": record["searchMode"],
        "aiIndex": record["aiIndex"],
        "solveIndex": record["solveIndex"],
        "succeeded": record["succeeded"],
        "searchSucceeded": record["searchSucceeded"],
        "fallbackUsed": record["fallbackUsed"],
        "outcome": record["outcome"],
        "elapsedSeconds": record["elapsedSeconds"],
        "setupMoveCount": record["setupMoveCount"],
        "moveCount": record["moveCount"],
        "score": record["score"],
        "directSearch": _compact_direct_search(record["directSearch"]),
        "setup": [],
        "moves": [],
        "setupEffect": effect,
        "aiSettings": record["aiSettings"],
    }


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
        "setupMoveCount": _safe_int(payload.get("setupMoveCount", len(payload.get("setup", [])))),
        "moveCount": _safe_int(payload["moveCount"]),
        "score": _safe_float(payload.get("score")),
        "directSearch": _normalize_json_object(
            payload.get("directSearch") if schema_version >= 4 else {}
        ),
        "setup": list(payload.get("setup", [])),
        "moves": list(payload.get("moves", [])),
        "setupEffect": _setup_effect(payload.get("setupEffect")),
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
        # Keep direct-search success stratified by the initial setup length.
        # A changed scramble mix must not look like a model improvement.
        "setupLengthBands": _setup_length_bands(rows),
        "setupEffectBands": _setup_effect_bands(rows),
        "search3DepthBands": _search3_depth_bands(rows),
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


def _setup_length_bands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize direct-search quality without mixing setup difficulty."""
    result = []
    for minimum, maximum, label in SETUP_LENGTH_BANDS:
        band_rows = [
            row for row in rows
            if row.get("setupMoveCount") is not None
            and int(row["setupMoveCount"]) >= minimum
            and (maximum is None or int(row["setupMoveCount"]) <= maximum)
        ]
        classified = [row for row in band_rows if row["outcome"] != "legacy_unknown"]
        direct = [row for row in band_rows if row["outcome"] == "search_success"]
        fallback = [
            row for row in band_rows
            if row["outcome"] in ("greedy_fallback_success", "greedy_fallback_failed")
        ]
        result.append({
            "label": label,
            "minimum": minimum,
            "maximum": maximum,
            "runCount": len(band_rows),
            "classifiedRunCount": len(classified),
            "directSearchSuccessCount": len(direct),
            "directSearchSuccessRate": _rate(len(direct), len(classified)),
            "fallbackCount": len(fallback),
        })
    return result


def _setup_effect_bands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize direct success by net visible-piece effect, not raw moves."""
    result = []
    for minimum, maximum, label in SETUP_EFFECT_BANDS:
        band_rows = [
            row for row in rows
            if isinstance(row.get("setupEffect"), dict)
            and isinstance(row["setupEffect"].get("movedPieceCount"), int)
            and row["setupEffect"]["movedPieceCount"] >= minimum
            and (maximum is None or row["setupEffect"]["movedPieceCount"] <= maximum)
        ]
        classified = [row for row in band_rows if row["outcome"] != "legacy_unknown"]
        direct = [row for row in band_rows if row["outcome"] == "search_success"]
        orientations = [
            row["setupEffect"].get("orientationCount") for row in band_rows
            if isinstance(row["setupEffect"].get("orientationCount"), int)
        ]
        reduced = [
            row["setupEffect"].get("reducedMoveCount") for row in band_rows
            if isinstance(row["setupEffect"].get("reducedMoveCount"), int)
        ]
        attempts = [
            (row, (row.get("directSearch") or {}).get("finalAttempt") or {})
            for row in band_rows if row.get("searchMode") in ("search3", "transformer")
        ]
        direct_visits = [
            _safe_float(attempt.get("returnedFirstMoveVisitShare"))
            for row, attempt in attempts if row.get("searchSucceeded")
        ]
        failed_visits = [
            _safe_float(attempt.get("returnedFirstMoveVisitShare"))
            for row, attempt in attempts if not row.get("searchSucceeded")
        ]
        fallback_visits = [
            _safe_float(attempt.get("fallbackFirstMoveVisitShare"))
            for _, attempt in attempts
        ]
        direct_deltas = [
            _safe_float(attempt.get("returnedFirstMoveValueDelta"))
            for row, attempt in attempts if row.get("searchSucceeded")
        ]
        failed_deltas = [
            _safe_float(attempt.get("returnedFirstMoveValueDelta"))
            for row, attempt in attempts if not row.get("searchSucceeded")
        ]
        probed_failures = [
            (row.get("directSearch") or {}).get("finalAttempt") or {}
            for row in band_rows if row.get("outcome") not in ("search_success", "legacy_unknown")
        ]
        probed_failures = [
            attempt for attempt in probed_failures
            if isinstance(attempt.get("terminalImprovingSuccessorCount"), int)
        ]
        improving_failures = sum(
            attempt["terminalImprovingSuccessorCount"] > 0
            for attempt in probed_failures
        )
        result.append({
            "label": label,
            "minimum": minimum,
            "maximum": maximum,
            "runCount": len(band_rows),
            "classifiedRunCount": len(classified),
            "directSearchSuccessCount": len(direct),
            "directSearchSuccessRate": _rate(len(direct), len(classified)),
            "terminalProbedFailureCount": len(probed_failures),
            "terminalImprovingFailureCount": improving_failures,
            "terminalImprovingFailureRate": _rate(improving_failures, len(probed_failures)),
            "averageOrientationCount": _mean_number(orientations),
            "averageReducedMoveCount": _mean_number(reduced),
            "directFirstMoveVisitShare": _mean_number(direct_visits),
            "directFirstMoveVisitCount": sum(value is not None for value in direct_visits),
            "failedFirstMoveVisitShare": _mean_number(failed_visits),
            "failedFirstMoveVisitCount": sum(value is not None for value in failed_visits),
            "fallbackFirstMoveVisitShare": _mean_number(fallback_visits),
            "fallbackFirstMoveVisitCount": sum(value is not None for value in fallback_visits),
            "directFirstMoveValueDelta": _mean_number(direct_deltas),
            "directFirstMoveValueIncreaseRate": _mean_number([
                float(value > 0.0) for value in direct_deltas if value is not None
            ]),
            "failedFirstMoveValueDelta": _mean_number(failed_deltas),
            "failedFirstMoveValueIncreaseRate": _mean_number([
                float(value > 0.0) for value in failed_deltas if value is not None
            ]),
            "fallbackFirstMovePolicyRank": _mean_number([
                _safe_float(attempt.get("fallbackFirstMovePolicyRank"))
                for _, attempt in attempts
            ]),
        })
    return result


def _search3_depth_bands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Search3 attempts by the reached playout-depth band."""
    diagnostic_rows = []
    for row in rows:
        if row.get("searchMode") not in ("search3", "transformer"):
            continue
        attempt = row.get("directSearch", {}).get("finalAttempt", {})
        depth = _safe_float(attempt.get("playoutDepthMedian"))
        if depth is not None:
            diagnostic_rows.append((row, attempt, depth))

    result = []
    for minimum, maximum, label in SEARCH3_PLAYOUT_DEPTH_BANDS:
        band_rows = [
            item for item in diagnostic_rows
            if item[2] >= minimum and (maximum is None or item[2] <= maximum)
        ]
        classified = [item for item in band_rows if item[0]["outcome"] != "legacy_unknown"]
        direct = [item for item in band_rows if item[0]["outcome"] == "search_success"]
        node_counts = [_safe_float(item[1].get("treeNodeCount")) for item in band_rows]
        root_widths = [_safe_float(item[1].get("visitedRootChildCount")) for item in band_rows]
        coefficients = [_safe_float(item[1].get("selectionCMean")) for item in band_rows]
        mean_depths = [_safe_float(item[1].get("playoutDepthMean")) for item in band_rows]
        widening_limits = [_safe_float(item[1].get("wideningActionLimitMean")) for item in band_rows]
        result.append({
            "label": label,
            "minimum": minimum,
            "maximum": maximum,
            "runCount": len(band_rows),
            "classifiedRunCount": len(classified),
            "directSearchSuccessCount": len(direct),
            "directSearchSuccessRate": _rate(len(direct), len(classified)),
            "averageNodeCount": _mean_number(node_counts),
            "averageVisitedRootChildCount": _mean_number(root_widths),
            "averageSelectionC": _mean_number(coefficients),
            "averagePlayoutDepth": _mean_number(mean_depths),
            "averageWideningActionLimit": _mean_number(widening_limits),
        })
    return result


def _mean_number(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return round(float(sum(numeric) / len(numeric)), 6)


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
        elif isinstance(item, (list, tuple)):
            normalized[str(key)] = [_json_scalar(child) for child in item]
        elif isinstance(item, (str, int, float, bool)) or item is None:
            normalized[str(key)] = _json_scalar(item)
    return normalized


def _settings_fingerprint(settings: dict[str, Any]) -> str:
    return json.dumps(settings, ensure_ascii = False, sort_keys = True, separators = (",", ":"))


def _ai_group_sort_key(item):
    (puzzle, search_mode, ai_index, settings), _rows = item
    return (puzzle, search_mode, -1 if ai_index is None else ai_index, settings)

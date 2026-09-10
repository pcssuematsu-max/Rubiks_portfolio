"""Append-only experiment logs for comparing solve runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


EXPERIMENT_LOG_SCHEMA_VERSION = 1
EXPERIMENT_LOG_FILE_NAME = "ai-experiments.jsonl"
EXPERIMENT_CSV_FILE_NAME = "ai-experiments.csv"
CSV_FIELDS = (
    "schemaVersion", "timestamp", "puzzleType", "cubeSize", "puzzle",
    "searchMode", "aiIndex", "solveIndex", "stage", "succeeded",
    "elapsedSeconds", "setupMoveCount", "moveCount", "score", "rootScore",
    "bestScore", "endReason", "stats", "setup", "moves",
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
    elapsed_seconds: float
    setup: tuple[str, ...]
    moves: tuple[str, ...]
    score: float | None
    root_score: float | None
    best_score: float | None
    end_reason: str | None
    stats: tuple[Any, ...]

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
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = {field: self._csv_value(payload[field]) for field in CSV_FIELDS}
        with self.csv_path.open("a", encoding = "utf-8", newline = "") as stream:
            writer = csv.DictWriter(stream, fieldnames = CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii = False, separators = (",", ":"))
        return value


def completed_experiment_record(
    *, puzzle_type: str, cube_size: int, search_mode: str, ai_index: int,
    solve_index: int, stage: int, succeeded: bool, elapsed_seconds: float,
    setup, moves, root_score: float | None, best_score: float | None,
    end_reason: str | None, stats = (),
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
        elapsed_seconds = round(max(0.0, float(elapsed_seconds)), 6),
        setup = tuple(str(move).strip() for move in setup if str(move).strip()),
        moves = tuple(str(move).strip() for move in moves if str(move).strip()),
        score = best_score,
        root_score = root_score,
        best_score = best_score,
        end_reason = None if end_reason is None else str(end_reason),
        stats = tuple(_json_scalar(value) for value in normalized_stats),
    )


def _json_scalar(value: Any) -> Any:
    """Convert NumPy-like scalar values before passing them to json.dump."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

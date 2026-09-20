"""Append-only records for AI lifecycle events that change comparability."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


AI_LIFECYCLE_LOG_SCHEMA_VERSION = 1
AI_LIFECYCLE_LOG_FILE_NAME = "ai-lifecycle-events.jsonl"


def default_ai_lifecycle_log_path() -> Path:
    """Return the local lifecycle log path, with an experiment override."""
    configured_path = os.environ.get("TWISTY_AI_LIFECYCLE_LOG_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parents[1] / "exports" / AI_LIFECYCLE_LOG_FILE_NAME


class AiLifecycleLogStore:
    """Persist starts, parameter loads, and model mutations independently of solves."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_ai_lifecycle_log_path()

    def append(
        self,
        event_type: str,
        session_id: str,
        *,
        ai_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one JSON-safe lifecycle event and return its stored record."""
        record = {
            "schemaVersion": AI_LIFECYCLE_LOG_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
            "eventType": str(event_type),
            "sessionId": str(session_id),
            "aiIndex": None if ai_index is None else int(ai_index),
            "details": {} if details is None else dict(details),
        }
        _validate_record(record)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        with self.path.open("a", encoding = "utf-8", newline = "\n") as stream:
            json.dump(record, stream, ensure_ascii = False, separators = (",", ":"))
            stream.write("\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        """Read valid lifecycle records in their append order."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding = "utf-8") as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid AI lifecycle log at line {line_number}: {self.path}"
                    ) from error
                _validate_record(record)
                records.append(record)
        return records


def _validate_record(record: Any) -> None:
    required = {
        "schemaVersion", "timestamp", "eventType", "sessionId", "aiIndex", "details",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Invalid AI lifecycle log record")
    if record["schemaVersion"] != AI_LIFECYCLE_LOG_SCHEMA_VERSION:
        raise ValueError("Unsupported AI lifecycle log schemaVersion")
    for field in ("timestamp", "eventType", "sessionId"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"Invalid AI lifecycle log {field}")
    ai_index = record["aiIndex"]
    if ai_index is not None and (isinstance(ai_index,bool) or not isinstance(ai_index,int) or ai_index < 0):
        raise ValueError("Invalid AI lifecycle log aiIndex")
    if not isinstance(record["details"], dict):
        raise ValueError("Invalid AI lifecycle log details")
    try:
        json.dumps(record["details"], ensure_ascii = False, allow_nan = False)
    except (TypeError, ValueError) as error:
        raise ValueError("AI lifecycle log details must be JSON-safe") from error

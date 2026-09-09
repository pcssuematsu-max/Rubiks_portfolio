"""Persist successful AI solves for the static web showcase."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from core.myperm_points import point_representative_transform


DISCOVERIES_FILE_NAME = "ai-discoveries.json"
DISCOVERY_SCHEMA_VERSION = 1
_PAYLOAD_FIELDS = frozenset({"schemaVersion", "updatedAt", "discoveries"})
_RECORD_FIELDS = frozenset(
    {"id", "puzzle", "setup", "moves", "moveCount", "foundAt", "updatedAt"}
)


def default_discoveries_path() -> Path:
    """Return the web project's discovery feed when it is available locally."""
    configured_path = os.environ.get("TWISTY_WEB_DISCOVERIES_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    project_root = Path(__file__).resolve().parents[1]
    for parent in project_root.parents:
        for name in ("ルービックキューブマニュアル", "ルービックキューブマニュアル"):
            candidate = parent / name / "assets" / "data" / DISCOVERIES_FILE_NAME
            if candidate.parent.parent.parent.exists():
                return candidate
    return project_root / "exports" / DISCOVERIES_FILE_NAME


def _clean_moves(moves) -> list[str]:
    return [str(move).strip() for move in moves if str(move).strip()]


def _record_id(puzzle: str, setup: list[str]) -> str:
    payload = "\0".join((puzzle, *setup)).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def point_canonical_discovery_sequences(cube, setup, moves) -> tuple[tuple, tuple]:
    """Return setup and solution in the same highest-point orientation.

    ``point_representative_transform`` chooses the most readable orientation
    from the solution.  Applying that same symmetry to the setup preserves the
    relationship that the stored solution solves the stored setup.
    """
    source_setup = tuple(setup)
    source_moves = tuple(moves)
    try:
        representative = point_representative_transform(cube, source_moves)
        canonical_setup = tuple(cube.transform(source_setup, representative.transform_index))
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        return source_setup, source_moves
    return canonical_setup, tuple(representative.moves)


def _new_payload() -> dict:
    return {
        "schemaVersion": DISCOVERY_SCHEMA_VERSION,
        "updatedAt": None,
        "discoveries": [],
    }


def _validation_error(path: str, message: str) -> ValueError:
    return ValueError(f"Invalid discovery file: {path}: {message}")


def _validate_timestamp(value, field: str, path: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not value:
        raise _validation_error(path, f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _validation_error(path, f"{field} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise _validation_error(path, f"{field} must include a timezone")


def _validate_move_list(value, field: str, path: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise _validation_error(path, f"{field} must be a{' non-empty' if not allow_empty else ''} list")
    if any(not isinstance(move, str) or not move.strip() for move in value):
        raise _validation_error(path, f"{field} must contain non-empty strings")
    return value


def _validate_record(record, index: int, path: str) -> None:
    location = f"discoveries[{index}]"
    if not isinstance(record, dict):
        raise _validation_error(path, f"{location} must be an object")
    if set(record) != _RECORD_FIELDS:
        raise _validation_error(path, f"{location} must contain exactly {sorted(_RECORD_FIELDS)}")

    puzzle = record["puzzle"]
    if not isinstance(puzzle, str) or not puzzle.strip():
        raise _validation_error(path, f"{location}.puzzle must be a non-empty string")
    setup = _validate_move_list(record["setup"], f"{location}.setup", path, allow_empty=True)
    moves = _validate_move_list(record["moves"], f"{location}.moves", path, allow_empty=False)

    move_count = record["moveCount"]
    if isinstance(move_count, bool) or not isinstance(move_count, int):
        raise _validation_error(path, f"{location}.moveCount must be an integer")
    if move_count != len(moves):
        raise _validation_error(path, f"{location}.moveCount must equal len(moves)")

    record_id = record["id"]
    expected_id = _record_id(puzzle, setup)
    if not isinstance(record_id, str) or record_id != expected_id:
        raise _validation_error(path, f"{location}.id does not match puzzle and setup")
    _validate_timestamp(record["foundAt"], f"{location}.foundAt", path)
    _validate_timestamp(record["updatedAt"], f"{location}.updatedAt", path)


def _validate_payload(payload, path: str) -> None:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
        raise _validation_error(path, f"root must contain exactly {sorted(_PAYLOAD_FIELDS)}")

    schema_version = payload["schemaVersion"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DISCOVERY_SCHEMA_VERSION
    ):
        raise _validation_error(path, f"schemaVersion must be {DISCOVERY_SCHEMA_VERSION}")
    _validate_timestamp(payload["updatedAt"], "updatedAt", path, allow_none=True)

    discoveries = payload["discoveries"]
    if not isinstance(discoveries, list):
        raise _validation_error(path, "discoveries must be a list")
    seen_ids = set()
    for index, record in enumerate(discoveries):
        _validate_record(record, index, path)
        if record["id"] in seen_ids:
            raise _validation_error(path, f"discoveries[{index}].id is duplicated")
        seen_ids.add(record["id"])


class AiDiscoveryStore:
    """Store the shortest successful solve for each puzzle and start position."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_discoveries_path()

    def save(self, puzzle: str, setup, moves) -> str:
        """Save a discovery and return ``added``, ``shorter``, or ``unchanged``."""
        normalized_puzzle = str(puzzle).strip()
        clean_setup = _clean_moves(setup)
        clean_moves = _clean_moves(moves)
        if not normalized_puzzle:
            raise ValueError("puzzle is required")
        if not clean_moves:
            raise ValueError("moves is required")

        payload = self._read()
        discoveries = payload["discoveries"]
        record_id = _record_id(normalized_puzzle, clean_setup)
        current = next((item for item in discoveries if item.get("id") == record_id), None)
        if current is not None and len(current.get("moves", ())) <= len(clean_moves):
            return "unchanged"

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "id": record_id,
            "puzzle": normalized_puzzle,
            "setup": clean_setup,
            "moves": clean_moves,
            "moveCount": len(clean_moves),
            "foundAt": current.get("foundAt", now) if current else now,
            "updatedAt": now,
        }
        if current is None:
            discoveries.append(record)
            outcome = "added"
        else:
            discoveries[discoveries.index(current)] = record
            outcome = "shorter"

        discoveries.sort(key=lambda item: (item["moveCount"], item["updatedAt"], item["id"]))
        payload["updatedAt"] = now
        self._write(payload)
        return outcome

    def _read(self) -> dict:
        if not self.path.exists():
            return _new_payload()
        try:
            with self.path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as error:
            raise _validation_error(str(self.path), "not valid JSON") from error
        if isinstance(payload, dict) and "schemaVersion" not in payload and payload.get("version") == 1:
            # Files written before schemaVersion was introduced are migrated on
            # the next save, without changing the individual discoveries.
            payload = dict(payload)
            payload.pop("version")
            payload.setdefault("updatedAt", None)
            payload["schemaVersion"] = DISCOVERY_SCHEMA_VERSION
        _validate_payload(payload, str(self.path))
        return payload

    def _write(self, payload: dict) -> None:
        _validate_payload(payload, str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary_path.replace(self.path)

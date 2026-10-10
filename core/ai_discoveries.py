"""Persist successful AI solves for the static web showcase."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from core.myperm_points import MypermPointCalculator, load_myperm_points


DISCOVERIES_FILE_NAME = "ai-discoveries.json"
DISCOVERY_SCHEMA_VERSION = 1
_PAYLOAD_FIELDS = frozenset({"schemaVersion", "updatedAt", "discoveries"})
_RECORD_FIELDS = frozenset(
    {"id", "puzzle", "setup", "moves", "moveCount", "foundAt", "updatedAt"}
)
_OPTIONAL_RECORD_FIELDS = frozenset({"discoveryKind"})
_DISCOVERY_KINDS = frozenset({"full-solve", "terminal-last-perm"})
_EFFECT_FIELDS = frozenset(
    {"effectName", "effectClass", "effectLabel", "effectCount", "orientationCount"}
)

_EFFECT_PART_LABELS = {
    "C": "コーナー",
    "E": "エッジ",
    "EAll": "エッジ束",
    "ME": "中エッジ",
    "W1": "第1ウイング",
    "W2": "第2ウイング",
    "W3": "第3ウイング",
    "CtrCore": "中心センター",
    "CtrObl": "斜めセンター",
    "CtrPlus": "十字センター",
    "CtrX": "X型センター",
}

# Keep the complete set of discoveries that can be presented on the public
# showcase. Outside that set, one concise representative per visible effect is
# enough for the local archive; retaining every variation makes the checked-in
# feed impractically large.
PUBLIC_EFFECT_COUNT_LIMIT = 5
COMPACT_EFFECT_COUNT_LIMIT = 10
COMPACT_MOVE_COUNT_LIMIT = 10
FEATURED_EFFECT_COMPONENT_PATTERNS = frozenset(
    tuple(sorted(parts))
    for parts in (
        ("C2", "CtrCore4", "ME2"),
        ("C2", "CtrCore6", "ME2"),
        ("C2", "CtrCore4"),
        ("C2", "CtrCore6"),
    )
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


def _record_id(puzzle: str, setup: list[str], discovery_kind: str = "full-solve") -> str:
    parts = (puzzle, *setup)
    if discovery_kind != "full-solve":
        parts = (discovery_kind, *parts)
    return sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _effect_component_type(effect_part: str) -> str:
    """Normalize component variants used by the showcase's featured effects."""
    if effect_part.startswith("C2"):
        return "C2"
    if effect_part.startswith("CtrCore4"):
        return "CtrCore4"
    if effect_part.startswith("CtrCore6"):
        return "CtrCore6"
    if effect_part.startswith("ME2"):
        return "ME2"
    return effect_part


def is_public_discovery(discovery: dict) -> bool:
    """Return whether the static Web viewer is allowed to present a discovery."""
    effect_count = discovery.get("effectCount")
    moves = discovery.get("moves")
    effect_class = discovery.get("effectClass")
    if not isinstance(effect_count, int) or effect_count <= 0:
        return False
    if not isinstance(moves, list) or not moves:
        return False
    if not isinstance(effect_class, str) or not effect_class:
        return False
    if not all(isinstance(discovery.get(field), str) and discovery[field] for field in (
        "effectName", "effectLabel",
    )):
        return False
    if not isinstance(discovery.get("orientationCount"), int):
        return False
    featured_pattern = tuple(sorted(
        _effect_component_type(part) for part in effect_class.split("+")
    ))
    return (
        effect_count <= PUBLIC_EFFECT_COUNT_LIMIT
        or (
            effect_count <= COMPACT_EFFECT_COUNT_LIMIT
            and len(moves) <= COMPACT_MOVE_COUNT_LIMIT
        )
        or featured_pattern in FEATURED_EFFECT_COMPONENT_PATTERNS
    )


def compact_discoveries(payload: dict) -> dict:
    """Retain every public replay and one best replay for each other effect.

    Full solves and terminal last-perm discoveries stay separate. Older records
    without effect metadata are retained individually instead of being grouped
    into an unknown effect.
    """
    discoveries = payload["discoveries"]
    best_by_effect = {}
    for discovery in discoveries:
        effect_class = discovery.get("effectClass")
        discovery_kind = discovery.get("discoveryKind", "full-solve")
        if not isinstance(effect_class, str) or not effect_class:
            key = ("legacy", discovery["id"])
        else:
            key = (discovery["puzzle"], effect_class, discovery_kind)
        priority = (
            discovery["moveCount"],
            len(discovery["setup"]),
            discovery["foundAt"],
            discovery["id"],
        )
        previous = best_by_effect.get(key)
        if previous is None or priority < previous[0]:
            best_by_effect[key] = (priority, discovery["id"])

    retained_ids = {
        discovery["id"] for discovery in discoveries if is_public_discovery(discovery)
    }
    retained_ids.update(record_id for _, record_id in best_by_effect.values())
    compacted = dict(payload)
    compacted["discoveries"] = [
        discovery for discovery in discoveries if discovery["id"] in retained_ids
    ]
    return compacted


def point_canonical_discovery_sequences(cube, setup, moves, *, point_calculator = None) -> tuple[tuple, tuple]:
    """Return setup and solution in the same highest-point orientation.

    Tied solution scores use setup and move text to select one stable orientation
    across equivalent records.  Setup and solution always receive the same
    symmetry transform, preserving their solve relationship.
    """
    source_setup = tuple(setup)
    source_moves = tuple(moves)
    calculator = point_calculator or MypermPointCalculator(
        cube,
        load_myperm_points(puzzle = getattr(cube, "myperm_point_puzzle", None)),
    )
    transform_count = len(getattr(cube, "transformation_keys", ()))
    if not transform_count:
        return source_setup, source_moves
    best = None
    for transform_index in range(transform_count):
        transformed_moves = tuple(cube.transform(source_moves, transform_index))
        transformed_setup = tuple(cube.transform(source_setup, transform_index))
        point = calculator.point_for_moves(transformed_moves)
        display_key = (
            tuple(_clean_moves(transformed_moves)),
            tuple(_clean_moves(transformed_setup)),
            transform_index,
        )
        if best is None or point > best[0] or (point == best[0] and display_key < best[1]):
            best = (point, display_key, transformed_setup, transformed_moves)
    return best[2], best[3]


def terminal_last_perm_sequences(setup, move_rows) -> tuple[tuple, tuple] | None:
    """Return the replayable final search step from a multi-step solve.

    The final row is stored with the state reached by all preceding rows, so
    its effect remains visible instead of being absorbed into the full solve.
    """
    rows = tuple(tuple(row) for row in move_rows)
    if len(rows) < 2 or not rows[-1]:
        return None
    preceding_moves = tuple(move for row in rows[:-1] for move in row)
    if not preceding_moves:
        return None
    return tuple(setup) + preceding_moves, rows[-1]


def _effect_component_label(component) -> str:
    """Return a short Japanese description of one visible effect component."""
    cycle_lengths = tuple(sorted(len(cycle) for cycle in component.cycles))
    operations = []
    for length in sorted(set(cycle_lengths)):
        count = cycle_lengths.count(length)
        if length == 2:
            operations.append(f"{count}組交換")
        else:
            suffix = f"×{count}" if count > 1 else ""
            operations.append(f"{length}巡回{suffix}")
    if not operations:
        operations.append(f"{component.moved_count}個移動")
    if component.orientation_count:
        operations.append(f"向き変化{component.orientation_count}")
    part_label = _EFFECT_PART_LABELS.get(component.part_code, component.part_code)
    return f"{part_label}：{'・'.join(operations)}"


def _effect_label(components) -> str:
    return "＋".join(_effect_component_label(component) for component in components)


def discovery_effect_metadata(effect) -> dict:
    """Build compact, display-ready effect information for a discovery."""
    visible_components = tuple(
        component
        for component in effect.components
        if not component.is_internal_center_permutation()
    )
    effect_count = sum(component.moved_count for component in visible_components)
    orientation_count = sum(component.orientation_count for component in visible_components)
    if effect_count <= 0:
        raise ValueError("discovery effect must move at least one visible piece")

    effect_name = effect.concise_name()
    effect_class = effect.concise_name(max_positions = 0)
    if not effect_name or not effect_class or effect_name == "Identity":
        raise ValueError("discovery effect must have a visible effect name")
    return {
        "effectName": effect_name,
        "effectClass": effect_class,
        "effectLabel": _effect_label(visible_components),
        "effectCount": effect_count,
        "orientationCount": orientation_count,
    }


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


def _validate_effect_metadata(metadata, path: str) -> dict:
    if not isinstance(metadata, dict) or set(metadata) != _EFFECT_FIELDS:
        raise _validation_error(path, f"effect metadata must contain exactly {sorted(_EFFECT_FIELDS)}")
    for field in ("effectName", "effectClass", "effectLabel"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise _validation_error(path, f"{field} must be a non-empty string")
    for field in ("effectCount", "orientationCount"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _validation_error(path, f"{field} must be a non-negative integer")
    if metadata["effectCount"] <= 0:
        raise _validation_error(path, "effectCount must be positive")
    if metadata["orientationCount"] > metadata["effectCount"]:
        raise _validation_error(path, "orientationCount cannot exceed effectCount")
    return dict(metadata)


def _validate_record(record, index: int, path: str) -> None:
    location = f"discoveries[{index}]"
    if not isinstance(record, dict):
        raise _validation_error(path, f"{location} must be an object")
    fields = set(record)
    non_optional_fields = fields - _OPTIONAL_RECORD_FIELDS
    allowed_fields = _RECORD_FIELDS | _EFFECT_FIELDS
    if non_optional_fields not in (_RECORD_FIELDS, allowed_fields):
        raise _validation_error(
            path,
            f"{location} must contain base fields with optional complete effect metadata",
        )
    if "discoveryKind" in record and record["discoveryKind"] not in _DISCOVERY_KINDS:
        raise _validation_error(path, f"{location}.discoveryKind is invalid")

    discovery_kind = record.get("discoveryKind", "full-solve")
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
    expected_id = _record_id(puzzle, setup, discovery_kind)
    if not isinstance(record_id, str) or record_id != expected_id:
        raise _validation_error(path, f"{location}.id does not match puzzle and setup")
    _validate_timestamp(record["foundAt"], f"{location}.foundAt", path)
    _validate_timestamp(record["updatedAt"], f"{location}.updatedAt", path)
    if non_optional_fields == allowed_fields:
        _validate_effect_metadata({field: record[field] for field in _EFFECT_FIELDS}, location)


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
    """Store one concise replay for each discovered puzzle procedure."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_discoveries_path()

    def save(self, puzzle: str, setup, moves, *, effect_metadata = None, discovery_kind = "full-solve") -> str:
        """Save a discovery and return ``added``, ``shorter``, or ``unchanged``."""
        normalized_puzzle = str(puzzle).strip()
        clean_setup = _clean_moves(setup)
        clean_moves = _clean_moves(moves)
        if not normalized_puzzle:
            raise ValueError("puzzle is required")
        if not clean_moves:
            raise ValueError("moves is required")
        if discovery_kind not in _DISCOVERY_KINDS:
            raise ValueError("discovery_kind is invalid")

        payload = self._read()
        discoveries = payload["discoveries"]
        record_id = _record_id(normalized_puzzle, clean_setup, discovery_kind)
        current = next((item for item in discoveries if item.get("id") == record_id), None)
        procedure_matches = [
            item
            for item in discoveries
            if item["puzzle"] == normalized_puzzle and item["moves"] == clean_moves
        ]

        def procedure_priority(item):
            return (
                len(item["setup"]),
                0 if item.get("discoveryKind", "full-solve") == "full-solve" else 1,
                item["foundAt"],
                item["id"],
            )

        incoming_priority = (
            len(clean_setup),
            0 if discovery_kind == "full-solve" else 1,
            "",
            record_id,
        )
        previous_found_at = None
        replaced_procedure = False
        if procedure_matches:
            representative = min(procedure_matches, key=procedure_priority)
            if procedure_priority(representative) <= incoming_priority:
                return "unchanged"
            previous_found_at = representative["foundAt"]
            replaced_procedure = True
            discoveries[:] = [item for item in discoveries if item not in procedure_matches]
            current = None
        elif current is not None and len(current.get("moves", ())) <= len(clean_moves):
            return "unchanged"

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "id": record_id,
            "puzzle": normalized_puzzle,
            "setup": clean_setup,
            "moves": clean_moves,
            "moveCount": len(clean_moves),
            "foundAt": previous_found_at or (current.get("foundAt", now) if current else now),
            "updatedAt": now,
        }
        if effect_metadata is not None:
            record.update(_validate_effect_metadata(effect_metadata, "effect metadata"))
        if discovery_kind != "full-solve":
            record["discoveryKind"] = discovery_kind
        if current is None:
            discoveries.append(record)
            outcome = "shorter" if replaced_procedure else "added"
        else:
            discoveries[discoveries.index(current)] = record
            outcome = "shorter"

        discoveries.sort(key=lambda item: (item["moveCount"], item["updatedAt"], item["id"]))
        payload["updatedAt"] = now
        retained_payload = compact_discoveries(payload)
        discoveries[:] = retained_payload["discoveries"]
        if not any(item["id"] == record_id for item in discoveries):
            return "unchanged"
        self._write(payload)
        return outcome

    def compact(self) -> tuple[int, int]:
        """Prune redundant discoveries and rewrite the feed in compact JSON."""
        payload = self._read()
        previous_count = len(payload["discoveries"])
        payload = compact_discoveries(payload)
        self._write(payload)
        return previous_count, len(payload["discoveries"])

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
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        temporary_path.replace(self.path)

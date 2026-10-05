"""Reorient saved AI discoveries to maximum myperm points and merge duplicates."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_discoveries import (
    AiDiscoveryStore,
    _EFFECT_FIELDS,
    _record_id,
    _validate_payload,
    discovery_effect_metadata,
    point_canonical_discovery_sequences,
)
from core.myperm_effects import MypermEffectAnalyzer
from core.myperm_points import MypermPointCalculator, load_myperm_points
from cube.rubiks_cube import Rubiks_3


_CUBE_PUZZLE = re.compile(r"^(?P<size>\d+)x\1x\1$")


def _internal_move(display_move):
    token = str(display_move).strip()
    if not token:
        raise ValueError("discovery contains an empty move")
    if token[0].isdigit():
        return token if len(token) > 2 else token + " "
    return f" {token} " if len(token) == 1 else f" {token}"


def _cube_size(puzzle):
    match = _CUBE_PUZZLE.fullmatch(puzzle)
    if match is None:
        raise ValueError(f"unsupported discovery puzzle: {puzzle}")
    return int(match.group("size"))


def _color_solved(cube, setup, moves):
    state = cube.state_0.copy()
    for move in setup + moves:
        state = state[np.asarray(cube.move[move], dtype=int)]
    return bool(np.array_equal(state, cube.state_0))


def _display_moves(moves):
    return [str(move).strip() for move in moves]


def _merge_timestamps(winner, loser):
    winner["foundAt"] = min(winner["foundAt"], loser["foundAt"])
    winner["updatedAt"] = max(winner["updatedAt"], loser["updatedAt"])
    return winner


def _start_priority(record):
    return (record["moveCount"], record["foundAt"], record["updatedAt"], tuple(record["moves"]), record["id"])


def _procedure_priority(record):
    return (
        len(record["setup"]),
        0 if record.get("discoveryKind", "full-solve") == "full-solve" else 1,
        record["foundAt"],
        record["id"],
    )


def migrate_payload(payload, *, migrated_at=None, progress=None):
    """Return a validated, point-canonical payload without modifying input."""
    _validate_payload(payload, "input")
    migrated_at = migrated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    contexts = {}
    by_start = {}
    stats = Counter()
    source_count = len(payload["discoveries"])

    for index, source in enumerate(payload["discoveries"], start=1):
        size = _cube_size(source["puzzle"])
        if size not in contexts:
            cube = Rubiks_3(size=size, RegisterMyperms=False)
            contexts[size] = (
                cube,
                MypermPointCalculator(cube, load_myperm_points(puzzle="cube")),
                MypermEffectAnalyzer(cube),
            )
        cube, calculator, analyzer = contexts[size]
        setup = tuple(_internal_move(move) for move in source["setup"])
        moves = tuple(_internal_move(move) for move in source["moves"])
        if not _color_solved(cube, setup, moves):
            stats["invalid_source_replays"] += 1

        canonical_setup, canonical_moves = point_canonical_discovery_sequences(
            cube, setup, moves, point_calculator=calculator,
        )
        display_setup = _display_moves(canonical_setup)
        display_moves = _display_moves(canonical_moves)
        if display_setup != source["setup"] or display_moves != source["moves"]:
            stats["reoriented"] += 1
        if not _color_solved(cube, canonical_setup, canonical_moves):
            stats["invalid_migrated_replays"] += 1

        record = dict(source)
        record["setup"] = display_setup
        record["moves"] = display_moves
        record["moveCount"] = len(display_moves)
        kind = record.get("discoveryKind", "full-solve")
        record["id"] = _record_id(record["puzzle"], display_setup, kind)
        if record["id"] != source["id"]:
            stats["changed_ids"] += 1

        try:
            metadata = discovery_effect_metadata(analyzer.analyze(canonical_moves))
        except ValueError:
            if any(field in source for field in _EFFECT_FIELDS):
                raise ValueError(f"existing effect metadata became invalid at source record {index}")
            stats["without_visible_effect"] += 1
        else:
            if any(source.get(field) != value for field, value in metadata.items()):
                stats["updated_effects"] += 1
            record.update(metadata)

        existing = by_start.get(record["id"])
        if existing is None:
            by_start[record["id"]] = record
        else:
            if (existing["puzzle"], existing["setup"], existing.get("discoveryKind", "full-solve")) != (
                record["puzzle"], record["setup"], kind,
            ):
                raise ValueError(f"discovery ID hash collision: {record['id']}")
            stats["merged_same_start"] += 1
            winner, loser = sorted((existing, record), key=_start_priority)
            by_start[record["id"]] = _merge_timestamps(winner, loser)

        if progress is not None and (index % 1000 == 0 or index == source_count):
            progress(index, source_count, stats)

    by_procedure = {}
    for record in by_start.values():
        key = (record["puzzle"], tuple(record["moves"]))
        existing = by_procedure.get(key)
        if existing is None:
            by_procedure[key] = record
        else:
            stats["merged_same_procedure"] += 1
            winner, loser = sorted((existing, record), key=_procedure_priority)
            by_procedure[key] = _merge_timestamps(winner, loser)

    records = sorted(
        by_procedure.values(),
        key=lambda item: (item["moveCount"], item["updatedAt"], item["id"]),
    )
    stats["source_count"] = source_count
    stats["result_count"] = len(records)
    if stats["invalid_migrated_replays"] != stats["invalid_source_replays"]:
        raise ValueError("migration changed the number of invalid replay relationships")
    changed = any(stats[field] for field in (
        "reoriented", "changed_ids", "updated_effects", "merged_same_start", "merged_same_procedure",
    ))
    result = {
        "schemaVersion": payload["schemaVersion"],
        "updatedAt": migrated_at if changed else payload["updatedAt"],
        "discoveries": records,
    }
    _validate_payload(result, "migrated")
    return result, stats


def _backup_path(source):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return source.with_name(f"ai-discoveries.pre-point-migration-{stamp}.json.gz")


def migrate_file(path, *, write=False, progress=None):
    path = Path(path)
    original_bytes = path.read_bytes()
    original_mode = path.stat().st_mode & 0o777
    original_digest = sha256(original_bytes).digest()
    payload = json.loads(original_bytes)
    migrated, stats = migrate_payload(payload, progress=progress)
    backup_path = None
    if write and migrated != payload:
        if sha256(path.read_bytes()).digest() != original_digest:
            raise RuntimeError("discovery file changed during migration; rerun with the newest data")
        backup_path = _backup_path(path)
        with backup_path.open("xb") as stream:
            stream.write(gzip.compress(original_bytes, compresslevel=6))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(migrated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(temporary_path, original_mode)
        try:
            if sha256(path.read_bytes()).digest() != original_digest:
                raise RuntimeError("discovery file changed before replacement; backup retained")
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        AiDiscoveryStore(path)._read()
    return stats, backup_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "exports" / "ai-discoveries.json")
    parser.add_argument("--write", action="store_true", help="back up and replace the validated feed")
    args = parser.parse_args()

    def progress(done, total, stats):
        print(f"normalized {done}/{total}; reoriented={stats['reoriented']}", flush=True)

    stats, backup_path = migrate_file(args.input, write=args.write, progress=progress)
    print("migration:", ", ".join(f"{name}={value}" for name, value in sorted(stats.items())))
    if backup_path is not None:
        print(f"backup: {backup_path}")


if __name__ == "__main__":
    main()

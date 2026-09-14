"""Backfill effect metadata used by the AI discovery web showcase."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_discoveries import discovery_effect_metadata
from core.myperm_effects import MypermEffectAnalyzer
from cube.rubiks_cube import Rubiks_3


_CUBE_PUZZLE = re.compile(r"^(?P<size>\d+)x\1x\1$")


def _cube_size(puzzle: str) -> int:
    match = _CUBE_PUZZLE.fullmatch(puzzle)
    if match is None:
        raise ValueError(f"unsupported puzzle for effect backfill: {puzzle}")
    return int(match.group("size"))


def _internal_rubiks_move(display_move: str) -> str:
    """Convert the compact web notation to the cube model's padded move key."""
    move = str(display_move).strip()
    if not move:
        raise ValueError("discovery contains an empty move")
    if move[0].isdigit():
        return move if len(move) > 2 else move + " "
    return f" {move} " if len(move) == 1 else f" {move}"


def add_effect_metadata(payload: dict) -> int:
    cubes = {}
    analyzers = {}
    updated = 0
    for discovery in payload.get("discoveries", []):
        size = _cube_size(discovery["puzzle"])
        cube = cubes.get(size)
        if cube is None:
            cube = Rubiks_3(size=size, RegisterMyperms=False)
            cubes[size] = cube
            analyzers[size] = MypermEffectAnalyzer(cube)
        analyzer = analyzers[size]
        moves = tuple(_internal_rubiks_move(move) for move in discovery["moves"])
        metadata = discovery_effect_metadata(analyzer.analyze(moves))
        if any(discovery.get(key) != value for key, value in metadata.items()):
            discovery.update(metadata)
            updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("exports/ai-discoveries.json"),
        help="AI discovery JSON file to enrich",
    )
    parser.add_argument("--write", action="store_true", help="write the enriched JSON")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    updated = add_effect_metadata(payload)
    print(f"{args.input}: {updated} records enriched")
    if not args.write or not updated:
        return

    payload["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.input.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

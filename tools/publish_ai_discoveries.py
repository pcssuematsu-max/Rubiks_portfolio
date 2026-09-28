"""Publish the validated AI discovery feed to the companion web project."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_discoveries import AiDiscoveryStore


DISCOVERIES_FILE_NAME = "ai-discoveries.json"
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


def _effect_component_type(effect_part: str) -> str:
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
    """Match the static site's public AI Discovery selection rules."""
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


def compact_public_payload(payload: dict) -> dict:
    """Return only records the public Web viewer can actually display."""
    compacted = dict(payload)
    compacted["discoveries"] = [
        discovery for discovery in payload["discoveries"]
        if is_public_discovery(discovery)
    ]
    compacted["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return compacted


def default_web_target() -> Path | None:
    """Find the local companion GitHub Pages feed when it is checked out nearby."""
    configured_path = os.environ.get("TWISTY_WEB_DISCOVERIES_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    documents_directory = PROJECT_ROOT.parents[1]
    candidate = documents_directory / "My Github Pages" / "assets" / "data" / DISCOVERIES_FILE_NAME
    return candidate if candidate.parent.exists() else None


def publish_discoveries(source: Path, target: Path) -> int:
    """Validate ``source`` and atomically replace ``target`` with its payload."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target:
        raise ValueError("source and target must be different files")
    if not target.parent.is_dir():
        raise ValueError(f"target directory does not exist: {target.parent}")

    payload = compact_public_payload(AiDiscoveryStore(source)._read())
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(payload["discoveries"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "exports" / DISCOVERIES_FILE_NAME,
        help="validated discovery feed produced by the Python app",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=default_web_target(),
        help="companion web project's assets/data/ai-discoveries.json",
    )
    args = parser.parse_args()
    if args.target is None:
        parser.error("--target is required when the companion web project is not found locally")

    count = publish_discoveries(args.source, args.target)
    print(f"Published {count} discoveries: {args.source} -> {args.target}")


if __name__ == "__main__":
    main()

"""Prune redundant AI discoveries and rewrite their JSON feed compactly."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_discoveries import AiDiscoveryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "exports" / "ai-discoveries.json",
        help="AI discovery feed to compact",
    )
    args = parser.parse_args()
    previous_count, retained_count = AiDiscoveryStore(args.input).compact()
    print(f"Compacted {args.input}: {previous_count} -> {retained_count} discoveries")


if __name__ == "__main__":
    main()

"""Standalone CLI for inspecting myperm effects."""

import argparse
import pprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cube.myperm_analyzer import MypermAnalyzer
from cube.rubiks_cube import Rubiks_3
from megaminx.cube import MegaminxCube


def build_parser():
    parser = argparse.ArgumentParser(description = "Inspect what a myperm changes.")
    parser.add_argument("key", help = "myperm key or move-sequence label")
    parser.add_argument("--puzzle", choices = ["rubiks", "megaminx"], default = "rubiks", help = "puzzle type")
    parser.add_argument("--size", type = int, default = 7, help = "cube size for Rubiks only")
    parser.add_argument(
        "--direct",
        action = "store_true",
        help = "apply the moves directly instead of analyzing invert_moves(moves)",
    )
    parser.add_argument(
        "--no-labeled-centers",
        action = "store_true",
        help = "skip unique-label center permutation analysis",
    )
    parser.add_argument(
        "--details",
        action = "store_true",
        help = "print full structured analysis",
    )
    return parser


def main():
    args = build_parser().parse_args()
    cube = Rubiks_3(size = args.size) if args.puzzle == "rubiks" else MegaminxCube()
    analyzer = MypermAnalyzer(cube)
    result = analyzer.analyze(
        args.key,
        use_inverse = not args.direct,
        labeled_centers = not args.no_labeled_centers,
    )

    print(f"key: {result['display_key'] or result['key']}")
    print(f"moves: {result['display_moves']}")
    print(f"applied_moves: {result['display_applied_moves']}")
    print(f"use_inverse: {result['use_inverse']}")
    print(analyzer.summarize(args.key, use_inverse = not args.direct, labeled_centers = not args.no_labeled_centers))

    if args.details:
        print()
        pprint.pprint(result)


if __name__ == "__main__":
    main()

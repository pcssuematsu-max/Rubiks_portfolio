"""Fit a first, unregularized ergonomic ranking model to candidate algorithms.

The input algorithms are treated as provisional preferred representatives of
their rotation/mirror orbits. This is a ranking experiment, not a measurement
of execution time or proof that the input algorithms are ergonomically best.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, lil_matrix

from cube.rubiks_cube import Rubiks_3


HERE = Path(__file__).resolve().parent
MARGIN = 0.1


@dataclass(frozen=True)
class Example:
    id: str
    original: tuple[str, ...]
    variants: tuple[tuple[str, ...], ...]


def read_candidates() -> list[dict[str, str]]:
    with (HERE / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate ids must be unique")
    return rows


def parse_algorithm(value: str) -> tuple[str, ...]:
    moves = value.replace("’", "'").split()
    if not moves:
        raise ValueError("Empty algorithm")
    for move in moves:
        if not re.fullmatch(r"(?:[23][UDFBLR]|[UDFBLRMSE])(?:['2])?", move):
            raise ValueError(f"Unsupported move: {move}")
    return tuple(moves)


def internal(move: str) -> str:
    prefix = move[0] if move[0] in "23" else " "
    face = move[1] if prefix != " " else move[0]
    suffix = move[-1] if move[-1] in "'2" else " "
    return prefix + face + suffix


def external(move: str) -> str:
    return move.strip()


def generate_examples(rows: list[dict[str, str]], transform_indices=None) -> list[Example]:
    cubes = {
        size: Rubiks_3(size=size, RegisterMyperms=False, PointReindex=False)
        for size in (3, 7)
    }
    examples = []
    for row in rows:
        if row["puzzle"] not in ("3x3", "7x7"):
            raise ValueError(f"Unsupported puzzle: {row['id']}: {row['puzzle']}")
        cube = cubes[int(row["puzzle"][0])]
        indices = range(len(cube.transformation_keys)) if transform_indices is None else transform_indices
        original = parse_algorithm(row["algorithm"])
        encoded = tuple(map(internal, original))
        if any(move not in cube.move for move in encoded):
            raise ValueError(f"Move not available on {row['puzzle']}: {row['id']}")
        variants = {
            tuple(map(external, cube.transform(encoded, i, flip_inside=flip_inside)))
            for i in indices
            for flip_inside in ((False, True) if row["puzzle"] == "7x7" else (False,))
        }
        variants.discard(original)
        examples.append(Example(row["id"], original, tuple(sorted(variants))))
    return examples


def features(moves: tuple[str, ...], use_pairs: bool) -> Counter[str]:
    counts = Counter("move:" + move for move in moves)
    if use_pairs:
        counts.update("pair:" + left + " " + right for left, right in zip(moves, moves[1:]))
    return counts


def feature_matrix(examples: list[Example], use_pairs: bool):
    all_sequences = [sequence for example in examples for sequence in (example.original, *example.variants)]
    vocabulary = sorted({name for sequence in all_sequences for name in features(sequence, use_pairs)})
    lookup = {name: i for i, name in enumerate(vocabulary)}
    vectors = {}
    for sequence in all_sequences:
        vector = np.zeros(len(vocabulary))
        for name, count in features(sequence, use_pairs).items():
            vector[lookup[name]] = count
        vectors[sequence] = vector
    return vocabulary, vectors


def fit(examples: list[Example], vectors: dict[tuple[str, ...], np.ndarray]):
    dimension = len(next(iter(vectors.values())))
    pairs = [(example, variant) for example in examples for variant in example.variants]
    if not pairs:
        raise ValueError("No nontrivial transformed algorithms")
    matrix = lil_matrix((len(pairs), dimension + len(pairs)))
    rhs = np.full(len(pairs), -MARGIN)
    costs = np.zeros(dimension + len(pairs))
    for row_index, (example, variant) in enumerate(pairs):
        difference = vectors[example.original] - vectors[variant]
        matrix[row_index, :dimension] = difference
        matrix[row_index, dimension + row_index] = -1
        costs[dimension + row_index] = 1 / (len(examples) * len(example.variants))
    equality = hstack([csr_matrix(np.ones((1, dimension))), csr_matrix((1, len(pairs)))], format="csr")
    solution = linprog(
        costs,
        A_ub=matrix.tocsr(),
        b_ub=rhs,
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=(0, None),
        method="interior-point",
        options={"tol": 1e-8},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.x[:dimension]


def evaluate(example: Example, weight: np.ndarray, vectors: dict[tuple[str, ...], np.ndarray]) -> dict:
    base = float(vectors[example.original] @ weight)
    gaps = np.array([float(vectors[variant] @ weight) - base for variant in example.variants])
    return {
        "id": example.id,
        "variants": len(example.variants),
        "strictly_preferred": int(np.count_nonzero(gaps > 1e-8)),
        "margin_satisfied": int(np.count_nonzero(gaps >= MARGIN - 1e-8)),
        "mean_gap": round(float(gaps.mean()), 6),
        "minimum_gap": round(float(gaps.min()), 6),
        "mean_slack": round(float(np.maximum(MARGIN - gaps, 0).mean()), 6),
    }


def run_model(examples: list[Example], use_pairs: bool) -> dict:
    vocabulary, vectors = feature_matrix(examples, use_pairs)
    weight = fit(examples, vectors)
    in_sample = [evaluate(example, weight, vectors) for example in examples]
    out_of_sample = []
    for held_out in examples:
        training = [example for example in examples if example.id != held_out.id]
        left_out_weight = fit(training, vectors)
        out_of_sample.append(evaluate(held_out, left_out_weight, vectors))
    nonzero = [(vocabulary[i], float(value)) for i, value in enumerate(weight) if value > 1e-8]
    nonzero.sort(key=lambda item: -item[1])
    return {
        "feature_count": len(vocabulary),
        "nonzero_coefficient_count": len(nonzero),
        "coefficients": [{"feature": name, "weight": round(value, 6)} for name, value in nonzero],
        "training": in_sample,
        "leave_one_algorithm_out": out_of_sample,
    }


def main() -> None:
    rows = read_candidates()
    examples = generate_examples(rows)
    result = {
        "method": "nonnegative coefficients summing to 1; minimize mean hinge slack per source algorithm",
        "margin": MARGIN,
        "assumption": "Each source algorithm is preferred to its distinct rotation/mirror variants",
        "candidate_count": len(examples),
        "models": {
            "single_moves": run_model(examples, False),
            "single_moves_and_adjacent_pairs": run_model(examples, True),
        },
    }
    path = HERE / "pilot_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")
    for name, model in result["models"].items():
        for kind in ("training", "leave_one_algorithm_out"):
            rows = model[kind]
            good = sum(row["strictly_preferred"] for row in rows)
            total = sum(row["variants"] for row in rows)
            print(name, kind, f"{good}/{total}", "nonzero", model["nonzero_coefficient_count"])


if __name__ == "__main__":
    main()

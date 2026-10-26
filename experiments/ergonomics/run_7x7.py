"""Re-evaluate all source algorithms as 7x7 move sequences.

All 96 rotation/reflection/2–3 swap transforms and both inside-flip
states are enumerated. Source rows are provisional preferred representatives.
"""

from __future__ import annotations

import json

import numpy as np

from cube.rubiks_cube import Rubiks_3
from experiments.ergonomics.run_l2 import model_result
from experiments.ergonomics.run_pilot import (
    HERE,
    external,
    feature_matrix,
    generate_examples,
    internal,
    read_candidates,
)


def trace_transform(original, target):
    cube = Rubiks_3(size=7, RegisterMyperms=False, PointReindex=False)
    encoded = tuple(map(internal, original))
    for index, key in enumerate(cube.transformation_keys):
        for inside in (False, True):
            transformed = tuple(map(external, cube.transform(encoded, index, flip_inside=inside)))
            if transformed == target:
                return {"index": index, "key": list(key), "flip_inside": inside}
    raise ValueError("Cannot trace a transformed sequence")


def best_variants(examples, rows, result, use_pairs):
    vocabulary, vectors = feature_matrix(examples, use_pairs)
    weights_by_name = {item["feature"]: item["weight"] for item in result["coefficients"]}
    weights = np.array([weights_by_name[name] for name in vocabulary])
    puzzle_by_id = {row["id"]: row["puzzle"] for row in rows}
    output = []
    for example in examples:
        original_cost = float(vectors[example.original] @ weights)
        ranked = sorted(
            ((float(vectors[variant] @ weights), variant) for variant in example.variants),
            key=lambda pair: (pair[0], pair[1]),
        )
        best_cost, best_moves = ranked[0]
        output.append({
            "id": example.id,
            "source_puzzle": puzzle_by_id[example.id],
            "analysis_puzzle": "7x7",
            "source_algorithm": " ".join(example.original),
            "source_cost": round(original_cost, 6),
            "best_transformed_algorithm": " ".join(best_moves),
            "best_transformed_cost": round(best_cost, 6),
            "cost_difference": round(best_cost - original_cost, 6),
            "strictly_lower_variants": sum(cost < original_cost - 1e-8 for cost, _ in ranked),
            "tied_variants": sum(abs(cost - original_cost) <= 1e-8 for cost, _ in ranked),
            "best_transform": trace_transform(example.original, best_moves),
        })
    return output


def main():
    rows = read_candidates()
    examples = generate_examples([{**row, "puzzle": "7x7"} for row in rows])
    models = {}
    for name, use_pairs in (("single_moves", False), ("single_moves_and_adjacent_pairs", True)):
        result = model_result(examples, use_pairs, 0.1)
        result["best_variants"] = best_variants(examples, rows, result, use_pairs)
        models[name] = result
        for group, included in (("3x3", examples[:10]), ("7x7", examples[10:])):
            selected_ids = {example.id for example in included}
            held_out = [item for item in result["leave_one_algorithm_out"] if item["id"] in selected_ids]
            training = [item for item in result["training"] if item["id"] in selected_ids]
            print(
                name, group,
                "fit", sum(x["strictly_preferred"] for x in training), "/", sum(x["variants"] for x in training),
                "LOO", sum(x["strictly_preferred"] for x in held_out), "/", sum(x["variants"] for x in held_out),
                "mean LOO slack", round(sum(x["mean_slack"] for x in held_out) / len(held_out), 6),
                flush=True,
            )
    output = {
        "method": "provisional preference for supplied source over transformed variants; mean hinge slack per source plus lambda/2 L2 penalty",
        "margin": 0.1,
        "lambda": 0.1,
        "source_count": len(examples),
        "large_cube_sources": sum(row["puzzle"] == "7x7" for row in rows),
        "comparison_count": sum(len(example.variants) for example in examples),
        "transform_policy": "All rows treated as 7x7: 96 rotation/reflection/2-3swap transforms times two flip_inside states; exact duplicate sequences removed",
        "models": models,
    }
    path = HERE / "results_7x7.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

"""Re-run the ranking experiment using only symmetries that keep U on top."""

from __future__ import annotations

import json

from cube.rubiks_cube import Rubiks_3
from experiments.ergonomics.run_l2 import model_result
from experiments.ergonomics.run_pilot import HERE, generate_examples, read_candidates, run_model


def main():
    cube = Rubiks_3(size=3, RegisterMyperms=False, PointReindex=False)
    indices = [
        index
        for index in range(len(cube.transformation_keys))
        if cube.transform((" U ",), index)[0][1] == "U"
    ]
    examples = generate_examples(read_candidates(), indices)
    results = {}
    for name, use_pairs in (("single_moves", False), ("single_moves_and_adjacent_pairs", True)):
        results[name] = {
            "without_l2": run_model(examples, use_pairs),
            "l2_lambda_0_1": model_result(examples, use_pairs, 0.1),
        }
    output = {
        "transformation_indices": indices,
        "transformation_keys": [cube.transformation_keys[index] for index in indices],
        "criterion": "The U face maps to the U face; turn direction may reverse under reflection",
        "candidate_count": len(examples),
        "comparison_count": sum(len(example.variants) for example in examples),
        "models": results,
    }
    path = HERE / "top_preserving_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("transformations", len(indices), "comparisons", output["comparison_count"])
    for name, models in results.items():
        for model_name, result in models.items():
            rows = result["leave_one_algorithm_out"]
            wins = sum(row["strictly_preferred"] for row in rows)
            total = sum(row["variants"] for row in rows)
            print(name, model_name, f"{wins}/{total}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

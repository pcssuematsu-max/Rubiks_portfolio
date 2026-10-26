"""Ablate the ranking margin and L2 term on the 7x7 symmetry experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ergonomics.run_7x7 import best_variants
from experiments.ergonomics.run_l2 import fit_l2
from experiments.ergonomics.run_pilot import (
    HERE,
    feature_matrix,
    fit,
    generate_examples,
    read_candidates,
)


def assess(example, weight, vectors, margin):
    original_cost = float(vectors[example.original] @ weight)
    gaps = np.array([float(vectors[variant] @ weight) - original_cost for variant in example.variants])
    return {
        "id": example.id,
        "variants": len(gaps),
        "wins": int(np.count_nonzero(gaps > 1e-8)),
        "ties": int(np.count_nonzero(np.abs(gaps) <= 1e-8)),
        "losses": int(np.count_nonzero(gaps < -1e-8)),
        "margin_satisfied": int(np.count_nonzero(gaps >= margin - 1e-8)),
        "mean_hinge_loss": round(float(np.maximum(margin - gaps, 0).mean()), 9),
    }


def fit_condition(examples, rows, use_pairs, margin, l2, leave_one_out):
    vocabulary, vectors = feature_matrix(examples, use_pairs)
    if l2:
        weight, optimization = fit_l2(examples, vectors, l2, margin=margin)
    else:
        if margin != 0.1:
            raise ValueError("The existing LP uses margin 0.1")
        weight = fit(examples, vectors)
        optimization = None
    result = {
        "margin": margin,
        "lambda": l2,
        "feature_count": len(vocabulary),
        "nonzero_coefficient_count": int(np.count_nonzero(weight > 1e-8)),
        "coefficients": [{"feature": name, "weight": round(float(value), 9)} for name, value in zip(vocabulary, weight)],
        "optimization": optimization,
        "training": [assess(example, weight, vectors, margin) for example in examples],
    }
    result["best_variants"] = best_variants(examples, rows, result, use_pairs)
    if leave_one_out:
        held_out = []
        for excluded in examples:
            included = [example for example in examples if example.id != excluded.id]
            excluded_weight, _ = fit_l2(included, vectors, l2, margin=margin)
            held_out.append(assess(excluded, excluded_weight, vectors, margin))
        result["leave_one_algorithm_out"] = held_out
    return result


def zero_margin_zero_l2_witnesses(examples, vocabulary, vectors):
    result = []
    for feature in ("move:2B", "move:S"):
        weight = np.zeros(len(vocabulary))
        weight[vocabulary.index(feature)] = 1.0
        training = [assess(example, weight, vectors, 0) for example in examples]
        result.append({"feature": feature, "objective": 0.0, "training": training})
    return result


def summarize(label, result):
    train = result["training"]
    print(
        label,
        "fit", sum(x["wins"] for x in train), "wins,", sum(x["ties"] for x in train),
        "ties,", sum(x["losses"] for x in train), "losses;",
        "reversed sources", [(x["id"], x["cost_difference"]) for x in result["best_variants"] if x["strictly_lower_variants"]],
        flush=True,
    )
    if "leave_one_algorithm_out" in result:
        held_out = result["leave_one_algorithm_out"]
        print(label, "LOO", sum(x["wins"] for x in held_out), "wins,", sum(x["ties"] for x in held_out), "ties,", sum(x["losses"] for x in held_out), "losses", flush=True)


def main():
    rows = read_candidates()
    examples = generate_examples([{**row, "puzzle": "7x7"} for row in rows])
    models = {}
    for name, use_pairs in (("single_moves", False), ("single_moves_and_adjacent_pairs", True)):
        models[name] = {}
        for condition, margin, l2, loo in (
            ("margin_0_l2_0_1", 0.0, 0.1, True),
            ("margin_0_1_l2_0", 0.1, 0.0, False),
        ):
            result = fit_condition(examples, rows, use_pairs, margin, l2, loo)
            models[name][condition] = result
            summarize(name + " " + condition, result)
        if not use_pairs:
            vocabulary, vectors = feature_matrix(examples, False)
            models[name]["margin_0_l2_0_multiple_optima"] = zero_margin_zero_l2_witnesses(examples, vocabulary, vectors)
    output = {
        "source_count": len(examples),
        "comparison_count": sum(len(example.variants) for example in examples),
        "analysis_puzzle": "7x7",
        "baseline": "results_7x7.json (margin=0.1, lambda=0.1)",
        "models": models,
    }
    path = Path(HERE) / "margin_l2_ablation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()

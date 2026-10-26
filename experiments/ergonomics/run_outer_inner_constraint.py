"""Pilot an ordering constraint between outer and inner single-move costs."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from experiments.ergonomics.run_7x7 import best_variants
from experiments.ergonomics.run_margin_l2_ablation import assess
from experiments.ergonomics.run_pilot import HERE, feature_matrix, generate_examples, read_candidates


MARGIN = 0.1
L2 = 0.1
MINIMUM_GAP = 0.001


def fit_ordered(examples, vectors, vocabulary, minimum_gap=MINIMUM_GAP):
    dimension = len(vocabulary)
    outer = np.array([i for i, name in enumerate(vocabulary) if name.startswith("move:") and name[5] in "UDFBLR"])
    inner = np.array([i for i, name in enumerate(vocabulary) if name.startswith("move:") and name[5] in "23MSE"])
    assert len(outer) == 18 and len(inner) == 45
    differences = np.array([vectors[variant] - vectors[example.original] for example in examples for variant in example.variants])
    capacities = np.array([1 / (len(examples) * len(example.variants)) for example in examples for _ in example.variants])

    # t represents the largest permitted outer-move cost. Every inner-move
    # cost must be at least t + minimum_gap.
    constraints = np.zeros((len(outer) + len(inner), dimension + 1))
    constraints[np.arange(len(outer)), outer] = -1
    constraints[:len(outer), -1] = 1
    constraints[len(outer) + np.arange(len(inner)), inner] = 1
    constraints[len(outer):, -1] = -1
    offset = np.r_[np.zeros(len(outer)), -np.full(len(inner), minimum_gap)]

    initial = np.zeros(dimension + 1)
    if dimension == len(outer) + len(inner):
        initial[inner] = 1 / len(inner)
    else:
        initial[inner] = minimum_gap
        pair_indices = [i for i, name in enumerate(vocabulary) if name.startswith("pair:")]
        initial[pair_indices] = (1 - len(inner) * minimum_gap) / len(pair_indices)

    def objective(values):
        weight = values[:-1]
        residual = MARGIN - differences @ weight
        active = residual > 0
        value = float(capacities @ np.maximum(residual, 0) + L2 * weight @ weight / 2)
        gradient = np.r_[-differences[active].T @ capacities[active] + L2 * weight, 0]
        return value, gradient

    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0, None)] * (dimension + 1),
        constraints=[
            {"type": "eq", "fun": lambda x: np.sum(x[:-1]) - 1, "jac": lambda x: np.r_[np.ones(dimension), 0]},
            {"type": "ineq", "fun": lambda x: constraints @ x + offset, "jac": lambda x: constraints},
        ],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success or min(constraints @ result.x + offset) < -1e-7 or abs(sum(result.x[:-1]) - 1) > 1e-7:
        raise RuntimeError("Ordered fit failed: " + result.message)
    weight = result.x[:-1]
    return weight, {
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "outer_maximum": float(max(weight[outer])),
        "inner_minimum": float(min(weight[inner])),
        "ordering_gap": float(min(weight[inner]) - max(weight[outer])),
    }


def run_model(examples, rows, use_pairs, leave_one_out):
    vocabulary, vectors = feature_matrix(examples, use_pairs)
    weight, optimization = fit_ordered(examples, vectors, vocabulary)
    result = {
        "margin": MARGIN,
        "lambda": L2,
        "minimum_ordering_gap": MINIMUM_GAP,
        "feature_count": len(vocabulary),
        "nonzero_coefficient_count": int(np.count_nonzero(weight > 1e-8)),
        "coefficients": [{"feature": name, "weight": round(float(value), 9)} for name, value in zip(vocabulary, weight)],
        "optimization": optimization,
        "training": [assess(example, weight, vectors, MARGIN) for example in examples],
    }
    result["best_variants"] = best_variants(examples, rows, result, use_pairs)
    if leave_one_out:
        result["leave_one_algorithm_out"] = []
        for excluded in examples:
            included = [example for example in examples if example.id != excluded.id]
            held_out_weight, _ = fit_ordered(included, vectors, vocabulary)
            result["leave_one_algorithm_out"].append(assess(excluded, held_out_weight, vectors, MARGIN))
    return result


def main():
    rows = read_candidates()
    examples = generate_examples([{**row, "puzzle": "7x7"} for row in rows])
    models = {}
    # In the pair model, pair coefficients can override any ordering imposed
    # on isolated moves. This pilot tests the ordering in the single-move model.
    for name, use_pairs, leave_one_out in (("single_moves", False, True),):
        model = run_model(examples, rows, use_pairs, leave_one_out)
        models[name] = model
        print(name, model["optimization"], flush=True)
        print("training wins", sum(x["wins"] for x in model["training"]), "of", sum(x["variants"] for x in model["training"]), flush=True)
        if leave_one_out:
            print("LOO wins", sum(x["wins"] for x in model["leave_one_algorithm_out"]), flush=True)
        print("reversed", [(x["id"], x["cost_difference"]) for x in model["best_variants"] if x["strictly_lower_variants"]], flush=True)
    output = {"source_count": len(examples), "comparison_count": sum(len(x.variants) for x in examples), "models": models}
    path = HERE / "outer_inner_constraint_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()

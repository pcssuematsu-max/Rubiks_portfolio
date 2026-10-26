"""Repeat the ergonomic ranking pilot with an L2 coefficient penalty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.ergonomics.run_pilot import (
    HERE,
    MARGIN,
    evaluate,
    feature_matrix,
    generate_examples,
    read_candidates,
)


LAMBDAS = (0.01, 0.1, 1.0)


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""
    ordered = np.sort(values)[::-1]
    partial = np.cumsum(ordered) - 1
    active = np.nonzero(ordered > partial / np.arange(1, len(values) + 1))[0]
    threshold = partial[active[-1]] / (active[-1] + 1)
    return np.maximum(values - threshold, 0)


def fit_l2(examples, vectors, lam: float, margin: float = MARGIN):
    dimension = len(next(iter(vectors.values())))
    differences = []
    capacities = []
    for example in examples:
        for variant in example.variants:
            differences.append(vectors[variant] - vectors[example.original])
            capacities.append(1 / (len(examples) * len(example.variants)))
    difference_matrix = np.asarray(differences)
    capacities = np.asarray(capacities)

    def negative_dual_and_gradient(dual):
        linear = difference_matrix.T @ dual
        weight = project_simplex(linear / lam)
        negative_dual = linear @ weight - lam * (weight @ weight) / 2 - margin * dual.sum()
        gradient = difference_matrix @ weight - margin
        return negative_dual, gradient

    solution = minimize(
        negative_dual_and_gradient,
        np.zeros(len(capacities)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, float(capacity)) for capacity in capacities],
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    def optimality(dual_variables):
        linear = difference_matrix.T @ dual_variables
        weight = project_simplex(linear / lam)
        gaps = difference_matrix @ weight
        primal = float(np.dot(capacities, np.maximum(margin - gaps, 0)) + lam * np.dot(weight, weight) / 2)
        dual = float(margin * dual_variables.sum() + lam * np.dot(weight, weight) / 2 - linear @ weight)
        return weight, primal, primal - dual

    dual_variables = solution.x
    weight, primal, gap = optimality(dual_variables)
    if gap > 2e-5:
        # The old SciPy L-BFGS-B sometimes stops at a dual kink. Continue with
        # projected accelerated gradient, checking the primal-dual gap.
        direction = np.ones(dimension) / np.sqrt(dimension)
        for _ in range(60):
            next_direction = difference_matrix.T @ (difference_matrix @ direction)
            direction = next_direction / np.linalg.norm(next_direction)
        spectral_squared = float(np.dot(difference_matrix @ direction, difference_matrix @ direction))
        step = 0.95 * lam / spectral_squared
        extrapolated = dual_variables.copy()
        momentum = 1.0
        for iteration in range(1, 30001):
            _, gradient = negative_dual_and_gradient(extrapolated)
            next_dual = np.clip(extrapolated - step * gradient, 0, capacities)
            next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
            extrapolated = next_dual + (momentum - 1) / next_momentum * (next_dual - dual_variables)
            dual_variables = next_dual
            momentum = next_momentum
            if iteration % 100 == 0:
                weight, primal, gap = optimality(dual_variables)
                if gap <= 2e-5:
                    break
    if gap > 2e-5:
        raise RuntimeError(f"L2 optimization did not converge: lambda={lam}, gap={gap}, {solution.message}")
    return weight, {"objective": round(primal, 9), "duality_gap": round(gap, 9), "iterations": int(solution.nit)}


def model_result(examples, use_pairs: bool, lam: float):
    vocabulary, vectors = feature_matrix(examples, use_pairs)
    weight, convergence = fit_l2(examples, vectors, lam)
    training = [evaluate(example, weight, vectors) for example in examples]
    held_out = []
    max_loo_gap = 0.0
    for excluded in examples:
        included = [example for example in examples if example.id != excluded.id]
        loo_weight, loo_convergence = fit_l2(included, vectors, lam)
        max_loo_gap = max(max_loo_gap, loo_convergence["duality_gap"])
        held_out.append(evaluate(excluded, loo_weight, vectors))
    weights = sorted(zip(vocabulary, weight), key=lambda item: -item[1])
    return {
        "lambda": lam,
        "feature_count": len(vocabulary),
        "nonzero_coefficient_count": int(np.count_nonzero(weight > 1e-8)),
        "coefficients": [
            {"feature": name, "weight": round(float(value), 9)}
            for name, value in weights
        ],
        "top_coefficients": [
            {"feature": name, "weight": round(float(value), 6)}
            for name, value in weights[:20]
        ],
        "training": training,
        "leave_one_algorithm_out": held_out,
        "optimization": convergence,
        "max_leave_one_out_duality_gap": round(max_loo_gap, 9),
    }


def main():
    examples = generate_examples(read_candidates())
    models = {}
    for name, use_pairs in (("single_moves", False), ("single_moves_and_adjacent_pairs", True)):
        models[name] = []
        for lam in LAMBDAS:
            result = model_result(examples, use_pairs, lam)
            models[name].append(result)
            held_out = result["leave_one_algorithm_out"]
            good = sum(row["strictly_preferred"] for row in held_out)
            total = sum(row["variants"] for row in held_out)
            print(name, lam, f"{good}/{total}", "nonzero", result["nonzero_coefficient_count"])
    output = {
        "method": "minimize mean hinge slack per source algorithm plus lambda/2 times squared L2 norm, with nonnegative coefficients summing to 1",
        "margin": MARGIN,
        "candidate_count": len(examples),
        "models": models,
    }
    path = Path(HERE) / "l2_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

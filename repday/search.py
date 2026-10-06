from __future__ import annotations

import random
from math import comb
from typing import List, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from .metrics import duration_curve_metrics


def run_full_opt(optimizer):
    return optimizer.solve()


def _draw_unique_combinations(
    rng: random.Random,
    candidates: List[int],
    selection_count: int,
    number_of_combinations: int,
) -> list[tuple[int, ...]]:
    """Draw unique unordered candidate sets without materializing all sets."""
    maximum_combinations = comb(len(candidates), selection_count)
    if number_of_combinations > maximum_combinations:
        raise ValueError(
            f"Requested {number_of_combinations} random iterations, but only "
            f"{maximum_combinations} unique day combinations are possible when "
            f"choosing {selection_count} from {len(candidates)} candidates."
        )

    combinations: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(combinations) < number_of_combinations:
        combination = tuple(sorted(rng.sample(candidates, selection_count)))
        if combination in seen:
            continue
        seen.add(combination)
        combinations.append(combination)
    return combinations


def run_hybrid_random_weighting(
    prepared,
    attributes,
    solver_config,
    n_representative_days: int,
    n_bins: int = 40,
    forced_day_ids: Optional[List[int]] = None,
    candidate_day_ids: Optional[List[int]] = None,
    n_random_iterations: int = 50,
    random_seed: int = 42,
    sampled_candidate_pool_size: Optional[int] = None,
    use_integer_weights: bool = False,
    enforce_positive_weight_for_selected_days: bool = True,
    min_weight_if_selected: float = 1.0,
):
    """
    Paper-like hybrid method:
    - randomly sample exactly n_representative_days
    - keep these days fixed
    - optimize only weights
    - repeat and keep the best solution
    """
    active_attributes = [a for a in attributes if a.active]
    forced_day_ids = sorted(set(forced_day_ids or []))

    all_candidates = (
        list(candidate_day_ids)
        if candidate_day_ids is not None
        else list(prepared.day_labels)
    )

    if len(forced_day_ids) > n_representative_days:
        raise ValueError("forced_day_ids cannot exceed n_representative_days")
    if n_random_iterations < 1:
        raise ValueError("n_random_iterations must be at least 1")

    remaining = [d for d in all_candidates if d not in forced_day_ids]
    rng = random.Random(random_seed)
    if sampled_candidate_pool_size is not None:
        required_random_days = n_representative_days - len(forced_day_ids)
        if sampled_candidate_pool_size < required_random_days:
            raise ValueError(
                "sampled_candidate_pool_size must be at least the number of "
                "non-forced representative days."
            )
        if sampled_candidate_pool_size < len(remaining):
            remaining = sorted(
                rng.sample(remaining, sampled_candidate_pool_size)
            )
    all_candidates = sorted(set(forced_day_ids + remaining))
    if len(remaining) + len(forced_day_ids) < n_representative_days:
        raise ValueError(
            "Not enough candidate days to create the requested number of "
            "representative days."
        )

    # Build L and A once, using full candidate information
    # Reuse the optimizer helper logic by instantiating a temporary optimizer
    from .model import RepresentativeDayOptimizer

    temp_optimizer = RepresentativeDayOptimizer(
        prepared=prepared,
        attributes=attributes,
        n_representative_days=n_representative_days,
        solver_config=solver_config,
        hours_per_day=24,
        use_integer_weights=False,
        n_bins=n_bins,
        forced_day_ids=forced_day_ids,
        candidate_day_ids=all_candidates,
    )

    day_ids_all = (
        list(temp_optimizer.candidate_day_ids)
        if temp_optimizer.candidate_day_ids is not None
        else list(prepared.day_labels)
    )
    L, A, bin_lower_bounds = temp_optimizer._build_L_A(day_ids_all)
    n_total = len(prepared.day_labels)
    attr_weights = {a.name: float(a.weight) for a in active_attributes}
    day_to_position = {
        day_id: position for position, day_id in enumerate(prepared.day_labels)
    }
    clustering_features = np.concatenate(
        [
            prepared.daily_profiles[attr.name]
            * np.sqrt(max(float(attr.weight), 0.0))
            for attr in active_attributes
            if float(attr.weight) > 0
        ],
        axis=1,
    )
    raw_daily_profiles = {
        attr.name: prepared.hourly.pivot(
            index="day_id",
            columns="hour_in_day",
            values=attr.column,
        ).sort_index().to_numpy(dtype=float)
        for attr in active_attributes
    }

    best_result = None
    best_obj = float("inf")
    best_iteration = None
    iteration_rows = []

    n_to_sample = n_representative_days - len(forced_day_ids)
    if n_to_sample < 0:
        raise ValueError("n_representative_days smaller than forced_day_ids count")
    random_combinations = _draw_unique_combinations(
        rng=rng,
        candidates=remaining,
        selection_count=n_to_sample,
        number_of_combinations=n_random_iterations,
    )

    for iteration, random_combination in enumerate(random_combinations, start=1):
        sampled = sorted(forced_day_ids + list(random_combination))

        # Build LP with sampled days fixed
        m = pyo.ConcreteModel(name="HybridRandomWeighting")
        m.D = pyo.Set(initialize=sampled, ordered=True)
        m.C = pyo.Set(initialize=[a.name for a in active_attributes], ordered=True)
        m.B = pyo.RangeSet(0, n_bins - 1)

        weight_domain = (
            pyo.NonNegativeIntegers
            if use_integer_weights
            else pyo.NonNegativeReals
        )
        m.w = pyo.Var(
            m.D, within=weight_domain, bounds=(0.0, float(n_total))
        )
        m.err = pyo.Var(m.C, m.B, within=pyo.NonNegativeReals)

        m.total_weight = pyo.Constraint(expr=sum(m.w[d] for d in m.D) == n_total)

        if enforce_positive_weight_for_selected_days:
            m.min_weight_link = pyo.Constraint( m.D,
                rule=lambda model, d: model.w[d] >= min_weight_if_selected
            )

        def err_upper_pos_rule(model, c, b):
            rhs = sum((model.w[d] / n_total) * A[(c, b, d)] for d in model.D)
            return L[(c, b)] - rhs <= model.err[c, b]

        def err_upper_neg_rule(model, c, b):
            rhs = sum((model.w[d] / n_total) * A[(c, b, d)] for d in model.D)
            return rhs - L[(c, b)] <= model.err[c, b]

        m.err_upper_pos = pyo.Constraint(m.C, m.B, rule=err_upper_pos_rule)
        m.err_upper_neg = pyo.Constraint(m.C, m.B, rule=err_upper_neg_rule)

        m.obj = pyo.Objective(
            expr=sum(attr_weights[c] * m.err[c, b] for c in m.C for b in m.B),
            sense=pyo.minimize,
        )

        solver = pyo.SolverFactory(solver_config.solver_name)
        if solver is None or not solver.available(exception_flag=False):
            raise RuntimeError(f"Solver '{solver_config.solver_name}' is not available.")

        if solver_config.timelimit_seconds is not None:
            try:
                solver.options["timelimit"] = solver_config.timelimit_seconds
            except Exception:
                pass
        if solver_config.mipgap is not None:
            try:
                solver.options["mipgap"] = solver_config.mipgap
            except Exception:
                pass
        if solver_config.threads is not None:
            try:
                solver.options["threads"] = solver_config.threads
            except Exception:
                pass

        result = solver.solve(m, tee=solver_config.tee)

        term = result.solver.termination_condition
        status = result.solver.status
        acceptable = (
            status in {SolverStatus.ok, SolverStatus.warning}
            and term in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
            }
        )
        if not acceptable:
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "metric_scope": "solver",
                    "attribute": "",
                    "solver_status": str(status),
                    "termination_condition": str(term),
                    "acceptable_solution": False,
                    "is_best_iteration": False,
                    "objective_value": None,
                    "selected_day_ids": ", ".join(map(str, sampled)),
                    "random_day_ids": ", ".join(map(str, random_combination)),
                    "forced_day_ids": ", ".join(map(str, forced_day_ids)),
                    "optimized_day_weights": "",
                }
            )
            continue

        obj = float(pyo.value(m.obj))
        iteration_day_weights = {
            d: float(pyo.value(m.w[d]))
            for d in sampled
            if pyo.value(m.w[d]) is not None and pyo.value(m.w[d]) > 1e-8
        }
        representative_positions = [day_to_position[d] for d in sampled]
        representative_features = clustering_features[representative_positions]
        nearest_assignments = np.argmin(
            np.sum(
                (
                    clustering_features[:, np.newaxis, :]
                    - representative_features[np.newaxis, :, :]
                )
                ** 2,
                axis=2,
            ),
            axis=1,
        )
        for cluster, position in enumerate(representative_positions):
            nearest_assignments[position] = cluster

        metric_values = []
        for attr in active_attributes:
            original_curve = prepared.original_duration_curves[attr.name]
            approximated_curve = temp_optimizer._weighted_duration_curve_from_selected_days(
                daily_profiles=prepared.daily_profiles[attr.name],
                day_ids=day_ids_all,
                day_weights=iteration_day_weights,
                target_length=len(original_curve),
            )
            duration_metrics = duration_curve_metrics(
                original_curve, approximated_curve
            )
            raw_profiles = raw_daily_profiles[attr.name]
            reconstructed_profiles = raw_profiles[representative_positions][
                nearest_assignments
            ]
            reconstruction_errors = raw_profiles - reconstructed_profiles
            metrics = {
                "chronological_rmse": float(
                    np.sqrt(np.mean(reconstruction_errors**2))
                ),
                "chronological_mae": float(
                    np.mean(np.abs(reconstruction_errors))
                ),
                "duration_curve_rmse": duration_metrics["rmse"],
                "duration_curve_mae": duration_metrics["mae"],
                "duration_curve_nrmse": duration_metrics["nrmse"],
                "peak_error": duration_metrics["peak_error"],
                "annual_energy_error": duration_metrics[
                    "annual_energy_error"
                ],
            }
            bin_absolute_error_sum = float(
                sum(pyo.value(m.err[attr.name, b]) for b in m.B)
            )
            objective_contribution = (
                float(attr.weight) * bin_absolute_error_sum
            )
            metric_values.append((attr, metrics))
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "metric_scope": "attribute",
                    "attribute": attr.name,
                    "attribute_weight": float(attr.weight),
                    "solver_status": str(status),
                    "termination_condition": str(term),
                    "acceptable_solution": True,
                    "is_best_iteration": False,
                    "selection_metric": "objective_value",
                    "objective_value": obj,
                    "bin_absolute_error_sum": bin_absolute_error_sum,
                    "objective_contribution": objective_contribution,
                    "selected_day_ids": ", ".join(map(str, sampled)),
                    "random_day_ids": ", ".join(map(str, random_combination)),
                    "forced_day_ids": ", ".join(map(str, forced_day_ids)),
                    "optimized_day_weights": "; ".join(
                        f"{day_id}={weight:.12g}"
                        for day_id, weight in sorted(iteration_day_weights.items())
                    ),
                    **metrics,
                }
            )

        total_attribute_weight = sum(float(attr.weight) for attr, _ in metric_values)
        if metric_values and total_attribute_weight > 0:
            aggregate_metrics = {
                metric_name: sum(
                    float(attr.weight) * metrics[metric_name]
                    for attr, metrics in metric_values
                )
                / total_attribute_weight
                for metric_name in next(iter(metric_values))[1]
            }
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "metric_scope": "weighted_mean",
                    "attribute": "<all active attributes>",
                    "attribute_weight": total_attribute_weight,
                    "solver_status": str(status),
                    "termination_condition": str(term),
                    "acceptable_solution": True,
                    "is_best_iteration": False,
                    "selection_metric": "objective_value",
                    "objective_value": obj,
                    "bin_absolute_error_sum": sum(
                        float(
                            sum(pyo.value(m.err[attr.name, b]) for b in m.B)
                        )
                        for attr in active_attributes
                    ),
                    "objective_contribution": obj,
                    "selected_day_ids": ", ".join(map(str, sampled)),
                    "random_day_ids": ", ".join(map(str, random_combination)),
                    "forced_day_ids": ", ".join(map(str, forced_day_ids)),
                    "optimized_day_weights": "; ".join(
                        f"{day_id}={weight:.12g}"
                        for day_id, weight in sorted(iteration_day_weights.items())
                    ),
                    **aggregate_metrics,
                }
            )

        if obj < best_obj:
            # Build a result object similar to optimizer.solve()
            day_weights = iteration_day_weights

            rows = []
            for d in day_ids_all:
                rows.append(
                    {
                        "day_id": d,
                        "selected": 1 if d in sampled else 0,
                        "weight": float(day_weights.get(d, 0.0)),
                    }
                )

            summary = pd.DataFrame(rows).sort_values(
                ["selected", "weight", "day_id"],
                ascending=[False, False, True],
            )

            approx_duration_curves = {}
            target_exceedance_shares = {}
            approx_exceedance_shares = {}

            for attr in active_attributes:
                original_len = len(prepared.original_duration_curves[attr.name])
                approx_duration_curves[attr.name] = temp_optimizer._weighted_duration_curve_from_selected_days(
                    daily_profiles=prepared.daily_profiles[attr.name],
                    day_ids=day_ids_all,
                    day_weights=day_weights,
                    target_length=original_len,
                )
                target_exceedance_shares[attr.name] = np.array(
                    [L[(attr.name, b)] for b in range(n_bins)],
                    dtype=float,
                )
                approx_exceedance_shares[attr.name] = np.array(
                    [
                        sum((day_weights.get(d, 0.0) / n_total) * A[(attr.name, b, d)] for d in day_ids_all)
                        for b in range(n_bins)
                    ],
                    dtype=float,
                )

            # L_table and A_table
            l_rows = []
            for attr in active_attributes:
                for b in range(n_bins):
                    l_rows.append(
                        {
                            "attribute": attr.name,
                            "bin_id": b + 1,
                            "bin_lower_bound": float(bin_lower_bounds[attr.name][b]),
                            "L_share": float(L[(attr.name, b)]),
                        }
                    )
            L_table = pd.DataFrame(l_rows)

            a_rows = []
            for attr in active_attributes:
                for b in range(n_bins):
                    for d in day_ids_all:
                        a_rows.append(
                            {
                                "attribute": attr.name,
                                "bin_id": b + 1,
                                "day_id": d,
                                "bin_lower_bound": float(bin_lower_bounds[attr.name][b]),
                                "A_share": float(A[(attr.name, b, d)]),
                            }
                        )
            A_table = pd.DataFrame(a_rows)

            from .model import OptimizationResult
            best_result = OptimizationResult(
                selected_days=sampled,
                day_weights=day_weights,
                summary=summary,
                approx_duration_curves=approx_duration_curves,
                objective_value=obj,
                bin_lower_bounds=bin_lower_bounds,
                target_exceedance_shares=target_exceedance_shares,
                approx_exceedance_shares=approx_exceedance_shares,
                L_table=L_table,
                A_table=A_table,
            )
            best_obj = obj
            best_iteration = iteration

    if best_result is None:
        raise RuntimeError("hybrid_random_weighting did not produce any valid solution.")

    iteration_history = pd.DataFrame(iteration_rows)
    if not iteration_history.empty:
        iteration_history["is_best_iteration"] = (
            iteration_history["iteration"] == best_iteration
        )
        iteration_history = iteration_history.sort_values(
            ["iteration", "metric_scope", "attribute"],
            kind="stable",
        ).reset_index(drop=True)
    best_result.iteration_history = iteration_history

    return best_result

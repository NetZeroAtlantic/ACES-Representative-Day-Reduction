from __future__ import annotations

import random
from typing import List, Optional

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


def run_full_opt(optimizer):
    return optimizer.solve()


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

    best_result = None
    best_obj = float("inf")



    for _ in range(n_random_iterations):
        n_to_sample = n_representative_days - len(forced_day_ids)
        if n_to_sample < 0:
            raise ValueError("n_representative_days smaller than forced_day_ids count")

        sampled = forced_day_ids + rng.sample(remaining, n_to_sample)
        sampled = sorted(set(sampled))

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
            continue

        obj = float(pyo.value(m.obj))
        if obj < best_obj:
            # Build a result object similar to optimizer.solve()
            day_weights = {
                d: float(pyo.value(m.w[d]))
                for d in sampled
                if pyo.value(m.w[d]) is not None and pyo.value(m.w[d]) > 1e-8
            }

            rows = []
            for d in day_ids_all:
                rows.append(
                    {
                        "day_id": d,
                        "selected": 1 if d in sampled else 0,
                        "weight": float(day_weights.get(d, 0.0)),
                    }
                )

            import pandas as pd
            import numpy as np

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

    if best_result is None:
        raise RuntimeError("hybrid_random_weighting did not produce any valid solution.")

    return best_result

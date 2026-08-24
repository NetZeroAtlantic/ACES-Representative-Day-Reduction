from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from .config import AttributeConfig, SolverConfig
from .preprocessing import PreparedData


@dataclass
class OptimizationResult:
    selected_days: List[int]
    day_weights: Dict[int, float]
    summary: pd.DataFrame
    approx_duration_curves: Dict[str, np.ndarray]
    objective_value: float
    bin_lower_bounds: Dict[str, np.ndarray]
    target_exceedance_shares: Dict[str, np.ndarray]
    approx_exceedance_shares: Dict[str, np.ndarray]
    L_table: pd.DataFrame
    A_table: pd.DataFrame


class RepresentativeDayOptimizer:
    """
    Poncelet-style representative day MILP.

    Core idea:
    - For each attribute c and bin b, compare:
        L[c,b] = share of time in the full-year series above the lower bound of bin b
      against
        sum_d (w[d] / N_total) * A[c,b,d]
      where A[c,b,d] is the share of time in day d above the same lower bound.

    This follows the basic OPT formulation in Poncelet et al. (2016):
      min sum_{c,b} error[c,b]
      s.t. error[c,b] = |L[c,b] - sum_d w[d]/N_total * A[c,b,d]|
           sum_d u[d] = N_repr
           w[d] <= u[d] * N_total
           sum_d w[d] = N_total

    Notes:
    - The paper uses nonnegative continuous weights. That is the default here.
    - If you want integer weights, set use_integer_weights=True.
    - Forced days can be added through `forced_day_ids`.
    """

    def __init__(
        self,
        prepared: PreparedData,
        attributes: List[AttributeConfig],
        n_representative_days: int,
        solver_config: SolverConfig,
        hours_per_day: int = 24,
        use_integer_weights: bool = False,
        n_bins: int = 5,
        forced_day_ids: Optional[List[int]] = None,
        candidate_day_ids: Optional[List[int]] = None,
        formulation: str = "milp",
        fixed_day_ids: Optional[List[int]] = None,
        enforce_positive_weight_for_selected_days: bool = False,
        min_weight_if_selected: float = 0.0,
    ) -> None:
        self.prepared = prepared
        self.attributes = [a for a in attributes if a.active]
        self.n_representative_days = n_representative_days
        self.solver_config = solver_config
        self.hours_per_day = hours_per_day
        self.use_integer_weights = use_integer_weights
        self.n_bins = n_bins
        self.forced_day_ids = sorted(set(forced_day_ids or []))
        self.candidate_day_ids = sorted(set(candidate_day_ids)) if candidate_day_ids is not None else None
        self.formulation = formulation
        self.fixed_day_ids = sorted(set(fixed_day_ids or []))
        self.enforce_positive_weight_for_selected_days = enforce_positive_weight_for_selected_days
        self.min_weight_if_selected = float(min_weight_if_selected)

    def _build_bin_lower_bounds(self, hourly_values: np.ndarray) -> np.ndarray:
        """
        Equal-range bins, following the paper's assumption.

        If max == min, all lower bounds collapse to that value.
        Bin 1 corresponds to the highest range.
        Bin n_bins corresponds to the lowest range, so its lower bound is the minimum.
        """
        vmax = float(np.max(hourly_values))
        vmin = float(np.min(hourly_values))
        if np.isclose(vmax, vmin):
            return np.full(self.n_bins, vmin, dtype=float)

        step = (vmax - vmin) / self.n_bins
        # lower bounds from high bins to low bins
        return np.array([vmax - (b + 1) * step for b in range(self.n_bins)], dtype=float)

    def _build_L_A(
        self,
        day_ids: List[int],
    ) -> Tuple[
        Dict[Tuple[str, int], float],
        Dict[Tuple[str, int, int], float],
        Dict[str, np.ndarray],
    ]:
        """
        Build:
        - L[(attr, b)]   : annual exceedance share for bin b
        - A[(attr, b, d)]: daily exceedance share for day d and bin b
        - bin_lower_bounds[attr]
        """
        day_to_pos = {
            day_id: position
            for position, day_id in enumerate(self.prepared.day_labels)
        }
        L: Dict[Tuple[str, int], float] = {}
        A: Dict[Tuple[str, int, int], float] = {}
        bin_lower_bounds: Dict[str, np.ndarray] = {}

        for attr in self.attributes:
            hourly_col = f"{attr.name}__norm"
            full_year_values = self.prepared.hourly[hourly_col].to_numpy(dtype=float)
            lower_bounds = self._build_bin_lower_bounds(full_year_values)
            bin_lower_bounds[attr.name] = lower_bounds

            daily_profiles = self.prepared.daily_profiles[attr.name]

            for b, lb in enumerate(lower_bounds):
                # Share of full-year time above threshold
                L[(attr.name, b)] = float(np.mean(full_year_values >= lb))

                # Share of time in each day above same threshold
                for d in day_ids:
                    pos = day_to_pos[d]
                    day_values = daily_profiles[pos, :]
                    A[(attr.name, b, d)] = float(np.mean(day_values >= lb))

        return L, A, bin_lower_bounds

    def _weighted_duration_curve_from_selected_days(
        self,
        daily_profiles: np.ndarray,
        day_ids: List[int],
        day_weights: Dict[int, float],
        target_length: int,
    ) -> np.ndarray:
        """
        Build a weighted duration curve for plotting/validation.

        This is a post-processing step only.
        It uses weighted hourly samples from the selected days and returns
        a stepwise duration curve evaluated on `target_length` equally spaced
        exceedance points.

        Works for continuous or integer weights.
        """
        if not day_weights:
            return np.zeros(target_length, dtype=float)

        day_to_pos = {
            day_id: position
            for position, day_id in enumerate(self.prepared.day_labels)
        }

        values = []
        weights = []
        for d, w in day_weights.items():
            if w <= 0:
                continue
            pos = day_to_pos[d]
            day_vals = daily_profiles[pos, :].astype(float)
            values.extend(day_vals.tolist())
            # each hourly value in that day inherits the day's weight
            weights.extend([float(w)] * len(day_vals))

        values_arr = np.asarray(values, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)

        if values_arr.size == 0 or np.sum(weights_arr) <= 0:
            return np.zeros(target_length, dtype=float)

        order = np.argsort(values_arr)[::-1]
        v = values_arr[order]
        w = weights_arr[order]
        cum_share = np.cumsum(w) / np.sum(w)

        # Match the same number of points as the original annual duration curve
        shares = np.arange(1, target_length + 1) / target_length
        out = np.empty(target_length, dtype=float)

        j = 0
        for i, s in enumerate(shares):
            while j < len(cum_share) - 1 and cum_share[j] < s:
                j += 1
            out[i] = v[j]
        return out

    def solve(self) -> OptimizationResult:

        print("Enforce positive weight:", self.enforce_positive_weight_for_selected_days)
        print("Min weight:", self.min_weight_if_selected)
        day_ids = (
                list(self.candidate_day_ids)
                if self.candidate_day_ids is not None
                else list(self.prepared.day_labels)
                    )

        if self.formulation not in {"milp", "lp_fixed_days"}:
            raise ValueError("formulation must be 'milp' or 'lp_fixed_days'.")
        if self.formulation == "lp_fixed_days":
            if self.use_integer_weights:
                raise ValueError("lp_fixed_days requires use_integer_weights=False.")
            if not self.fixed_day_ids:
                raise ValueError("lp_fixed_days requires clustering.fixed_day_ids.")
            if len(self.fixed_day_ids) != self.n_representative_days:
                raise ValueError("fixed_day_ids must contain exactly n_representative_days IDs.")
            day_ids = self.fixed_day_ids

        missing_candidate_days = sorted(set(day_ids) - set(self.prepared.day_labels))
        if missing_candidate_days:
            raise ValueError(
                f"Some candidate_day_ids were not found in prepared.day_labels: {missing_candidate_days}"
    )

        n_days = len(day_ids)
        n_total = len(self.prepared.day_labels) # FULL YEAR

        if n_days < self.n_representative_days:
            raise ValueError(
                "Number of available days is smaller than n_representative_days."
            )

        if self.formulation == "milp" and len(self.forced_day_ids) > self.n_representative_days:
            raise ValueError(
                "Number of forced days cannot exceed n_representative_days."
            )

        missing_forced = sorted(set(self.forced_day_ids) - set(day_ids))
        if missing_forced:
            raise ValueError(f"Forced day_ids not found in available day_labels: {missing_forced}")

        L, A, bin_lower_bounds = self._build_L_A(day_ids)

        l_rows = []
        for attr in self.attributes:
            for b in range(self.n_bins):
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
        for attr in self.attributes:
            for b in range(self.n_bins):
                for d in day_ids:
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

        attr_weights = {a.name: float(a.weight) for a in self.attributes}

        m = pyo.ConcreteModel(name=f"RepresentativeDay{self.formulation}_Poncelet")
        m.D = pyo.Set(initialize=day_ids, ordered=True)
        m.C = pyo.Set(initialize=[a.name for a in self.attributes], ordered=True)
        m.B = pyo.RangeSet(0, self.n_bins - 1)

        if self.formulation == "milp":
            m.u = pyo.Var(m.D, within=pyo.Binary)
        weight_domain = pyo.NonNegativeIntegers if self.use_integer_weights else pyo.NonNegativeReals
        m.w = pyo.Var(m.D, within=weight_domain, bounds=(0.0, float(n_total)))
        m.err = pyo.Var(m.C, m.B, within=pyo.NonNegativeReals)

        if self.formulation == "milp":
            m.rep_count = pyo.Constraint(expr=sum(m.u[d] for d in m.D) == self.n_representative_days)
            m.activation = pyo.Constraint(m.D, rule=lambda model, d: model.w[d] <= model.u[d] * n_total)
            if self.enforce_positive_weight_for_selected_days:
                m.min_weight_link = pyo.Constraint(m.D, rule=lambda model, d: model.w[d] >= self.min_weight_if_selected * model.u[d])
        elif self.enforce_positive_weight_for_selected_days:
            m.fixed_day_min_weight = pyo.Constraint(m.D, rule=lambda model, d: model.w[d] >= self.min_weight_if_selected)

        # Eq. (9)
        m.total_weight = pyo.Constraint(
            expr=sum(m.w[d] for d in m.D) == n_total
        )

        # Optional forced days: user asked to add F later
        if self.formulation == "milp" and self.forced_day_ids:
            def forced_rule(model, d):
                return model.u[d] == 1
            m.F = pyo.Set(initialize=self.forced_day_ids, ordered=True)
            m.forced_days = pyo.Constraint(m.F, rule=forced_rule)

        # Linearization of:
        # err[c,b] = | L[c,b] - sum_d w[d]/N_total * A[c,b,d] |
        def err_upper_pos_rule(model, c, b):
            rhs = sum((model.w[d] / n_total) * A[(c, b, d)] for d in model.D)
            return L[(c, b)] - rhs <= model.err[c, b]

        def err_upper_neg_rule(model, c, b):
            rhs = sum((model.w[d] / n_total) * A[(c, b, d)] for d in model.D)
            return rhs - L[(c, b)] <= model.err[c, b]

        m.err_upper_pos = pyo.Constraint(m.C, m.B, rule=err_upper_pos_rule)
        m.err_upper_neg = pyo.Constraint(m.C, m.B, rule=err_upper_neg_rule)

        # Eq. (5)
        m.obj = pyo.Objective(
            expr=sum(attr_weights[c] * m.err[c, b] for c in m.C for b in m.B),
            sense=pyo.minimize,
        )

        solver = pyo.SolverFactory(self.solver_config.solver_name)
        if solver is None or not solver.available(exception_flag=False):
            raise RuntimeError(
                f"Solver '{self.solver_config.solver_name}' is not available. "
                "Install the solver or change solver_name."
            )

        if self.solver_config.timelimit_seconds is not None:
            try:
                solver.options["timelimit"] = self.solver_config.timelimit_seconds
            except Exception:
                pass

        if self.solver_config.mipgap is not None:
            try:
                solver.options["mipgap"] = self.solver_config.mipgap
            except Exception:
                pass

        if self.solver_config.threads is not None:
            try:
                solver.options["threads"] = self.solver_config.threads
            except Exception:
                pass

        result = solver.solve(m, tee=self.solver_config.tee)

        term = result.solver.termination_condition
        status = result.solver.status
        has_var_values = any(
            v.value is not None for v in m.component_data_objects(pyo.Var, active=True)
        )

        acceptable = (
            status in {SolverStatus.ok, SolverStatus.warning}
            and term in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
            }
            and has_var_values
        )

        if not acceptable:
            raise RuntimeError(
                f"Solver finished without a usable solution. "
                f"status={status}, termination={term}"
            )

        selected_days = [d for d in day_ids if (pyo.value(m.u[d]) > 0.5 if self.formulation == "milp" else pyo.value(m.w[d]) > 1e-8)]
        day_weights = {
            d: float(pyo.value(m.w[d]))
            for d in day_ids
            if pyo.value(m.w[d]) is not None and pyo.value(m.w[d]) > 1e-8
        }

        rows = []
        for d in day_ids:
            rows.append(
                {
                    "day_id": d,
                    "selected": int(round(float(pyo.value(m.u[d]) or 0.0))) if self.formulation == "milp" else int(float(pyo.value(m.w[d]) or 0.0) > 1e-8),
                    "weight": float(pyo.value(m.w[d]) or 0.0),
                }
            )
        summary = pd.DataFrame(rows).sort_values(
            ["selected", "weight", "day_id"],
            ascending=[False, False, True],
        )

        approx_duration_curves: Dict[str, np.ndarray] = {}
        target_exceedance_shares: Dict[str, np.ndarray] = {}
        approx_exceedance_shares: Dict[str, np.ndarray] = {}

        for attr in self.attributes:
            original_len = len(self.prepared.original_duration_curves[attr.name])
            approx_duration_curves[attr.name] = self._weighted_duration_curve_from_selected_days(
                daily_profiles=self.prepared.daily_profiles[attr.name],
                day_ids=day_ids,
                day_weights=day_weights,
                target_length=original_len,
            )
            target_exceedance_shares[attr.name] = np.array(
                [L[(attr.name, b)] for b in range(self.n_bins)],
                dtype=float,
            )
            approx_exceedance_shares[attr.name] = np.array(
                [
                    sum((day_weights.get(d, 0.0) / n_total) * A[(attr.name, b, d)] for d in day_ids)
                    for b in range(self.n_bins)
                ],
                dtype=float,
            )

        return OptimizationResult(
            selected_days=selected_days,
            day_weights=day_weights,
            summary=summary,
            approx_duration_curves=approx_duration_curves,
            objective_value=float(pyo.value(m.obj)),
            bin_lower_bounds=bin_lower_bounds,
            target_exceedance_shares=target_exceedance_shares,
            approx_exceedance_shares=approx_exceedance_shares,
            L_table=L_table,
            A_table=A_table,
        )

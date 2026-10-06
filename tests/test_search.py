from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition

from repday.config import AttributeConfig, SolverConfig
from repday.preprocessing import prepare_daily_structures
from repday.search import run_hybrid_random_weighting


class FakeSolver:
    def __init__(self) -> None:
        self.options = {}

    def available(self, exception_flag=False):
        return True

    def solve(self, model, tee=False):
        equal_weight = 6.0 / len(model.D)
        for day_id in model.D:
            model.w[day_id].set_value(equal_weight)
        error_value = float(min(model.D)) / 1000.0
        for key in model.err:
            model.err[key].set_value(error_value)
        return SimpleNamespace(
            solver=SimpleNamespace(
                status=SolverStatus.ok,
                termination_condition=TerminationCondition.optimal,
            )
        )


def test_hybrid_iterations_use_unique_combinations_and_record_metrics():
    hourly = pd.DataFrame(
        [
            {
                "day_id": day_id,
                "hour_in_day": hour,
                "load": float(day_id * 10 + hour),
            }
            for day_id in range(1, 7)
            for hour in range(24)
        ]
    )
    attribute = AttributeConfig(
        name="load",
        column="load",
        weight=1.0,
        normalize=False,
    )
    prepared = prepare_daily_structures(hourly, [attribute])

    with patch("repday.search.pyo.SolverFactory", return_value=FakeSolver()):
        result = run_hybrid_random_weighting(
            prepared=prepared,
            attributes=[attribute],
            solver_config=SolverConfig(solver_name="fake"),
            n_representative_days=3,
            n_bins=4,
            forced_day_ids=[1],
            n_random_iterations=5,
            random_seed=9,
            use_integer_weights=False,
        )

    history = result.iteration_history
    aggregate = history[history["metric_scope"] == "weighted_mean"]
    assert len(aggregate) == 5
    assert aggregate["selected_day_ids"].nunique() == 5
    assert aggregate["is_best_iteration"].sum() == 1
    assert set(
        [
            "objective_value",
            "bin_absolute_error_sum",
            "objective_contribution",
            "chronological_rmse",
            "chronological_mae",
            "duration_curve_rmse",
            "duration_curve_mae",
            "duration_curve_nrmse",
            "peak_error",
            "annual_energy_error",
        ]
    ).issubset(history.columns)

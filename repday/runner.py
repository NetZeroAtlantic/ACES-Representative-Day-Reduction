from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import RunConfig
from .data import load_hourly_profiles
from .preprocessing import prepare_daily_structures
from .model import RepresentativeDayOptimizer
from .metrics import build_metrics_table
from .plots import plot_duration_curves
from .candidate_reduction import reduce_candidate_days
#from .search import run_full_opt
from .search import run_full_opt, run_hybrid_random_weighting

class RepresentativeDayPipeline:
    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def _compute_extreme_days(self, prepared):
        df = prepared.hourly.copy()
        attributes = self.config.attributes
        cfg = self.config.clustering

        daily = df.groupby("day_id")

        extreme_days = set()

        for attr in attributes:
            col = attr.column

            if col not in df.columns:
                continue

            # --------------------------
            # 1) MAX (e.g. peak demand)
            # --------------------------
            if attr.name == "demand":
                try:
                    max_day = int(daily[col].max().idxmax())
                    extreme_days.add(max_day)
                except:
                    pass

            # --------------------------
            # 2) MIN (e.g. low wind)
            # --------------------------
            if attr.name in ["wind", "solar"]:
                try:
                    min_day = int(daily[col].mean().idxmin())
                    extreme_days.add(min_day)
                except:
                    pass

            # --------------------------
            # 3) RAMP
            # --------------------------
            if cfg.include_max_daily_ramp_day:
                try:
                    ramp_per_day = (
                        df.sort_values(["day_id", "hour_in_day"])
                        .groupby("day_id")[col]
                        .apply(lambda x: (x.diff().abs().max()))
                    )

                    ramp_day = int(ramp_per_day.idxmax())
                    extreme_days.add(ramp_day)
                except:
                    pass

        return sorted(extreme_days)

    def _export_results(self, prepared, result, reduction_result, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # 1) Selected day summary
        # -----------------------------
        selected = result.summary[result.summary["selected"] == 1].copy()
        if reduction_result is not None:
            selected["is_extreme_day"] = selected["day_id"].isin(
                reduction_result.extreme_day_ids
            )
            selected["is_cluster_representative"] = selected["day_id"].isin(
                reduction_result.cluster_representative_day_ids
            )
        else:
            selected["is_extreme_day"] = False
            selected["is_cluster_representative"] = False

        if "timestamp" in prepared.hourly.columns:
            prepared.hourly["timestamp"] = pd.to_datetime(prepared.hourly["timestamp"])
            day_map = (
                prepared.hourly.groupby("day_id")["timestamp"]
                .min()
                .reset_index()
                .rename(columns={"timestamp": "day_start_timestamp"})
            )
            selected = selected.merge(day_map, on="day_id", how="left")

        selected.to_excel(
            output_dir / "selected_representative_days_summary.xlsx",
            index=False,
        )

        active_selected = selected[selected["weight"] > 1e-8].copy()
        active_selected.to_excel(
            output_dir / "active_representative_days_summary.xlsx",
            index=False,
        )

        # -----------------------------
        # 2) Selected representative days in hourly long format
        # -----------------------------
        selected_hourly = prepared.hourly[
            prepared.hourly["day_id"].isin(selected["day_id"])
        ].copy()

        selected_hourly = selected_hourly.merge(
            selected[["day_id", "weight"]],
            on="day_id",
            how="left",
        )

        selected_hourly.to_excel(
            output_dir / "selected_representative_days_hourly.xlsx",
            index=False,
        )

        # -----------------------------
        # 3) Wide format for direct database use
        #    one row per representative day
        # -----------------------------
        # Export original configured columns if they exist in hourly data
        value_columns = []
        for attr in self.config.attributes:
            if attr.active and attr.column in selected_hourly.columns:
                value_columns.append(attr.column)

        if value_columns:
            wide_parts = []
            for col in value_columns:
                wide = selected_hourly.pivot(
                    index="day_id",
                    columns="hour_in_day",
                    values=col,
                )
                wide.columns = [f"{col}_h{int(c) + 1}" for c in wide.columns]
                wide_parts.append(wide)

            repdays_wide = pd.concat(wide_parts, axis=1).reset_index()

            repdays_wide = repdays_wide.merge(
                selected[["day_id", "weight"]],
                on="day_id",
                how="left",
            )

            if "day_start_timestamp" in selected.columns:
                repdays_wide = repdays_wide.merge(
                    selected[["day_id", "day_start_timestamp"]],
                    on="day_id",
                    how="left",
                )

            repdays_wide.to_excel(
                output_dir / "representative_days_wide.xlsx",
                index=False,
            )

        # -----------------------------
        # 4) Export L and A tables if available from model.py
        # -----------------------------
        if hasattr(result, "L_table") and result.L_table is not None:
            result.L_table.to_excel(output_dir / "L_table.xlsx", index=False)

        if hasattr(result, "A_table") and result.A_table is not None:
            result.A_table.to_excel(output_dir / "A_table.xlsx", index=False)

        # -----------------------------
        # 5) Candidate reduction exports
        # -----------------------------
        if reduction_result is not None:
            candidate_df = pd.DataFrame(
                {"candidate_day_id": reduction_result.candidate_day_ids}
            )

            if "timestamp" in prepared.hourly.columns:
                day_map = (
                    prepared.hourly.groupby("day_id")["timestamp"]
                    .min()
                    .reset_index()
                    .rename(columns={"day_id": "candidate_day_id", "timestamp": "day_start_timestamp"})
                )
                candidate_df = candidate_df.merge(
                    day_map,
                    on="candidate_day_id",
                    how="left",
                )

            candidate_df["is_extreme_day"] = candidate_df["candidate_day_id"].isin(
                reduction_result.extreme_day_ids
            )
            candidate_df["is_cluster_representative"] = candidate_df["candidate_day_id"].isin(
                reduction_result.cluster_representative_day_ids
            )

            candidate_df.to_excel(
                output_dir / "candidate_days_summary.xlsx",
                index=False,
            )

            pd.DataFrame(
                {"extreme_day_id": reduction_result.extreme_day_ids}
            ).to_excel(
                output_dir / "extreme_days_selected.xlsx",
                index=False,
            )

            pd.DataFrame(
                {"cluster_representative_day_id": reduction_result.cluster_representative_day_ids}
            ).to_excel(
                output_dir / "cluster_representative_days.xlsx",
                index=False,
            )

            reduction_result.feature_matrix.to_excel(
                output_dir / "candidate_feature_matrix.xlsx",
                index=False,
            )

    def run(self) -> dict:
        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_json(str(output_dir / "run_config.json"))

        hourly = load_hourly_profiles(
            excel_path=self.config.excel_path,
            timestamp_column=self.config.clustering.timestamp_column,
            day_id_column=self.config.clustering.day_id_column,
            hours_per_day=self.config.clustering.hours_per_day,
        )

        prepared = prepare_daily_structures(
            hourly=hourly,
            attributes=self.config.attributes,
            hours_per_day=self.config.clustering.hours_per_day,
            require_full_days=self.config.clustering.require_full_days,
        )

        reduction_result = reduce_candidate_days(
            prepared=prepared,
            attributes=self.config.attributes,
            method=self.config.clustering.candidate_reduction_method,
            feature_mode=self.config.clustering.candidate_feature_mode,
            n_candidate_days=self.config.clustering.n_candidate_days,
            include_extreme_days=self.config.clustering.include_extreme_days,
            include_peak_demand_day=self.config.clustering.include_peak_demand_day,
            include_min_wind_day=self.config.clustering.include_min_wind_day,
            include_max_daily_ramp_day=self.config.clustering.include_max_daily_ramp_day,
            random_seed=self.config.clustering.random_seed,
        )

        # Forced day handling
        manual_forced_days = getattr(self.config.clustering, "forced_day_ids", None) or []

        if self.config.clustering.force_extreme_days_in_final_selection:
            if reduction_result is not None and reduction_result.extreme_day_ids:
                auto_forced_days = reduction_result.extreme_day_ids
            else:
                # fallback: compute extreme days directly
                auto_forced_days = self._compute_extreme_days(prepared)
        else:
            auto_forced_days = []

        forced_day_ids = sorted(set(manual_forced_days + auto_forced_days))

        if reduction_result is not None:
            candidate_day_ids = sorted(
                set(reduction_result.candidate_day_ids + forced_day_ids)
            )
        else:
            candidate_day_ids = sorted(
                set(list(prepared.day_labels) + forced_day_ids)
            )

        optimizer = RepresentativeDayOptimizer(
            prepared=prepared,
            attributes=self.config.attributes,
            n_representative_days=self.config.clustering.n_representative_days,
            solver_config=self.config.solver,
            hours_per_day=self.config.clustering.hours_per_day,
            use_integer_weights=self.config.clustering.use_integer_weights,
            n_bins=self.config.clustering.n_bins,
            forced_day_ids=forced_day_ids,
            candidate_day_ids=candidate_day_ids,
            formulation=self.config.clustering.formulation,
            fixed_day_ids=self.config.clustering.fixed_day_ids,
            enforce_positive_weight_for_selected_days=self.config.clustering.enforce_positive_weight_for_selected_days,
            min_weight_if_selected=self.config.clustering.min_weight_if_selected,


        )

        #result = optimizer.solve()
        #result = run_full_opt(optimizer)

        if self.config.clustering.solution_method == "opt":
            result = run_full_opt(optimizer)

        elif self.config.clustering.solution_method == "hybrid_random_weighting":
                print("RUNNER solution_method =", self.config.clustering.solution_method)
                print("RUNNER forced_day_ids =", forced_day_ids)
                print("RUNNER candidate_day_ids count =", len(candidate_day_ids))
                result = run_hybrid_random_weighting(
                    prepared=prepared,
                    attributes=self.config.attributes,
                    solver_config=self.config.solver,
                    n_representative_days=self.config.clustering.n_representative_days,
                    n_bins=self.config.clustering.n_bins,
                    forced_day_ids=forced_day_ids,
                    candidate_day_ids=candidate_day_ids,
                    n_random_iterations=self.config.clustering.n_random_iterations,
                    random_seed=self.config.clustering.random_seed,
                    sampled_candidate_pool_size=self.config.clustering.sampled_candidate_pool_size,
                    use_integer_weights=self.config.clustering.use_integer_weights,
                    enforce_positive_weight_for_selected_days=self.config.clustering.enforce_positive_weight_for_selected_days,
                    min_weight_if_selected=self.config.clustering.min_weight_if_selected,
                )

        else:
            raise ValueError(
                f"Unsupported solution_method: {self.config.clustering.solution_method}"
            )

        metrics = build_metrics_table(
            prepared.original_duration_curves,
            result.approx_duration_curves,
        )

        if self.config.output.save_csvs:
            prepared.hourly.to_csv(output_dir / "processed_hourly.csv", index=False)
            result.summary.to_csv(output_dir / "selected_days_and_weights.csv", index=False)
            metrics.to_csv(output_dir / "duration_curve_metrics.csv", index=False)

        # New Excel exports for direct use
        self._export_results(prepared, result, reduction_result, output_dir)

        if self.config.output.save_plots or self.config.output.show_plots:
            plot_duration_curves(
                original_curves=prepared.original_duration_curves,
                approx_curves=result.approx_duration_curves,
                output_dir=str(output_dir),
                show=self.config.output.show_plots,
                save=self.config.output.save_plots,
            )

        return {
            "prepared": prepared,
            "result": result,
            "metrics": metrics,
        }

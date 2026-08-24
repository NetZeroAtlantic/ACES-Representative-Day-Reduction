from __future__ import annotations
from pathlib import Path
import pandas as pd
from repday import (
    AttributeConfig,
    ClusteringConfig,
    SolverConfig,
    OutputConfig,
    RunConfig,
    RepresentativeDayPipeline,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    excel_path = root / "data" / "example_wind_demand_8760.xlsx"
    output_dir = root / "example_output"


    # 1) create combined file
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    wind_columns = [c for c in df.columns if c.startswith("E_WIND-ON-")]
    wind_series = df[wind_columns].mean(axis=1).rename("wind")
    df2 = pd.concat([df, wind_series], axis=1).copy()

    combined_excel_path = excel_path.parent / "wind_demand_combined.xlsx"
    df2.to_excel(combined_excel_path, index=False)

    config = RunConfig(
        excel_path=str(combined_excel_path),
        attributes=[
            AttributeConfig(name="wind", column="wind", weight=1.0, normalize=True, normalization="minmax"),
            AttributeConfig(name="demand", column="Demand", weight=5.0, normalize=True, normalization="minmax"),
        ],
        #attributes = attributes,

        clustering=ClusteringConfig(
            n_representative_days=12,
            hours_per_day=24,
            timestamp_column="timestamp",
            day_id_column=None,
            require_full_days=True,
            use_integer_weights=False,
            n_bins=1000,
            candidate_reduction_method="none",
            candidate_feature_mode="hybrid_profile_and_duration",
            n_candidate_days=None,
            include_extreme_days=True,
            include_peak_demand_day=True,
            include_min_wind_day=True,
            include_max_daily_ramp_day=True,
            random_seed=40,
            solution_method = "hybrid_random_weighting",
            n_random_iterations = 50000,
            sampled_candidate_pool_size = None,

            force_extreme_days_in_final_selection=True,
            enforce_positive_weight_for_selected_days=True,
            min_weight_if_selected=1.0,

        ),
        solver=SolverConfig(
            solver_name="cplex",
            tee=True,
            timelimit_seconds=3000,
            mipgap=0.01,
            threads=6,
        ),
        output=OutputConfig(
            output_dir=str(output_dir),
            save_plots=True,
            show_plots=True,
            save_csvs=True,
        ),
    )

    pipeline = RepresentativeDayPipeline(config)
    outputs = pipeline.run()
    print("Sum of weights =", sum(outputs["result"].day_weights.values()))

    print("\nSelected days and weights:")
    print(outputs["result"].summary.head(20))
    print("\nMetrics:")
    print(outputs["metrics"])


if __name__ == "__main__":
    main()

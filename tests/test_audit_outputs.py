import pandas as pd

from run_aces_tsam import AccuracySummary, write_audits


def test_write_audits_uses_separate_csv_files(tmp_path):
    accuracy = AccuracySummary(
        rmse=pd.Series({"profile_a": 0.1}),
        mae=pd.Series({"profile_a": 0.08}),
        rmse_duration=pd.Series({"profile_a": 0.05}),
    )
    iteration_history = pd.DataFrame(
        [{"iteration": 1, "objective_value": 0.25}]
    )

    write_audits(
        output_directory=tmp_path,
        seasons=["2021-01-01", "2021-01-02"],
        config={
            "clustering": {"n_representative_days": 1},
            "outputs": {"write_accuracy_metrics": True},
        },
        representative_source_days={0: "2021-01-01"},
        cluster_assignments=[0, 0],
        cluster_weights={0: 2.0},
        accuracy=accuracy,
        engine="pyomo",
        forced_day_ids=[],
        objective_value=0.25,
        extreme_records=[],
        selected_columns=["profile_a"],
        column_weights={"profile_a": 1.0},
        metadata={},
        iteration_history=iteration_history,
    )

    expected_csv_files = {
        "cluster_assignments.csv",
        "configuration.csv",
        "extreme_day_provenance.csv",
        "profile_weights.csv",
        "pyomo_accuracy.csv",
        "pyomo_forced_days.csv",
        "pyomo_iteration_history.csv",
        "pyomo_summary.csv",
        "representative_day_weights.csv",
        "run_summary.csv",
    }
    assert {path.name for path in tmp_path.glob("*.csv")} == expected_csv_files
    assert not list(tmp_path.glob("*.xlsx"))

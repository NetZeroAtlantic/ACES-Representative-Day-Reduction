from __future__ import annotations

import argparse
import copy
import inspect
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class ProfileMetadata:
    table: str
    logical_table: str
    key_columns: list[str]
    key_values: tuple[Any, ...]
    season_column: str
    hour_column: str
    value_column: str
    preserve_columns: list[str]
    preserve_values: tuple[Any, ...]
    output_rule: str


@dataclass
class AccuracySummary:
    rmse: pd.Series
    mae: pd.Series
    rmse_duration: pd.Series


@dataclass
class PyomoAcesResult:
    representatives: pd.DataFrame
    cluster_weights: dict[int, float]
    representative_source_days: dict[int, str]
    cluster_assignments: list[int]
    accuracy: AccuracySummary
    forced_day_ids: list[int]
    objective_value: float
    iteration_history: pd.DataFrame | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce an hourly ACES SQLite database using TSAM clustering or "
            "Pyomo-based duration-curve exceedance-error minimization."
        )
    )
    parser.add_argument(
        "config", nargs="?", default="aces_tsam_8760.yaml", help="YAML configuration"
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Print SQLite tables and columns"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and ACES temporal profiles without clustering",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    database = resolve_path(
        config_path.parent, config["project"]["input_database"]
    )
    config = configure_database_schema(database, config)
    print(f'Using ACES database schema: {config["aces"]["schema"]}')
    if args.inspect:
        print(inspect_database(database).to_string(index=False))
        return

    profiles, metadata, seasons, hours = load_aces_profiles(
        database, config, config_path.parent
    )
    print(
        f"Loaded {len(seasons)} days x {len(hours)} hours "
        f"with {len(metadata)} ACES profiles."
    )

    selected_columns, column_weights = select_clustering_columns(
        profiles.columns.tolist(), metadata, config["clustering"]["attributes"]
    )
    print(f"Clustering with {len(selected_columns)} selected profiles.")
    engine = str(config["clustering"]["engine"]).lower()
    validate_clustering_settings(
        database=database,
        profiles=profiles,
        selected_columns=selected_columns,
        metadata=metadata,
        clustering=config["clustering"],
        seasons=seasons,
        hours=hours,
        engine=engine,
        calendar=config["aces"]["calendar"],
    )
    if args.validate_only:
        print(
            f"Configuration, ACES temporal profiles, and {engine} settings "
            "are valid."
        )
        return

    extreme_records = build_extreme_audit_records(
        database=database,
        profiles=profiles,
        selected_columns=selected_columns,
        metadata=metadata,
        clustering=config["clustering"],
        seasons=seasons,
        hours=hours,
        engine=engine,
        calendar=config["aces"]["calendar"],
    )
    if engine == "tsam":
        custom_ramp_rules = {
            "include_max_daily_ramp_day",
            "include_max_net_load_ramp_day",
        }
        custom_forced_day_ids = sorted(
            {
                int(record["day_id"])
                for record in extreme_records
                if record.get("rule") in custom_ramp_rules
            }
        )
        tsam_profiles, tsam_weights, marker_columns = add_tsam_day_markers(
            profiles[selected_columns],
            column_weights,
            custom_forced_day_ids,
            len(hours),
            float(
                config["clustering"]
                .get("tsam", {})
                .get("extremes", {})
                .get("custom_marker_weight", 1e-6)
            ),
        )
        result = run_tsam(
            tsam_profiles,
            tsam_weights,
            config["clustering"],
            marker_columns,
        )
        cluster_centers = result.clustering.cluster_centers
        if cluster_centers is None:
            raise ValueError(
                "TSAM did not return historical center indices. Use medoid "
                "representation with the ACES writer."
            )
        representatives = build_pyomo_representatives(
            profiles,
            [int(center) + 1 for center in cluster_centers],
            len(hours),
        )
        cluster_weights = get_tsam_cluster_weights(result)
        representative_source_days = map_representative_source_days(
            representatives,
            cluster_centers,
            seasons,
        )
        cluster_assignments = list(result.clustering.cluster_assignments)
        accuracy = calculate_pyomo_accuracy(
            profiles=profiles,
            representatives=representatives,
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
        )
        forced_day_ids = sorted(
            {int(record["day_id"]) for record in extreme_records}
        )
        objective_value = None
        iteration_history = None
    elif engine in {"pyomo", "pyomo_milp", "pyomo_lp_fixed_days"}:
        pyomo_result = run_pyomo(
            database=database,
            profiles=profiles,
            selected_columns=selected_columns,
            column_weights=column_weights,
            metadata=metadata,
            clustering=config["clustering"],
            seasons=seasons,
            hours=hours,
            engine=engine,
            calendar=config["aces"]["calendar"],
            extreme_records=extreme_records,
        )
        representatives = pyomo_result.representatives
        cluster_weights = pyomo_result.cluster_weights
        representative_source_days = pyomo_result.representative_source_days
        cluster_assignments = pyomo_result.cluster_assignments
        accuracy = pyomo_result.accuracy
        forced_day_ids = pyomo_result.forced_day_ids
        objective_value = pyomo_result.objective_value
        iteration_history = pyomo_result.iteration_history
    else:
        raise ValueError(
            "clustering.engine must be tsam, pyomo, pyomo_milp, or "
            "pyomo_lp_fixed_days."
        )

    output_database = write_reduced_database(
        source_database=database,
        config=config,
        base_directory=config_path.parent,
        representatives=representatives,
        cluster_weights=cluster_weights,
        representative_source_days=representative_source_days,
        metadata=metadata,
        seasons=seasons,
        hours=hours,
        cluster_assignments=cluster_assignments,
    )
    write_audits(
        output_database.parent,
        seasons,
        config,
        representative_source_days,
        cluster_assignments,
        cluster_weights,
        accuracy,
        engine,
        forced_day_ids,
        objective_value,
        extreme_records,
        selected_columns,
        column_weights,
        metadata,
        iteration_history,
    )
    print(f"Created reduced ACES database: {output_database}")


def validate_clustering_settings(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    clustering: dict[str, Any],
    seasons: list[str],
    hours: list[str],
    engine: str,
    calendar: dict[str, Any],
) -> None:
    allowed_engines = {
        "tsam",
        "pyomo",
        "pyomo_milp",
        "pyomo_lp_fixed_days",
    }
    if engine not in allowed_engines:
        raise ValueError(
            "clustering.engine must be tsam, pyomo, pyomo_milp, or "
            "pyomo_lp_fixed_days."
        )
    n_representative_days = int(clustering["n_representative_days"])
    if n_representative_days < 1 or n_representative_days > len(seasons):
        raise ValueError(
            "n_representative_days must be between 1 and the number of "
            "complete ACES days."
        )
    if engine == "tsam":
        tsam_config = clustering.get("tsam")
        if not isinstance(tsam_config, dict):
            raise ValueError("Missing clustering.tsam configuration.")
        if int(tsam_config.get("period_duration", 24)) != 24:
            raise ValueError(
                "The ACES database writer requires tsam.period_duration: 24."
            )
        cluster_config = tsam_config.get("cluster", {})
        allowed_methods = {
            "hierarchical",
            "kmeans",
            "kmedoids",
            "kmaxoids",
            "averaging",
            "contiguous",
        }
        if cluster_config.get("method") not in allowed_methods:
            raise ValueError(
                "clustering.tsam.cluster.method must be hierarchical, kmeans, "
                "kmedoids, kmaxoids, averaging, or contiguous."
            )
        representation = cluster_config.get("representation", "medoid")
        representation_type = (
            representation.get("type", "medoid")
            if isinstance(representation, dict)
            else representation
        )
        if representation_type != "medoid":
            raise ValueError(
                "The ACES source-day writer requires TSAM representation.type: "
                "medoid. Synthetic representations such as mean or distribution "
                "cannot be labeled as one historical source day."
            )
        extremes = tsam_config.get("extremes", {})
        if extremes.get("enabled", False) and extremes.get(
            "method", "append"
        ) == "replace":
            raise ValueError(
                "TSAM extremes.method: replace is not supported by the ACES "
                "profile-transfer workflow. Use append or new_cluster."
            )
        segmentation = tsam_config.get("segmentation", {})
        if segmentation.get("enabled", False) and int(
            segmentation.get("n_segments", 24)
        ) != 24:
            raise ValueError(
                "The ACES database writer requires 24 TSAM segments (H00-H23)."
            )
        if bool(
            tsam_config.get("aggregate", {}).get(
                "preserve_column_means", False
            )
        ):
            raise ValueError(
                "Use tsam.aggregate.preserve_column_means: false with medoid "
                "output so representative profiles remain exact historical days."
            )
        build_extreme_audit_records(
            database=database,
            profiles=profiles,
            selected_columns=selected_columns,
            metadata=metadata,
            clustering=clustering,
            seasons=seasons,
            hours=hours,
            engine=engine,
            calendar=calendar,
        )
        return

    pyomo_config = clustering.get("pyomo")
    if not isinstance(pyomo_config, dict):
        raise ValueError("Missing clustering.pyomo configuration.")
    solution_method = pyomo_config.get("solution_method", "opt")
    if solution_method not in {"opt", "hybrid_random_weighting"}:
        raise ValueError(
            "clustering.pyomo.solution_method must be opt or "
            "hybrid_random_weighting."
        )
    formulation = pyomo_config.get("formulation", "milp")
    if engine == "pyomo_milp":
        formulation = "milp"
    elif engine == "pyomo_lp_fixed_days":
        formulation = "lp_fixed_days"
    if formulation not in {"milp", "lp_fixed_days"}:
        raise ValueError(
            "clustering.pyomo.formulation must be milp or lp_fixed_days."
        )
    if formulation == "lp_fixed_days":
        fixed_day_ids = pyomo_config.get("fixed_day_ids", [])
        if len(set(fixed_day_ids)) != n_representative_days:
            raise ValueError(
                "lp_fixed_days requires exactly n_representative_days unique "
                "fixed_day_ids."
            )
        if bool(pyomo_config.get("use_integer_weights", False)):
            raise ValueError(
                "lp_fixed_days requires use_integer_weights: false."
            )
    if solution_method == "hybrid_random_weighting":
        if formulation != "milp":
            raise ValueError(
                "hybrid_random_weighting requires formulation: milp."
            )
        if int(pyomo_config.get("n_random_iterations", 50)) < 1:
            raise ValueError("n_random_iterations must be at least 1.")
    elif pyomo_config.get("sampled_candidate_pool_size") is not None:
        raise ValueError(
            "sampled_candidate_pool_size is used only by "
            "hybrid_random_weighting; set it to null for solution_method: opt."
        )

    candidate_method = pyomo_config.get("candidate_reduction_method", "none")
    allowed_candidate_methods = {
        "none",
        "kmeans_nearest_day",
        "ward_nearest_day",
        "extreme_plus_kmeans",
        "extreme_plus_ward",
    }
    if candidate_method not in allowed_candidate_methods:
        raise ValueError(
            "Unsupported clustering.pyomo.candidate_reduction_method."
        )
    candidate_count = pyomo_config.get("n_candidate_days")
    if candidate_method != "none":
        if candidate_count is None:
            raise ValueError(
                "n_candidate_days is required when Pyomo candidate reduction "
                "is enabled."
            )
        if not n_representative_days <= int(candidate_count) <= len(seasons):
            raise ValueError(
                "n_candidate_days must be between n_representative_days and "
                "the number of original days."
            )
    feature_mode = pyomo_config.get(
        "candidate_feature_mode", "chronological_daily_profile"
    )
    if feature_mode not in {
        "chronological_daily_profile",
        "daily_duration_curve",
        "hybrid_profile_and_duration",
        "summary_statistics",
    }:
        raise ValueError("Unsupported clustering.pyomo.candidate_feature_mode.")

    extreme_records = build_extreme_audit_records(
        database=database,
        profiles=profiles,
        selected_columns=selected_columns,
        metadata=metadata,
        clustering=clustering,
        seasons=seasons,
        hours=hours,
        engine=engine,
        calendar=calendar,
    )
    forced_day_ids = {
        int(record["day_id"]) for record in extreme_records
    }
    if len(forced_day_ids) > n_representative_days:
        raise ValueError(
            f"Pyomo configuration forces {len(forced_day_ids)} unique days, "
            f"which exceeds n_representative_days={n_representative_days}."
        )
    invalid = sorted(
        day_id
        for day_id in forced_day_ids
        if day_id < 1 or day_id > len(seasons)
    )
    if invalid:
        raise ValueError(f"Invalid Pyomo forced day IDs: {invalid}")


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, inherited: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    inherited = set() if inherited is None else set(inherited)
    if path in inherited:
        raise ValueError(f"Circular YAML configuration inheritance: {path}")
    inherited.add(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a top-level mapping.")
    parent = config.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        config = merge_config(load_config(parent_path, inherited), config)
    for section in ("project", "aces", "clustering"):
        if section not in config:
            raise ValueError(f"Missing required YAML section: {section}")
    return config


V4_TEMPORAL_TABLES: dict[str, dict[str, Any]] = {
    "DemandSpecificDistribution": {
        "table": "demand_specific_distribution",
        "source_season_format": "%m-%d",
        "season_column": "season",
        "hour_column": "tod",
        "value_column": "dsd",
        "key_columns": ["region", "period", "demand_name"],
        "preserve_columns": ["notes"],
    },
    "CapacityFactorTech": {
        "table": "capacity_factor_tech",
        "source_season_format": "%m-%d",
        "season_column": "season",
        "hour_column": "tod",
        "value_column": "factor",
        "key_columns": ["region", "tech"],
        "preserve_columns": ["notes"],
    },
    "CapacityFactorProcess": {
        "table": "capacity_factor_process",
        "source_season_format": "%m-%d",
        "season_column": "season",
        "hour_column": "tod",
        "value_column": "factor",
        "key_columns": ["region", "tech", "vintage"],
        "preserve_columns": ["notes"],
    },
    "EfficiencyVariable": {
        "table": "efficiency_variable",
        "source_season_format": "%m-%d",
        "season_column": "season",
        "hour_column": "tod",
        "value_column": "efficiency",
        "key_columns": [
            "region", "input_comm", "tech", "vintage", "output_comm"
        ],
        "preserve_columns": ["notes"],
    },
    "CostVariableVariable": {
        "table": "cost_variable_multiplier",
        "source_season_format": "%m-%d",
        "season_column": "season",
        "hour_column": "tod",
        "value_column": "multiplier",
        "key_columns": ["region", "tech"],
        "preserve_columns": ["notes"],
    },
}


def detect_database_schema(database: Path) -> str:
    if not database.exists():
        raise FileNotFoundError(f"Input ACES database not found: {database}")
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "metadata" in tables:
            columns = {
                row[1] for row in connection.execute('PRAGMA table_info("metadata")')
            }
            if {"element", "value"}.issubset(columns):
                row = connection.execute(
                    "SELECT value FROM metadata WHERE element = 'DB_MAJOR'"
                ).fetchone()
                if row is not None and int(row[0]) == 4:
                    return "temoa_v4"
        if "time_season" in tables:
            columns = {
                row[1]
                for row in connection.execute('PRAGMA table_info("time_season")')
            }
            if {"season", "segment_fraction"}.issubset(columns):
                return "temoa_v4"
            if "t_season" in columns:
                return "aces_legacy"
        if "DemandSpecificDistribution" in tables:
            return "aces_legacy"
    raise ValueError(
        "Could not identify the ACES database schema. Expected either legacy "
        "ACES tables or Temoa metadata with DB_MAJOR=4."
    )


def configure_database_schema(
    database: Path, original_config: dict[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(original_config)
    aces = config["aces"]
    requested = str(aces.get("schema", "auto")).lower()
    aliases = {
        "legacy": "aces_legacy",
        "aces": "aces_legacy",
        "aces_current": "aces_legacy",
        "v4": "temoa_v4",
        "aces_temoa_v4": "temoa_v4",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "aces_legacy", "temoa_v4"}:
        raise ValueError(
            "aces.schema must be auto, aces_legacy, or temoa_v4."
        )
    detected = detect_database_schema(database)
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"Configured schema {requested!r} does not match detected schema "
            f"{detected!r} in {database}."
        )
    aces["schema"] = detected
    if detected == "aces_legacy":
        for table in aces.get("temporal_tables", []):
            table.setdefault("logical_table", table["table"])
        aces.setdefault("weights", {})["mode"] = "hourly_table"
        return config

    aces["calendar"].update(
        {
            "season_table": "time_season",
            "season_column": "season",
            "hour_table": "time_of_day",
            "hour_column": "tod",
        }
    )
    aces["weights"] = {
        "mode": "season_fraction",
        "table": "time_season",
        "season_column": "season",
        "value_column": "segment_fraction",
        "notes_column": "notes",
        "sequence_column": "sequence",
        "create_audit_table": aces.get("weights", {}).get(
            "create_audit_table", True
        ),
        "audit_table": aces.get("weights", {}).get(
            "audit_table", "RepresentativeDayWeight"
        ),
    }
    aces["sequential_time"] = {
        "table": "time_season_sequential",
        "sequence_column": "sequence",
        "label_column": "seas_seq",
        "season_column": "season",
        "value_column": "segment_fraction",
        "notes_column": "notes",
    }
    for table in aces.get("temporal_tables", []):
        logical = table.get("logical_table", table["table"])
        table["logical_table"] = logical
        mapping = V4_TEMPORAL_TABLES.get(logical)
        if mapping is not None:
            table.update(mapping)

    for engine_name in ("tsam", "pyomo"):
        engine = config.get("clustering", {}).get(engine_name, {})
        for rule_name in (
            "include_max_net_load_ramp_day",
            "include_peak_net_load_day",
        ):
            net_load = engine.get("extremes", {}).get(rule_name, {})
            if not isinstance(net_load, dict):
                continue
            demand = net_load.setdefault("demand", {})
            demand.update(
                {
                    "annual_table": "demand",
                    "region_column": "region",
                    "period_column": "period",
                    "commodity_column": "commodity",
                    "value_column": "demand",
                    "unit_column": "units",
                }
            )
            demand.setdefault("profile_table", "DemandSpecificDistribution")
            demand["segfrac"] = {"mode": "calendar"}
            generation = net_load.setdefault("existing_generation", {})
            generation.update(
                {
                    "capacity_table": "existing_capacity",
                    "region_column": "region",
                    "tech_column": "tech",
                    "value_column": "capacity",
                    "unit_column": "units",
                }
            )
            generation.setdefault(
                "capacity_factor_table", "CapacityFactorTech"
            )
        extremes = engine.get("extremes", {})
        for scope_name in ("system_wide", "subsystem"):
            scope = extremes.get(scope_name, {})
            if not isinstance(scope, dict):
                continue
            demand = scope.setdefault("demand", {})
            demand.update(
                {
                    "annual_table": "demand",
                    "region_column": "region",
                    "period_column": "period",
                    "commodity_column": "commodity",
                    "value_column": "demand",
                    "unit_column": "units",
                }
            )
            demand.setdefault("profile_table", "DemandSpecificDistribution")
            demand["segfrac"] = {"mode": "calendar"}
            generation = scope.setdefault("existing_generation", {})
            generation.update(
                {
                    "capacity_table": "existing_capacity",
                    "region_column": "region",
                    "tech_column": "tech",
                    "value_column": "capacity",
                    "unit_column": "units",
                }
            )
            generation.setdefault(
                "capacity_factor_table", "CapacityFactorTech"
            )
    return config


def load_aces_profiles(
    database: Path, config: dict[str, Any], base_directory: Path
) -> tuple[pd.DataFrame, dict[str, ProfileMetadata], list[str], list[str]]:
    if not database.exists():
        raise FileNotFoundError(f"Input ACES database not found: {database}")

    calendar = config["aces"]["calendar"]
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        seasons = [
            row[0]
            for row in connection.execute(
                f'SELECT "{calendar["season_column"]}" '
                f'FROM "{calendar["season_table"]}"'
            )
        ]
        hours = [
            row[0]
            for row in connection.execute(
                f'SELECT "{calendar["hour_column"]}" '
                f'FROM "{calendar["hour_table"]}"'
            )
        ]
        seasons = sort_seasons(
            seasons, calendar["season_format"], int(calendar["base_year"])
        )
        hours = sorted(hours, key=parse_hour)
        expected_hours = int(calendar["hours_per_day"])
        if len(hours) != expected_hours:
            raise ValueError(
                f"Expected {expected_hours} time_of_day rows, found {len(hours)}."
            )
        if config["aces"].get("schema") == "temoa_v4":
            season_fractions = [
                float(row[0])
                for row in connection.execute(
                    'SELECT segment_fraction FROM "time_season"'
                )
            ]
            tod_hours = [
                float(row[0])
                for row in connection.execute('SELECT hours FROM "time_of_day"')
            ]
            if (
                any(value <= 0 for value in season_fractions)
                or not np.isclose(sum(season_fractions), 1.0, atol=1e-8)
            ):
                raise ValueError(
                    "Temoa v4 time_season.segment_fraction values must be "
                    "positive and sum to 1."
                )
            if any(value <= 0 for value in tod_hours) or not np.isclose(
                sum(tod_hours), 24.0, atol=1e-8
            ):
                raise ValueError(
                    "Temoa v4 time_of_day.hours values must be positive and "
                    "sum to 24."
                )

        index = build_datetime_index(seasons, hours, calendar)
        frames: list[pd.DataFrame] = []
        metadata: dict[str, ProfileMetadata] = {}
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table_config in config["aces"]["temporal_tables"]:
            if not table_config.get("enabled", True):
                continue
            table = table_config["table"]
            if table not in table_names:
                if config.get("validation", {}).get(
                    "fail_on_missing_temporal_table", False
                ):
                    raise ValueError(f"Configured temporal table not found: {table}")
                print(f"Skipping missing table: {table}")
                continue
            frame, table_metadata = load_temporal_table(
                connection,
                table_config,
                seasons,
                hours,
                index,
                calendar["season_format"],
                int(calendar["base_year"]),
            )
            if frame is None:
                print(f"Skipping empty table: {table}")
                continue
            frames.append(frame)
            metadata.update(table_metadata)

    if not frames:
        raise ValueError("No enabled ACES temporal table contained data.")
    profiles = pd.concat(frames, axis=1)
    profiles = merge_extra_attributes(
        profiles,
        resolve_path(
            base_directory,
            config["project"].get(
                "extra_attributes_directory", "extra_attributes"
            ),
        ),
    )
    if profiles.isna().any().any():
        missing = profiles.columns[profiles.isna().any()].tolist()
        raise ValueError(
            "Incomplete hourly profiles were found. First missing profiles: "
            f"{missing[:10]}"
        )
    return profiles, metadata, seasons, hours


def load_temporal_table(
    connection: sqlite3.Connection,
    table_config: dict[str, Any],
    seasons: list[str],
    hours: list[str],
    index: pd.DatetimeIndex,
    canonical_season_format: str,
    base_year: int,
) -> tuple[pd.DataFrame | None, dict[str, ProfileMetadata]]:
    table = table_config["table"]
    logical_table = table_config.get("logical_table", table)
    key_columns = table_config["key_columns"]
    season_column = table_config["season_column"]
    hour_column = table_config["hour_column"]
    value_column = table_config["value_column"]
    preserve_columns = table_config.get("preserve_columns", [])

    row_count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    if row_count == 0:
        return None, {}

    quoted_keys = ", ".join(f'"{column}"' for column in key_columns)
    key_rows = connection.execute(
        f'SELECT DISTINCT {quoted_keys} FROM "{table}" ORDER BY {quoted_keys}'
    ).fetchall()
    key_to_position = {tuple(row): position for position, row in enumerate(key_rows)}
    season_to_position = {value: position for position, value in enumerate(seasons)}
    hour_to_position = {value: position for position, value in enumerate(hours)}
    season_mapping = build_season_mapping(
        connection,
        table,
        season_column,
        seasons,
        table_config.get("source_season_format", canonical_season_format),
        canonical_season_format,
        base_year,
    )
    matrix = np.full((len(seasons) * len(hours), len(key_rows)), np.nan)

    selected = key_columns + [season_column, hour_column, value_column]
    selected_columns = ", ".join(f'"{column}"' for column in selected)
    query = f'SELECT {selected_columns} FROM "{table}"'
    cursor = connection.execute(query)
    while True:
        rows = cursor.fetchmany(100_000)
        if not rows:
            break
        for row in rows:
            key = tuple(row[: len(key_columns)])
            raw_season = row[len(key_columns)]
            season = season_mapping.get(raw_season)
            hour = row[len(key_columns) + 1]
            value = row[len(key_columns) + 2]
            if season is None or hour not in hour_to_position:
                continue
            row_position = (
                season_to_position[season] * len(hours) + hour_to_position[hour]
            )
            matrix[row_position, key_to_position[key]] = value

    columns = [profile_name(logical_table, key) for key in key_rows]
    frame = pd.DataFrame(matrix, index=index, columns=columns)
    metadata: dict[str, ProfileMetadata] = {}
    for key, column in zip(key_rows, columns):
        preserve_values: tuple[Any, ...] = ()
        if preserve_columns:
            where = " AND ".join(f'"{name}" = ?' for name in key_columns)
            selected_preserve_columns = ", ".join(
                f'"{name}"' for name in preserve_columns
            )
            preserve_query = (
                f"SELECT {selected_preserve_columns} "
                f'FROM "{table}" WHERE {where} LIMIT 1'
            )
            preserve_row = connection.execute(preserve_query, tuple(key)).fetchone()
            preserve_values = tuple(preserve_row or [None] * len(preserve_columns))
        metadata[column] = ProfileMetadata(
            table=table,
            logical_table=logical_table,
            key_columns=key_columns,
            key_values=tuple(key),
            season_column=season_column,
            hour_column=hour_column,
            value_column=value_column,
            preserve_columns=preserve_columns,
            preserve_values=preserve_values,
            output_rule=table_config.get("output_rule", "representative_profile"),
        )
    return frame, metadata


def build_season_mapping(
    connection: sqlite3.Connection,
    table: str,
    season_column: str,
    canonical_seasons: list[str],
    source_season_format: str,
    canonical_season_format: str,
    base_year: int,
) -> dict[str, str]:
    raw_seasons = {
        row[0]
        for row in connection.execute(
            f'SELECT DISTINCT "{season_column}" FROM "{table}"'
        )
    }
    canonical = set(canonical_seasons)
    mapping: dict[str, str] = {}
    invalid: list[str] = []
    for raw_season in raw_seasons:
        try:
            parsed = datetime.strptime(
                f"{base_year}-{raw_season}",
                f"%Y-{source_season_format}",
            )
        except (TypeError, ValueError):
            invalid.append(str(raw_season))
            continue
        canonical_season = parsed.strftime(canonical_season_format)
        if canonical_season in canonical:
            mapping[raw_season] = canonical_season

    mapped = list(mapping.values())
    if len(mapping) != len(raw_seasons) or len(set(mapped)) != len(mapped):
        missing = sorted(raw_seasons - set(mapping))
        raise ValueError(
            f'{table}.{season_column} cannot be mapped uniquely from '
            f'{source_season_format} to the time_season calendar format '
            f'{canonical_season_format}. First invalid or unmapped labels: '
            f"{sorted(set(invalid + [str(value) for value in missing]))[:10]}"
        )
    if source_season_format != canonical_season_format:
        example = next(iter(sorted(raw_seasons)))
        print(
            f"Aligned all {len(mapping)} {table} season labels from "
            f"{source_season_format} to {canonical_season_format}; "
            f"example: {example} -> {mapping[example]}"
        )
    return mapping


def merge_extra_attributes(
    profiles: pd.DataFrame, directory: Path
) -> pd.DataFrame:
    if not directory.exists():
        return profiles
    result = profiles.copy()
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".csv":
            extra = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            extra = pd.read_excel(path)
        else:
            continue
        if "timestamp" not in extra:
            raise ValueError(f"Extra attribute file needs timestamp: {path}")
        extra["timestamp"] = pd.to_datetime(extra["timestamp"], errors="raise")
        extra = extra.set_index("timestamp")
        duplicate = set(extra.columns).intersection(result.columns)
        if duplicate:
            raise ValueError(f"Duplicate extra attribute columns: {sorted(duplicate)}")
        result = result.join(extra, how="left")
    return result


def select_clustering_columns(
    columns: list[str],
    metadata: dict[str, ProfileMetadata],
    attributes: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    include = attributes.get("include", {})
    exclude = attributes.get("exclude", {})
    included_tables = set(include.get("tables", []))
    included_extra = set(include.get("extra_attributes", []))
    excluded_regions = set(str(value) for value in exclude.get("regions", []))
    excluded_techs = set(str(value) for value in exclude.get("technologies", []))
    excluded_demands = set(str(value) for value in exclude.get("demands", []))
    excluded_columns = set(exclude.get("columns", []))
    defaults = attributes.get("table_default_weights", {})
    overrides = attributes.get("column_weight_overrides", {})

    selected: list[str] = []
    weights: dict[str, float] = {}
    for column in columns:
        item = metadata.get(column)
        if item is None:
            if column not in included_extra:
                continue
            selected.append(column)
            weights[column] = float(overrides.get(column, 1.0))
            continue
        if item.logical_table not in included_tables or column in excluded_columns:
            continue
        keyed = {
            name: str(value)
            for name, value in zip(item.key_columns, item.key_values)
        }
        if (keyed.get("regions") or keyed.get("region")) in excluded_regions:
            continue
        if keyed.get("tech") in excluded_techs:
            continue
        if keyed.get("demand_name") in excluded_demands:
            continue
        selected.append(column)
        weights[column] = float(
            overrides.get(column, defaults.get(item.logical_table, 1.0))
        )
    if not selected:
        raise ValueError("Attribute filters selected zero clustering profiles.")
    return selected, weights


def add_tsam_day_markers(
    profiles: pd.DataFrame,
    column_weights: dict[str, float],
    forced_day_ids: list[int],
    hours_per_day: int,
    marker_weight: float,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    if not forced_day_ids:
        return profiles, dict(column_weights), []
    if marker_weight <= 0:
        raise ValueError("tsam.extremes.custom_marker_weight must be positive.")
    if len(profiles) % hours_per_day != 0:
        raise ValueError("TSAM custom extreme markers require complete days.")
    n_days = len(profiles) // hours_per_day
    invalid = [
        day_id for day_id in forced_day_ids if day_id < 1 or day_id > n_days
    ]
    if invalid:
        raise ValueError(f"Invalid TSAM custom extreme day IDs: {invalid}")

    augmented = profiles.copy()
    weights = dict(column_weights)
    marker_columns = []
    for day_id in sorted(set(forced_day_ids)):
        column = f"__ACES_FORCED_DAY_{day_id:04d}"
        values = np.zeros(len(augmented), dtype=float)
        start = (day_id - 1) * hours_per_day
        values[start : start + hours_per_day] = 1.0
        augmented[column] = values
        weights[column] = marker_weight
        marker_columns.append(column)
    return augmented, weights, marker_columns


def get_tsam_cluster_weights(result: Any) -> dict[int, float]:
    raw_weights = getattr(result, "cluster_weights", None)
    if raw_weights is None:
        raw_weights = getattr(result, "cluster_counts", None)
    if raw_weights is None:
        raise ValueError("TSAM result does not provide cluster weights/counts.")
    return {
        int(cluster): float(weight)
        for cluster, weight in dict(raw_weights).items()
    }


def run_tsam(
    profiles: pd.DataFrame,
    column_weights: dict[str, float],
    clustering: dict[str, Any],
    custom_extreme_columns: list[str] | None = None,
):
    try:
        import tsam
        from tsam import (
            ClusterConfig,
            Distribution,
            ExtremeConfig,
            MinMaxMean,
            SegmentConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "TSAM is not installed in this Python environment. "
            "Run: python -m pip install tsam"
        ) from exc

    tsam_config = clustering["tsam"]
    cluster_data = tsam_config["cluster"]
    if cluster_data.get("normalize_column_means", False):
        tolerance = float(
            tsam_config.get("aggregate", {}).get("numerical_tolerance", 1e-13)
        )
        constant_columns = profiles.columns[
            (profiles.max(axis=0) - profiles.min(axis=0)).abs() <= tolerance
        ].tolist()
        if constant_columns:
            profiles = profiles.drop(columns=constant_columns)
            column_weights = {
                column: weight
                for column, weight in column_weights.items()
                if column in profiles.columns
            }
            print(
                f"Excluded {len(constant_columns)} constant profiles from "
                "distance calculations; they remain in the output database."
            )
    cluster_options = {
        "method": cluster_data["method"],
        "representation": build_representation(
            cluster_data.get("representation"), Distribution, MinMaxMean
        ),
        "use_duration_curves": bool(
            cluster_data.get("use_duration_curves", False)
        ),
        "include_period_sums": bool(
            cluster_data.get("include_period_sums", False)
        ),
        "solver": cluster_data.get("solver", "highs"),
    }
    normalize_means = bool(
        cluster_data.get("normalize_column_means", False)
    )
    cluster_parameters = inspect.signature(ClusterConfig).parameters
    if "scale_by_column_means" in cluster_parameters:
        cluster_options["scale_by_column_means"] = normalize_means
    else:
        cluster_options["normalize_column_means"] = normalize_means
    cluster = ClusterConfig(**cluster_options)
    extreme_data = tsam_config.get("extremes", {})
    extremes = None
    custom_extreme_columns = custom_extreme_columns or []
    if extreme_data.get("enabled", False) or custom_extreme_columns:
        extreme_columns = {
            key: extreme_data.get(key, [])
            for key in ("max_value", "min_value", "max_period", "min_period")
        }
        extreme_columns["max_value"] = list(
            dict.fromkeys(
                list(extreme_columns["max_value"]) + custom_extreme_columns
            )
        )
        missing = sorted(
            {
                column
                for values in extreme_columns.values()
                for column in values
                if column not in profiles.columns
            }
        )
        if missing:
            raise ValueError(
                "Extreme profile names were not found. Check spelling: "
                f"{missing}"
            )
        if any(extreme_columns.values()):
            extremes = ExtremeConfig(
                method=extreme_data.get("method", "append"),
                **extreme_columns,
            )

    segment_data = tsam_config.get("segmentation", {})
    segments = None
    if segment_data.get("enabled", False):
        if int(segment_data["n_segments"]) != 24:
            raise ValueError(
                "ACES database writing currently requires 24 segments (H00-H23)."
            )
        segments = SegmentConfig(
            n_segments=24,
            representation=build_representation(
                segment_data.get("representation"), Distribution, MinMaxMean
            ),
        )

    aggregate_data = tsam_config.get("aggregate", {})
    return tsam.aggregate(
        profiles,
        n_clusters=int(clustering["n_representative_days"]),
        period_duration=tsam_config.get("period_duration", 24),
        temporal_resolution=tsam_config.get("temporal_resolution", "1h"),
        cluster=cluster,
        segments=segments,
        extremes=extremes,
        weights=column_weights,
        preserve_column_means=bool(
            aggregate_data.get("preserve_column_means", True)
        ),
        rescale_exclude_columns=aggregate_data.get(
            "rescale_exclude_columns", []
        ),
        round_decimals=aggregate_data.get("round_decimals"),
        numerical_tolerance=float(
            aggregate_data.get("numerical_tolerance", 1e-13)
        ),
    )


def run_pyomo(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    column_weights: dict[str, float],
    metadata: dict[str, ProfileMetadata],
    clustering: dict[str, Any],
    seasons: list[str],
    hours: list[str],
    engine: str,
    calendar: dict[str, Any],
    extreme_records: list[dict[str, Any]] | None = None,
) -> PyomoAcesResult:
    try:
        from repday.candidate_reduction import reduce_candidate_days
        from repday.config import AttributeConfig, SolverConfig
        from repday.model import RepresentativeDayOptimizer
        from repday.preprocessing import prepare_daily_structures
        from repday.search import run_full_opt, run_hybrid_random_weighting
    except ImportError as exc:
        raise RuntimeError(
            "Pyomo representative-day dependencies are not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    pyomo_config = clustering.get("pyomo", {})
    n_representative_days = int(clustering["n_representative_days"])
    hours_per_day = len(hours)
    if len(profiles) != len(seasons) * hours_per_day:
        raise ValueError("The ACES profile matrix does not contain complete days.")

    normalization = pyomo_config.get("normalization", "minmax")
    attributes = [
        AttributeConfig(
            name=column,
            column=column,
            weight=float(column_weights[column]),
            normalize=normalization != "none",
            normalization=normalization,
        )
        for column in selected_columns
    ]
    if not any(attribute.weight > 0 for attribute in attributes):
        raise ValueError("Pyomo requires at least one positive attribute weight.")

    hourly = profiles[selected_columns].reset_index()
    hourly["day_id"] = np.repeat(
        np.arange(1, len(seasons) + 1), hours_per_day
    )
    hourly["hour_in_day"] = np.tile(
        np.arange(hours_per_day), len(seasons)
    )
    prepared = prepare_daily_structures(
        hourly=hourly,
        attributes=attributes,
        hours_per_day=hours_per_day,
        require_full_days=True,
    )

    if extreme_records is None:
        extreme_records = build_extreme_audit_records(
            database=database,
            profiles=profiles,
            selected_columns=selected_columns,
            metadata=metadata,
            clustering=clustering,
            seasons=seasons,
            hours=hours,
            engine=engine,
            calendar=calendar,
        )
    forced_day_ids = sorted(
        {int(record["day_id"]) for record in extreme_records}
    )
    invalid_forced = [
        day_id
        for day_id in forced_day_ids
        if day_id < 1 or day_id > len(seasons)
    ]
    if invalid_forced:
        raise ValueError(
            f"Pyomo forced_day_ids are outside the ACES year: {invalid_forced}"
        )
    if len(forced_day_ids) > n_representative_days:
        raise ValueError(
            f"Pyomo found {len(forced_day_ids)} unique forced extreme days, "
            f"but n_representative_days is {n_representative_days}. Increase "
            "the representative-day count or remove some extreme rules."
        )

    candidate_method = pyomo_config.get(
        "candidate_reduction_method", "none"
    )
    reduction_result = reduce_candidate_days(
        prepared=prepared,
        attributes=attributes,
        method=candidate_method,
        feature_mode=pyomo_config.get(
            "candidate_feature_mode", "chronological_daily_profile"
        ),
        n_candidate_days=pyomo_config.get("n_candidate_days"),
        include_extreme_days=False,
        random_seed=int(pyomo_config.get("random_seed", 42)),
    )
    candidate_day_ids = sorted(
        set(reduction_result.candidate_day_ids) | set(forced_day_ids)
    )

    solver_data = pyomo_config.get("solver", {})
    solver_config = SolverConfig(
        solver_name=solver_data.get("name", "highs"),
        tee=bool(solver_data.get("tee", False)),
        timelimit_seconds=solver_data.get("time_limit_seconds"),
        mipgap=solver_data.get("mip_gap"),
        threads=solver_data.get("threads"),
    )
    formulation = pyomo_config.get("formulation", "milp")
    if engine == "pyomo_milp":
        formulation = "milp"
    elif engine == "pyomo_lp_fixed_days":
        formulation = "lp_fixed_days"

    solution_method = pyomo_config.get("solution_method", "opt")
    use_integer_weights = bool(
        pyomo_config.get("use_integer_weights", False)
    )
    enforce_positive = bool(
        pyomo_config.get("enforce_positive_weight", True)
    )
    minimum_weight = float(pyomo_config.get("minimum_weight", 0.01))
    fixed_day_ids = [
        int(day_id) for day_id in pyomo_config.get("fixed_day_ids", [])
    ]

    if formulation == "lp_fixed_days":
        if solution_method != "opt":
            raise ValueError(
                "pyomo_lp_fixed_days supports solution_method: opt only."
            )
        missing_extremes = sorted(set(forced_day_ids) - set(fixed_day_ids))
        if missing_extremes:
            raise ValueError(
                "Every forced extreme must appear in fixed_day_ids when using "
                f"pyomo_lp_fixed_days. Missing: {missing_extremes}"
            )

    optimizer = RepresentativeDayOptimizer(
        prepared=prepared,
        attributes=attributes,
        n_representative_days=n_representative_days,
        solver_config=solver_config,
        hours_per_day=hours_per_day,
        use_integer_weights=use_integer_weights,
        n_bins=int(pyomo_config.get("n_bins", 100)),
        forced_day_ids=forced_day_ids,
        candidate_day_ids=candidate_day_ids,
        formulation=formulation,
        fixed_day_ids=fixed_day_ids,
        enforce_positive_weight_for_selected_days=enforce_positive,
        min_weight_if_selected=minimum_weight,
    )

    if solution_method == "opt":
        optimization_result = run_full_opt(optimizer)
    elif solution_method == "hybrid_random_weighting":
        if formulation != "milp":
            raise ValueError(
                "hybrid_random_weighting selects fixed random day sets itself; "
                "use formulation: milp."
            )
        optimization_result = run_hybrid_random_weighting(
            prepared=prepared,
            attributes=attributes,
            solver_config=solver_config,
            n_representative_days=n_representative_days,
            n_bins=int(pyomo_config.get("n_bins", 100)),
            forced_day_ids=forced_day_ids,
            candidate_day_ids=candidate_day_ids,
            n_random_iterations=int(
                pyomo_config.get("n_random_iterations", 50)
            ),
            random_seed=int(pyomo_config.get("random_seed", 42)),
            sampled_candidate_pool_size=pyomo_config.get(
                "sampled_candidate_pool_size"
            ),
            use_integer_weights=use_integer_weights,
            enforce_positive_weight_for_selected_days=enforce_positive,
            min_weight_if_selected=minimum_weight,
        )
    else:
        raise ValueError(
            "clustering.pyomo.solution_method must be opt or "
            "hybrid_random_weighting."
        )

    selected_day_ids = sorted(optimization_result.selected_days)
    missing_weights = [
        day_id
        for day_id in selected_day_ids
        if optimization_result.day_weights.get(day_id, 0.0) <= 0
    ]
    if missing_weights:
        raise ValueError(
            "The ACES writer requires every selected Pyomo day to have a "
            f"positive weight. Zero-weight days: {missing_weights}"
        )

    representatives = build_pyomo_representatives(
        profiles, selected_day_ids, hours_per_day
    )
    cluster_weights = {
        cluster: float(optimization_result.day_weights[day_id])
        for cluster, day_id in enumerate(selected_day_ids)
    }
    representative_source_days = {
        cluster: seasons[day_id - 1]
        for cluster, day_id in enumerate(selected_day_ids)
    }
    cluster_assignments = assign_days_to_pyomo_representatives(
        prepared=prepared,
        attributes=attributes,
        selected_day_ids=selected_day_ids,
    )
    accuracy = calculate_pyomo_accuracy(
        profiles=profiles[selected_columns],
        representatives=representatives[selected_columns],
        cluster_assignments=cluster_assignments,
        cluster_weights=cluster_weights,
    )
    print(
        "Duration-curve optimization selected "
        f"{len(selected_day_ids)} days using Pyomo; "
        f"{len(forced_day_ids)} were forced extreme/manual days."
    )
    return PyomoAcesResult(
        representatives=representatives,
        cluster_weights=cluster_weights,
        representative_source_days=representative_source_days,
        cluster_assignments=cluster_assignments,
        accuracy=accuracy,
        forced_day_ids=forced_day_ids,
        objective_value=float(optimization_result.objective_value),
        iteration_history=optimization_result.iteration_history,
    )


def build_extreme_audit_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    clustering: dict[str, Any],
    seasons: list[str],
    hours: list[str],
    engine: str,
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    hours_per_day = len(hours)
    is_pyomo = engine.startswith("pyomo")
    if is_pyomo:
        pyomo_config = clustering.get("pyomo", {})
        extreme_config = pyomo_config.get("extremes", {})
        legacy_calculated_keys = sorted(
            set(extreme_config)
            & {
                "include_peak_demand_day",
                "include_min_wind_day",
                "include_peak_net_load_day",
                "include_min_generation_period",
                "include_max_daily_ramp_day",
                "include_max_net_load_ramp_day",
            }
        )
        if legacy_calculated_keys:
            raise ValueError(
                "Pyomo calculated extremes must be placed under "
                "extremes.system_wide or extremes.subsystem. Remove the old "
                f"top-level keys: {legacy_calculated_keys}"
            )
        if (
            (
                extreme_config.get("enabled", False)
                or any(
                    bool(extreme_config.get(name, {}).get("enabled", False))
                    for name in ("system_wide", "subsystem")
                    if isinstance(extreme_config.get(name, {}), dict)
                )
            )
            and extreme_config.get(
                "selection_mode", "force_within_count"
            )
            != "force_within_count"
        ):
            raise ValueError(
                "Pyomo extremes.selection_mode must be force_within_count."
            )
        selection_method = "force_within_count"
    else:
        pyomo_config = {}
        extreme_config = clustering.get("tsam", {}).get("extremes", {})
        selection_method = extreme_config.get("method", "append")

    rules = {
        key: list(extreme_config.get(key, []))
        for key in ("max_value", "min_value", "max_period", "min_period")
    } if extreme_config.get("enabled", False) else {
        key: [] for key in ("max_value", "min_value", "max_period", "min_period")
    }
    configured_columns = {
        column for columns in rules.values() for column in columns
    }
    missing = sorted(configured_columns - set(profiles.columns))
    if missing:
        raise ValueError(
            "Pyomo extreme profile names were not found. Check spelling: "
            f"{missing}"
        )

    n_days = len(profiles) // hours_per_day
    records: list[dict[str, Any]] = []
    for rule_name, columns in rules.items():
        for column in columns:
            values = profiles[column].to_numpy(dtype=float).reshape(
                n_days, hours_per_day
            )
            if rule_name == "max_value":
                flat_position = int(np.argmax(values))
                day_position = flat_position // hours_per_day
                hour = hours[flat_position % hours_per_day]
                metric_value = float(values.reshape(-1)[flat_position])
            elif rule_name == "min_value":
                flat_position = int(np.argmin(values))
                day_position = flat_position // hours_per_day
                hour = hours[flat_position % hours_per_day]
                metric_value = float(values.reshape(-1)[flat_position])
            elif rule_name == "max_period":
                period_values = values.sum(axis=1)
                day_position = int(np.argmax(period_values))
                hour = ""
                metric_value = float(period_values[day_position])
            else:
                period_values = values.sum(axis=1)
                day_position = int(np.argmin(period_values))
                hour = ""
                metric_value = float(period_values[day_position])
            records.append(
                extreme_record(
                    engine=engine,
                    selection_method=selection_method,
                    rule=rule_name,
                    profile=column,
                    metadata=metadata.get(column),
                    day_position=day_position,
                    seasons=seasons,
                    hour=hour,
                    metric_value=metric_value,
                    profile_in_clustering=column in selected_columns,
                )
            )

    if is_pyomo:
        records.extend(
            build_scoped_pyomo_extreme_records(
                database=database,
                profiles=profiles,
                selected_columns=selected_columns,
                metadata=metadata,
                seasons=seasons,
                hours=hours,
                engine=engine,
                selection_method=selection_method,
                extreme_config=extreme_config,
                calendar=calendar,
            )
        )
        records.extend(
            build_manual_forced_day_records(
                pyomo_config.get("forced_day_ids", []),
                seasons,
                engine,
                selection_method,
            )
        )
        return records

    ramp_config = extreme_config.get("include_max_daily_ramp_day", False)
    if ramp_rule_enabled(ramp_config):
        ramp_options = ramp_config if isinstance(ramp_config, dict) else {}
        direction = str(ramp_options.get("direction", "absolute")).lower()
        if direction not in {"absolute", "upward", "downward"}:
            raise ValueError(
                "include_max_daily_ramp_day.direction must be absolute, "
                "upward, or downward."
            )
        demand_columns = [
            column
            for column in selected_columns
            if metadata.get(column) is not None
            and metadata[column].logical_table == "DemandSpecificDistribution"
        ]
        available_regions = sorted(
            {
                profile_key(metadata[column], "regions")
                for column in demand_columns
                if profile_key(metadata[column], "regions") not in {None, ""}
            }
        )
        configured_regions = ramp_options.get("regions", "all")
        if configured_regions == "all":
            regions = available_regions
        elif isinstance(configured_regions, list):
            regions = [str(region) for region in configured_regions]
        else:
            raise ValueError(
                "include_max_daily_ramp_day.regions must be all or a list."
            )
        missing_regions = sorted(set(regions) - set(available_regions))
        if missing_regions:
            raise ValueError(
                "Demand-ramp regions were not found in selected DSD profiles: "
                f"{missing_regions}"
            )
        available_periods = sorted(
            {
                profile_key(metadata[column], "periods")
                for column in demand_columns
                if profile_key(metadata[column], "periods") not in {None, ""}
            },
            key=lambda value: int(value),
        )
        reference_period = resolve_reference_period(
            ramp_options.get("reference_period", "first_model_period"),
            available_periods,
            "demand ramp",
        )
        for region in regions:
            regional_columns = [
                column
                for column in demand_columns
                if str(profile_key(metadata[column], "regions")) == region
                and str(profile_key(metadata[column], "periods"))
                == str(reference_period)
            ]
            if not regional_columns:
                raise ValueError(
                    f"No selected DSD profiles found for demand ramp region "
                    f"{region} and period {reference_period}."
                )
            demand = normalize_for_extreme_detection(
                profiles[regional_columns]
            ).sum(axis=1)
            daily = demand.to_numpy().reshape(n_days, hours_per_day)
            daily_ramps = calculate_ramps(daily, direction)
            day_position = int(np.argmax(daily_ramps.max(axis=1)))
            ramp_position = int(np.argmax(daily_ramps[day_position]))
            records.append(
                extreme_record(
                    engine=engine,
                    selection_method=selection_method,
                    rule="include_max_daily_ramp_day",
                    profile="<selected regional DSD profiles>",
                    metadata=None,
                    day_position=day_position,
                    seasons=seasons,
                    hour=f"{hours[ramp_position]}->{hours[ramp_position + 1]}",
                    metric_value=float(
                        daily_ramps[day_position, ramp_position]
                    ),
                    profile_in_clustering=True,
                    table="DemandSpecificDistribution",
                    profile_count=len(regional_columns),
                    regions=region,
                    periods=reference_period,
                    demand_name="<all selected>",
                )
            )

    peak_net_load_config = extreme_config.get(
        "include_peak_net_load_day", {}
    )
    if ramp_rule_enabled(peak_net_load_config):
        if not isinstance(peak_net_load_config, dict):
            raise ValueError(
                "include_peak_net_load_day must be a YAML mapping."
            )
        records.extend(
            build_peak_net_load_records(
                database=database,
                profiles=profiles,
                selected_columns=selected_columns,
                metadata=metadata,
                seasons=seasons,
                hours=hours,
                engine=engine,
                selection_method=selection_method,
                config=peak_net_load_config,
                calendar=calendar,
            )
        )

    net_load_config = extreme_config.get(
        "include_max_net_load_ramp_day", {}
    )
    if ramp_rule_enabled(net_load_config):
        if not isinstance(net_load_config, dict):
            raise ValueError(
                "include_max_net_load_ramp_day must be a YAML mapping."
            )
        records.extend(
            build_net_load_ramp_records(
                database=database,
                profiles=profiles,
                selected_columns=selected_columns,
                metadata=metadata,
                seasons=seasons,
                hours=hours,
                engine=engine,
                selection_method=selection_method,
                config=net_load_config,
                calendar=calendar,
            )
        )

    return records


def build_manual_forced_day_records(
    day_ids: list[Any],
    seasons: list[str],
    engine: str,
    selection_method: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for day_id in day_ids:
        day_position = int(day_id) - 1
        if day_position < 0 or day_position >= len(seasons):
            records.append(
                {
                    "engine": engine,
                    "selection_method": selection_method,
                    "rule": "manual_forced_day_id",
                    "profile": "",
                    "table": "",
                    "regions": "",
                    "periods": "",
                    "demand_name": "",
                    "tech": "",
                    "profile_keys": "",
                    "day_id": int(day_id),
                    "source_day": "",
                    "hour": "",
                    "metric_value": np.nan,
                    "profile_in_clustering": False,
                    "profile_count": 0,
                }
            )
            continue
        records.append(
            extreme_record(
                engine=engine,
                selection_method=selection_method,
                rule="manual_forced_day_id",
                profile="",
                metadata=None,
                day_position=day_position,
                seasons=seasons,
                hour="",
                metric_value=np.nan,
                profile_in_clustering=False,
            )
        )
    return records


def extreme_record(
    engine: str,
    selection_method: str,
    rule: str,
    profile: str,
    metadata: ProfileMetadata | None,
    day_position: int,
    seasons: list[str],
    hour: str,
    metric_value: float,
    profile_in_clustering: bool,
    table: str | None = None,
    profile_count: int = 1,
    regions: Any | None = None,
    periods: Any | None = None,
    demand_name: Any | None = None,
    tech: Any | None = None,
) -> dict[str, Any]:
    keyed = (
        {
            name: value
            for name, value in zip(
                metadata.key_columns, metadata.key_values
            )
        }
        if metadata is not None
        else {}
    )
    return {
        "engine": engine,
        "selection_method": selection_method,
        "rule": rule,
        "profile": profile,
        "table": table or (metadata.table if metadata is not None else ""),
        "regions": keyed.get("regions", "") if regions is None else regions,
        "periods": keyed.get("periods", "") if periods is None else periods,
        "demand_name": (
            keyed.get("demand_name", "")
            if demand_name is None
            else demand_name
        ),
        "tech": keyed.get("tech", "") if tech is None else tech,
        "profile_keys": " | ".join(
            f"{name}={value}" for name, value in keyed.items()
        ),
        "day_id": day_position + 1,
        "source_day": seasons[day_position],
        "hour": hour,
        "metric_value": metric_value,
        "profile_in_clustering": profile_in_clustering,
        "profile_count": profile_count,
    }


def normalize_for_extreme_detection(frame: pd.DataFrame) -> pd.DataFrame:
    minimum = frame.min(axis=0)
    ranges = frame.max(axis=0) - minimum
    ranges = ranges.mask(ranges.abs() <= 1e-13, 1.0)
    return (frame - minimum) / ranges


def profile_key(metadata: ProfileMetadata, name: str) -> Any | None:
    keyed = dict(zip(metadata.key_columns, metadata.key_values))
    aliases = {
        "regions": ("regions", "region"),
        "region": ("region", "regions"),
        "periods": ("periods", "period"),
        "period": ("period", "periods"),
    }
    for candidate in aliases.get(name, (name,)):
        if candidate in keyed:
            return keyed[candidate]
    return None


def ramp_rule_enabled(config: Any) -> bool:
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return bool(config)


def resolve_reference_period(
    configured_period: Any,
    available_periods: list[Any],
    rule_name: str,
) -> Any:
    if not available_periods:
        raise ValueError(f"No periods are available for {rule_name} selection.")
    if configured_period == "first_model_period":
        return min(available_periods, key=lambda value: int(value))
    matches = [
        period
        for period in available_periods
        if str(period) == str(configured_period)
    ]
    if not matches:
        raise ValueError(
            f"Configured {rule_name} reference period {configured_period} was "
            f"not found. Available periods: {available_periods}"
        )
    return matches[0]


def calculate_ramps(daily_values: np.ndarray, direction: str) -> np.ndarray:
    changes = np.diff(daily_values, axis=1)
    if direction == "absolute":
        return np.abs(changes)
    if direction == "upward":
        return changes
    return -changes


def build_scoped_pyomo_extreme_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    extreme_config: dict[str, Any],
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build explicitly scoped, physically weighted Pyomo extremes."""
    records: list[dict[str, Any]] = []
    for scope_name in ("system_wide", "subsystem"):
        scope = extreme_config.get(scope_name, {})
        if not isinstance(scope, dict):
            raise ValueError(f"Pyomo extremes.{scope_name} must be a mapping.")
        if not bool(scope.get("enabled", False)):
            continue

        regions = resolve_scoped_extreme_regions(scope, metadata)
        region_groups = [regions] if scope_name == "system_wide" else [
            [region] for region in regions
        ]
        for region_group in region_groups:
            peak_demand = scope.get("include_peak_demand_day", {})
            if ramp_rule_enabled(peak_demand):
                config = make_scoped_rule_config(scope, peak_demand, region_group)
                scoped = build_weighted_peak_demand_records(
                    database, profiles, selected_columns, metadata, seasons,
                    hours, engine, selection_method, config, calendar,
                )
                tag_scoped_extreme_records(scoped, scope_name)
                records.extend(scoped)

            min_wind = scope.get("include_min_wind_day", {})
            if ramp_rule_enabled(min_wind):
                config = make_scoped_rule_config(scope, min_wind, region_group)
                config["skip_if_no_positive_capacity"] = (
                    scope_name == "subsystem"
                )
                scoped = build_capacity_weighted_min_wind_records(
                    database, profiles, selected_columns, metadata, seasons,
                    hours, engine, selection_method, config,
                )
                tag_scoped_extreme_records(scoped, scope_name)
                records.extend(scoped)

            peak_net_load = scope.get("include_peak_net_load_day", {})
            if ramp_rule_enabled(peak_net_load):
                config = make_scoped_rule_config(
                    scope, peak_net_load, region_group
                )
                config["include_min_generation_period"] = False
                config["allow_zero_generation"] = scope_name == "subsystem"
                scoped = build_peak_net_load_records(
                    database, profiles, selected_columns, metadata, seasons,
                    hours, engine, selection_method, config, calendar,
                )
                tag_scoped_extreme_records(scoped, scope_name)
                records.extend(scoped)

            min_generation = scope.get(
                "include_min_generation_period", {}
            )
            if ramp_rule_enabled(min_generation):
                if not isinstance(min_generation, dict):
                    raise ValueError(
                        f"extremes.{scope_name}.include_min_generation_period "
                        "must be a mapping containing groups."
                    )
                groups = min_generation.get("groups", [])
                if not isinstance(groups, list) or not groups:
                    raise ValueError(
                        f"extremes.{scope_name}.include_min_generation_period."
                        "groups must contain at least one named technology group."
                    )
                for group in groups:
                    if not isinstance(group, dict) or not group.get("name"):
                        raise ValueError(
                            "Each minimum-generation group must be a mapping "
                            "containing name and technologies."
                        )
                    group_name = str(group["name"])
                    config = make_scoped_rule_config(
                        scope,
                        {
                            "enabled": True,
                            "technologies": group.get("technologies", []),
                        },
                        region_group,
                    )
                    config["include_min_generation_period"] = True
                    config["allow_zero_generation"] = True
                    scoped = build_peak_net_load_records(
                        database, profiles, selected_columns, metadata, seasons,
                        hours, engine, selection_method, config, calendar,
                    )
                    minimum_records = [
                        record for record in scoped
                        if record["rule"] == "include_min_generation_period"
                        and record.get("tech")
                    ]
                    tag_scoped_extreme_records(
                        minimum_records, scope_name, group_name
                    )
                    records.extend(minimum_records)

            demand_ramp = scope.get("include_max_daily_ramp_day", {})
            if ramp_rule_enabled(demand_ramp):
                config = make_scoped_rule_config(
                    scope, demand_ramp, region_group
                )
                scoped = build_weighted_demand_ramp_records(
                    database, profiles, selected_columns, metadata, seasons,
                    hours, engine, selection_method, config, calendar,
                )
                tag_scoped_extreme_records(scoped, scope_name)
                records.extend(scoped)

            net_load_ramp = scope.get(
                "include_max_net_load_ramp_day", {}
            )
            if ramp_rule_enabled(net_load_ramp):
                config = make_scoped_rule_config(
                    scope, net_load_ramp, region_group
                )
                config["aggregate_regions"] = True
                scoped = build_net_load_ramp_records(
                    database, profiles, selected_columns, metadata, seasons,
                    hours, engine, selection_method, config, calendar,
                )
                tag_scoped_extreme_records(scoped, scope_name)
                records.extend(scoped)
    return records


def resolve_scoped_extreme_regions(
    scope: dict[str, Any], metadata: dict[str, ProfileMetadata]
) -> list[str]:
    demand = scope.get("demand", {})
    table = demand.get("profile_table", "DemandSpecificDistribution")
    commodity = str(demand.get("commodity", "D_ELEC"))
    matching = [
        item for item in metadata.values()
        if item.logical_table == table
        and str(profile_key(item, "demand_name")) == commodity
    ]
    periods = sorted(
        {
            profile_key(item, "periods") for item in matching
            if profile_key(item, "periods") not in {None, ""}
        },
        key=lambda value: int(value),
    )
    reference_period = resolve_reference_period(
        scope.get("reference_period", "first_model_period"),
        periods,
        "scoped Pyomo extremes",
    )
    available = sorted(
        {
            str(profile_key(item, "regions")) for item in matching
            if str(profile_key(item, "periods")) == str(reference_period)
        }
    )
    configured = scope.get("regions", "all")
    if configured == "all":
        return available
    if not isinstance(configured, list):
        raise ValueError("Scoped Pyomo extreme regions must be all or a list.")
    regions = [str(region) for region in configured]
    missing = sorted(set(regions) - set(available))
    if missing:
        raise ValueError(
            f"Scoped Pyomo extreme regions were not found: {missing}"
        )
    return regions


def make_scoped_rule_config(
    scope: dict[str, Any], rule: Any, regions: list[str]
) -> dict[str, Any]:
    rule_mapping = rule if isinstance(rule, dict) else {}
    config: dict[str, Any] = {
        "enabled": True,
        "reference_period": scope.get(
            "reference_period", "first_model_period"
        ),
        "regions": regions,
        "demand": copy.deepcopy(scope.get("demand", {})),
        "existing_generation": copy.deepcopy(
            scope.get("existing_generation", {})
        ),
    }
    config = merge_config(config, rule_mapping)
    if "technologies" in rule_mapping:
        config.setdefault("existing_generation", {})["technologies"] = (
            copy.deepcopy(rule_mapping["technologies"])
        )
    return config


def tag_scoped_extreme_records(
    records: list[dict[str, Any]], scope: str, group: str = ""
) -> None:
    for record in records:
        record["extreme_scope"] = scope
        record["generation_group"] = group


def build_weighted_peak_demand_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    config: dict[str, Any],
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    calculation = str(
        config.get("calculation", "annual_demand_weighted")
    ).lower()
    if calculation != "annual_demand_weighted":
        raise ValueError(
            "include_peak_demand_day.calculation must be "
            "annual_demand_weighted."
        )
    demand_config = config.get("demand", {})
    demand_profile_table = demand_config.get(
        "profile_table", "DemandSpecificDistribution"
    )
    demand_commodity = str(demand_config.get("commodity", "D_ELEC"))
    demand_columns = [
        column
        for column, item in metadata.items()
        if item.logical_table == demand_profile_table
        and str(profile_key(item, "demand_name")) == demand_commodity
    ]
    if not demand_columns:
        raise ValueError(
            f"No {demand_profile_table} profiles were found for demand "
            f"commodity {demand_commodity}."
        )
    available_periods = sorted(
        {
            profile_key(metadata[column], "periods")
            for column in demand_columns
            if profile_key(metadata[column], "periods") not in {None, ""}
        },
        key=lambda value: int(value),
    )
    reference_period = resolve_reference_period(
        config.get("reference_period", "first_model_period"),
        available_periods,
        "weighted peak demand",
    )
    available_regions = sorted(
        {
            str(profile_key(metadata[column], "regions"))
            for column in demand_columns
            if str(profile_key(metadata[column], "periods"))
            == str(reference_period)
        }
    )
    configured_regions = config.get("regions", "all")
    if configured_regions == "all":
        regions = available_regions
    elif isinstance(configured_regions, list):
        regions = [str(region) for region in configured_regions]
    else:
        raise ValueError(
            "include_peak_demand_day.regions must be all or a list."
        )
    missing_regions = sorted(set(regions) - set(available_regions))
    if missing_regions:
        raise ValueError(
            "Peak-demand regions were not found in the configured demand "
            f"profiles: {missing_regions}"
        )

    segfrac = load_hourly_segfrac(
        database,
        demand_config.get("segfrac", {}),
        seasons,
        hours,
        calendar["season_format"],
        int(calendar["base_year"]),
    )
    calendar_hours = len(seasons) * len(hours)
    system_demand_gw = np.zeros(len(profiles), dtype=float)
    used_profiles: list[str] = []
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        for region in regions:
            matches = [
                column
                for column in demand_columns
                if str(profile_key(metadata[column], "regions")) == region
                and str(profile_key(metadata[column], "periods"))
                == str(reference_period)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Weighted peak demand requires exactly one DSD profile "
                    f"for region={region}, period={reference_period}, "
                    f"commodity={demand_commodity}; found {len(matches)}."
                )
            column = matches[0]
            annual_energy_gwh = load_annual_demand_gwh(
                connection,
                demand_config,
                region,
                reference_period,
                demand_commodity,
            )
            system_demand_gw += (
                annual_energy_gwh
                * profiles[column].to_numpy(dtype=float)
                / (segfrac * calendar_hours)
            )
            used_profiles.append(column)

    peak_position = int(np.argmax(system_demand_gw))
    day_position, hour_position = divmod(peak_position, len(hours))
    record = extreme_record(
        engine=engine,
        selection_method=selection_method,
        rule="include_peak_demand_day",
        profile="<annual-demand-weighted system electricity demand>",
        metadata=None,
        day_position=day_position,
        seasons=seasons,
        hour=hours[hour_position],
        metric_value=float(system_demand_gw[peak_position]),
        profile_in_clustering=all(
            column in selected_columns for column in used_profiles
        ),
        table=f"{demand_profile_table} + AnnualDemand",
        profile_count=len(used_profiles),
        regions=" | ".join(regions),
        periods=reference_period,
        demand_name=demand_commodity,
    )
    record.update(
        {
            "metric_unit": "GW",
            "demand_at_peak_gw": float(system_demand_gw[peak_position]),
        }
    )
    return [record]


def build_capacity_weighted_min_wind_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    calculation = str(config.get("calculation", "capacity_weighted")).lower()
    if calculation != "capacity_weighted":
        raise ValueError(
            "include_min_wind_day.calculation must be capacity_weighted."
        )
    generation_config = config.get("existing_generation", {})
    technologies = parse_generation_technologies(generation_config)
    if not technologies:
        raise ValueError(
            "include_min_wind_day.existing_generation.technologies must "
            "contain at least one wind technology."
        )
    non_wind = [
        item["tech"] for item in technologies if "WIND" not in item["tech"].upper()
    ]
    if non_wind:
        raise ValueError(
            "include_min_wind_day accepts wind technologies only; found "
            f"{non_wind}. Use include_peak_net_load_day."
        )
    available_regions = sorted(
        {
            str(profile_key(item, "regions"))
            for item in metadata.values()
            if any(
                item.logical_table
                == technology.get(
                    "capacity_factor_table",
                    generation_config.get(
                        "capacity_factor_table", "CapacityFactorTech"
                    ),
                )
                and str(profile_key(item, "tech")) == technology["tech"]
                for technology in technologies
            )
        }
    )
    configured_regions = config.get("regions", "all")
    if configured_regions == "all":
        regions = available_regions
    elif isinstance(configured_regions, list):
        regions = [str(region) for region in configured_regions]
    else:
        raise ValueError(
            "include_min_wind_day.regions must be all or a list."
        )
    system_generation_gw = np.zeros(len(profiles), dtype=float)
    used_profiles: list[str] = []
    used_regions: set[str] = set()
    used_technologies: set[str] = set()
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        for region in regions:
            for technology in technologies:
                technology_name = technology["tech"]
                capacity_gw = load_existing_capacity_gw(
                    connection,
                    generation_config,
                    region,
                    technology_name,
                )
                if capacity_gw <= 0:
                    continue
                cf_table = technology.get(
                    "capacity_factor_table",
                    generation_config.get(
                        "capacity_factor_table", "CapacityFactorTech"
                    ),
                )
                matches = [
                    column
                    for column, item in metadata.items()
                    if item.logical_table == cf_table
                    and str(profile_key(item, "regions")) == region
                    and str(profile_key(item, "tech")) == technology_name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Capacity-weighted minimum wind requires exactly one "
                        f"CF profile for region={region}, "
                        f"technology={technology_name}, table={cf_table}; "
                        f"found {len(matches)}."
                    )
                column = matches[0]
                system_generation_gw += (
                    capacity_gw * profiles[column].to_numpy(dtype=float)
                )
                used_profiles.append(column)
                used_regions.add(region)
                used_technologies.add(technology_name)
    if not used_profiles:
        if bool(config.get("skip_if_no_positive_capacity", False)):
            return []
        raise ValueError(
            "No positive existing wind capacity was found for "
            "include_min_wind_day."
        )

    daily_generation_gwh = system_generation_gw.reshape(
        len(seasons), len(hours)
    ).sum(axis=1)
    day_position = int(np.argmin(daily_generation_gwh))
    record = extreme_record(
        engine=engine,
        selection_method=selection_method,
        rule="include_min_wind_day",
        profile="<capacity-weighted system available wind generation>",
        metadata=None,
        day_position=day_position,
        seasons=seasons,
        hour="",
        metric_value=float(daily_generation_gwh[day_position]),
        profile_in_clustering=all(
            column in selected_columns for column in used_profiles
        ),
        table="ExistingCapacity + CapacityFactor",
        profile_count=len(used_profiles),
        regions=" | ".join(sorted(used_regions)),
        tech=" | ".join(sorted(used_technologies)),
    )
    record.update(
        {
            "metric_unit": "GWh",
            "generation_period_gwh": float(daily_generation_gwh[day_position]),
        }
    )
    return [record]


def build_weighted_demand_ramp_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    config: dict[str, Any],
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select one physical demand-ramp day for the configured region sum."""
    direction = str(config.get("direction", "absolute")).lower()
    if direction not in {"absolute", "upward", "downward"}:
        raise ValueError(
            "include_max_daily_ramp_day.direction must be absolute, upward, "
            "or downward."
        )
    demand_config = config.get("demand", {})
    table = demand_config.get(
        "profile_table", "DemandSpecificDistribution"
    )
    commodity = str(demand_config.get("commodity", "D_ELEC"))
    demand_columns = [
        column for column, item in metadata.items()
        if item.logical_table == table
        and str(profile_key(item, "demand_name")) == commodity
    ]
    periods = sorted(
        {
            profile_key(metadata[column], "periods")
            for column in demand_columns
            if profile_key(metadata[column], "periods") not in {None, ""}
        },
        key=lambda value: int(value),
    )
    reference_period = resolve_reference_period(
        config.get("reference_period", "first_model_period"),
        periods,
        "weighted demand ramp",
    )
    regions = [str(region) for region in config.get("regions", [])]
    if not regions:
        raise ValueError("Weighted demand ramp requires at least one region.")
    segfrac = load_hourly_segfrac(
        database,
        demand_config.get("segfrac", {}),
        seasons,
        hours,
        calendar["season_format"],
        int(calendar["base_year"]),
    )
    calendar_hours = len(seasons) * len(hours)
    demand_gw = np.zeros(len(profiles), dtype=float)
    used_profiles: list[str] = []
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        for region in regions:
            matches = [
                column for column in demand_columns
                if str(profile_key(metadata[column], "regions")) == region
                and str(profile_key(metadata[column], "periods"))
                == str(reference_period)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Weighted demand ramp requires exactly one DSD profile "
                    f"for region={region}, period={reference_period}, "
                    f"commodity={commodity}; found {len(matches)}."
                )
            column = matches[0]
            annual_gwh = load_annual_demand_gwh(
                connection,
                demand_config,
                region,
                reference_period,
                commodity,
            )
            demand_gw += (
                annual_gwh
                * profiles[column].to_numpy(dtype=float)
                / (segfrac * calendar_hours)
            )
            used_profiles.append(column)

    daily = demand_gw.reshape(len(seasons), len(hours))
    ramps = calculate_ramps(daily, direction)
    day_position = int(np.argmax(ramps.max(axis=1)))
    ramp_position = int(np.argmax(ramps[day_position]))
    before = day_position * len(hours) + ramp_position
    after = before + 1
    record = extreme_record(
        engine=engine,
        selection_method=selection_method,
        rule="include_max_daily_ramp_day",
        profile="<annual-demand-weighted electricity demand>",
        metadata=None,
        day_position=day_position,
        seasons=seasons,
        hour=f"{hours[ramp_position]}->{hours[ramp_position + 1]}",
        metric_value=float(ramps[day_position, ramp_position]),
        profile_in_clustering=all(
            column in selected_columns for column in used_profiles
        ),
        table=f"{table} + AnnualDemand",
        profile_count=len(used_profiles),
        regions=" | ".join(regions),
        periods=reference_period,
        demand_name=commodity,
    )
    record.update(
        {
            "ramp_direction": direction,
            "metric_unit": "GW",
            "demand_before_gw": float(demand_gw[before]),
            "demand_after_gw": float(demand_gw[after]),
        }
    )
    return [record]


def build_peak_net_load_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    config: dict[str, Any],
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select one system peak from capacity- and demand-weighted net load."""
    demand_config = config.get("demand", {})
    generation_config = config.get("existing_generation", {})
    demand_profile_table = demand_config.get(
        "profile_table", "DemandSpecificDistribution"
    )
    demand_commodity = str(demand_config.get("commodity", "D_ELEC"))
    demand_columns = [
        column
        for column, item in metadata.items()
        if item.logical_table == demand_profile_table
        and str(profile_key(item, "demand_name")) == demand_commodity
    ]
    if not demand_columns:
        raise ValueError(
            f"No {demand_profile_table} profiles were found for demand "
            f"commodity {demand_commodity}."
        )
    available_periods = sorted(
        {
            profile_key(metadata[column], "periods")
            for column in demand_columns
            if profile_key(metadata[column], "periods") not in {None, ""}
        },
        key=lambda value: int(value),
    )
    reference_period = resolve_reference_period(
        config.get("reference_period", "first_model_period"),
        available_periods,
        "peak net load",
    )
    available_regions = sorted(
        {
            str(profile_key(metadata[column], "regions"))
            for column in demand_columns
            if str(profile_key(metadata[column], "periods"))
            == str(reference_period)
        }
    )
    configured_regions = config.get("regions", "all")
    if configured_regions == "all":
        regions = available_regions
    elif isinstance(configured_regions, list):
        regions = [str(region) for region in configured_regions]
    else:
        raise ValueError(
            "include_peak_net_load_day.regions must be all or a list."
        )
    missing_regions = sorted(set(regions) - set(available_regions))
    if missing_regions:
        raise ValueError(
            "Peak-net-load regions were not found in the configured demand "
            f"profiles: {missing_regions}"
        )

    technologies = parse_generation_technologies(generation_config)
    if not technologies:
        raise ValueError(
            "include_peak_net_load_day.existing_generation.technologies "
            "must contain at least one technology."
        )
    segfrac = load_hourly_segfrac(
        database,
        demand_config.get("segfrac", {}),
        seasons,
        hours,
        calendar["season_format"],
        int(calendar["base_year"]),
    )
    calendar_hours = len(seasons) * len(hours)
    system_demand_gw = np.zeros(len(profiles), dtype=float)
    system_generation_gw = np.zeros(len(profiles), dtype=float)
    used_profiles: list[str] = []
    used_technologies: set[str] = set()

    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        for region in regions:
            matching_demand_columns = [
                column
                for column in demand_columns
                if str(profile_key(metadata[column], "regions")) == region
                and str(profile_key(metadata[column], "periods"))
                == str(reference_period)
            ]
            if len(matching_demand_columns) != 1:
                raise ValueError(
                    "Peak net load requires exactly one DSD profile for "
                    f"region={region}, period={reference_period}, "
                    f"commodity={demand_commodity}; found "
                    f"{len(matching_demand_columns)}."
                )
            demand_column = matching_demand_columns[0]
            annual_energy_gwh = load_annual_demand_gwh(
                connection,
                demand_config,
                region,
                reference_period,
                demand_commodity,
            )
            system_demand_gw += (
                annual_energy_gwh
                * profiles[demand_column].to_numpy(dtype=float)
                / (segfrac * calendar_hours)
            )
            used_profiles.append(demand_column)

            for technology in technologies:
                technology_name = technology["tech"]
                capacity_gw = load_existing_capacity_gw(
                    connection,
                    generation_config,
                    region,
                    technology_name,
                )
                if capacity_gw <= 0:
                    continue
                cf_table = technology.get(
                    "capacity_factor_table",
                    generation_config.get(
                        "capacity_factor_table", "CapacityFactorTech"
                    ),
                )
                cf_columns = [
                    column
                    for column, item in metadata.items()
                    if item.logical_table == cf_table
                    and str(profile_key(item, "regions")) == region
                    and str(profile_key(item, "tech")) == technology_name
                ]
                if len(cf_columns) != 1:
                    raise ValueError(
                        "Expected exactly one capacity-factor profile for "
                        f"region={region}, technology={technology_name}, "
                        f"table={cf_table}; found {len(cf_columns)}."
                    )
                cf_column = cf_columns[0]
                system_generation_gw += (
                    capacity_gw
                    * profiles[cf_column].to_numpy(dtype=float)
                )
                used_profiles.append(cf_column)
                used_technologies.add(technology_name)

    if not used_technologies and not bool(
        config.get("allow_zero_generation", False)
    ):
        raise ValueError(
            "None of the configured peak-net-load technologies has positive "
            "existing capacity in the selected regions."
        )
    system_net_load_gw = system_demand_gw - system_generation_gw
    peak_position = int(np.argmax(system_net_load_gw))
    day_position, hour_position = divmod(peak_position, len(hours))
    record = extreme_record(
        engine=engine,
        selection_method=selection_method,
        rule="include_peak_net_load_day",
        profile="<system actual demand - selected existing generation>",
        metadata=None,
        day_position=day_position,
        seasons=seasons,
        hour=hours[hour_position],
        metric_value=float(system_net_load_gw[peak_position]),
        profile_in_clustering=all(
            column in selected_columns for column in used_profiles
        ),
        table=(
            f"{demand_profile_table} + ExistingCapacity + CapacityFactor"
        ),
        profile_count=len(used_profiles),
        regions=" | ".join(regions),
        periods=reference_period,
        demand_name=demand_commodity,
        tech=" | ".join(sorted(used_technologies)),
    )
    record.update(
        {
            "metric_unit": "GW",
            "demand_at_peak_gw": float(system_demand_gw[peak_position]),
            "generation_at_peak_gw": float(
                system_generation_gw[peak_position]
            ),
            "net_load_at_peak_gw": float(system_net_load_gw[peak_position]),
        }
    )
    records = [record]
    if bool(config.get("include_min_generation_period", False)):
        daily_generation_gwh = system_generation_gw.reshape(
            len(seasons), len(hours)
        ).sum(axis=1)
        minimum_day_position = int(np.argmin(daily_generation_gwh))
        minimum_record = extreme_record(
            engine=engine,
            selection_method=selection_method,
            rule="include_min_generation_period",
            profile="<system available generation from selected technologies>",
            metadata=None,
            day_position=minimum_day_position,
            seasons=seasons,
            hour="",
            metric_value=float(daily_generation_gwh[minimum_day_position]),
            profile_in_clustering=all(
                column in selected_columns for column in used_profiles
            ),
            table="ExistingCapacity + CapacityFactor",
            profile_count=max(0, len(used_profiles) - len(regions)),
            regions=" | ".join(regions),
            periods=reference_period,
            demand_name="",
            tech=" | ".join(sorted(used_technologies)),
        )
        minimum_record.update(
            {
                "metric_unit": "GWh",
                "generation_period_gwh": float(
                    daily_generation_gwh[minimum_day_position]
                ),
            }
        )
        records.append(minimum_record)
    return records


def build_net_load_ramp_records(
    database: Path,
    profiles: pd.DataFrame,
    selected_columns: list[str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    engine: str,
    selection_method: str,
    config: dict[str, Any],
    calendar: dict[str, Any],
) -> list[dict[str, Any]]:
    direction = str(config.get("direction", "absolute")).lower()
    if direction not in {"absolute", "upward", "downward"}:
        raise ValueError(
            "include_max_net_load_ramp_day.direction must be absolute, "
            "upward, or downward."
        )
    demand_config = config.get("demand", {})
    generation_config = config.get("existing_generation", {})
    demand_profile_table = demand_config.get(
        "profile_table", "DemandSpecificDistribution"
    )
    demand_commodity = str(demand_config.get("commodity", "D_ELEC"))
    demand_columns = [
        column
        for column, item in metadata.items()
        if item.logical_table == demand_profile_table
        and str(profile_key(item, "demand_name")) == demand_commodity
    ]
    if not demand_columns:
        raise ValueError(
            f"No {demand_profile_table} profiles were found for demand "
            f"commodity {demand_commodity}."
        )
    available_periods = sorted(
        {
            profile_key(metadata[column], "periods")
            for column in demand_columns
            if profile_key(metadata[column], "periods") not in {None, ""}
        },
        key=lambda value: int(value),
    )
    reference_period = resolve_reference_period(
        config.get("reference_period", "first_model_period"),
        available_periods,
        "net-load ramp",
    )
    available_regions = sorted(
        {
            str(profile_key(metadata[column], "regions"))
            for column in demand_columns
            if str(profile_key(metadata[column], "periods"))
            == str(reference_period)
        }
    )
    configured_regions = config.get("regions", "all")
    if configured_regions == "all":
        regions = available_regions
    elif isinstance(configured_regions, list):
        regions = [str(region) for region in configured_regions]
    else:
        raise ValueError(
            "include_max_net_load_ramp_day.regions must be all or a list."
        )
    missing_regions = sorted(set(regions) - set(available_regions))
    if missing_regions:
        raise ValueError(
            "Net-load-ramp regions were not found in the configured demand "
            f"profiles: {missing_regions}"
        )

    technologies = parse_generation_technologies(generation_config)
    if not technologies:
        raise ValueError(
            "include_max_net_load_ramp_day.existing_generation.technologies "
            "must contain at least one technology."
        )
    segfrac = load_hourly_segfrac(
        database,
        demand_config.get("segfrac", {}),
        seasons,
        hours,
        calendar["season_format"],
        int(calendar["base_year"]),
    )
    calendar_hours = len(seasons) * len(hours)
    records: list[dict[str, Any]] = []
    region_groups = (
        [regions]
        if bool(config.get("aggregate_regions", False))
        else [[region] for region in regions]
    )
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        for region_group in region_groups:
            demand_gw = np.zeros(len(profiles), dtype=float)
            generation_gw = np.zeros(len(profiles), dtype=float)
            demand_profiles: list[str] = []
            generation_profiles: list[str] = []
            technology_labels: set[str] = set()
            for region in region_group:
                matching_demand_columns = [
                    column
                    for column in demand_columns
                    if str(profile_key(metadata[column], "regions")) == region
                    and str(profile_key(metadata[column], "periods"))
                    == str(reference_period)
                ]
                if len(matching_demand_columns) != 1:
                    raise ValueError(
                        "Net-load ramp requires exactly one DSD profile for "
                        f"region={region}, period={reference_period}, "
                        f"commodity={demand_commodity}; found "
                        f"{len(matching_demand_columns)}."
                    )
                demand_column = matching_demand_columns[0]
                annual_energy_gwh = load_annual_demand_gwh(
                    connection,
                    demand_config,
                    region,
                    reference_period,
                    demand_commodity,
                )
                demand_gw += (
                    annual_energy_gwh
                    * profiles[demand_column].to_numpy(dtype=float)
                    / (segfrac * calendar_hours)
                )
                demand_profiles.append(demand_column)
                for technology in technologies:
                    technology_name = technology["tech"]
                    capacity_gw = load_existing_capacity_gw(
                        connection,
                        generation_config,
                        region,
                        technology_name,
                    )
                    if capacity_gw <= 0:
                        continue
                    cf_table = technology.get(
                        "capacity_factor_table",
                        generation_config.get(
                            "capacity_factor_table", "CapacityFactorTech"
                        ),
                    )
                    cf_columns = [
                        column
                        for column, item in metadata.items()
                        if item.logical_table == cf_table
                        and str(profile_key(item, "regions")) == region
                        and str(profile_key(item, "tech")) == technology_name
                    ]
                    if len(cf_columns) != 1:
                        raise ValueError(
                            "Expected exactly one capacity-factor profile for "
                            f"region={region}, technology={technology_name}, "
                            f"table={cf_table}; found {len(cf_columns)}."
                        )
                    cf_column = cf_columns[0]
                    generation_gw += (
                        capacity_gw
                        * profiles[cf_column].to_numpy(dtype=float)
                    )
                    generation_profiles.append(cf_column)
                    technology_labels.add(technology_name)

            net_load_gw = demand_gw - generation_gw
            daily_net_load = net_load_gw.reshape(len(seasons), len(hours))
            ramp_values = calculate_ramps(daily_net_load, direction)
            day_position = int(np.argmax(ramp_values.max(axis=1)))
            ramp_position = int(np.argmax(ramp_values[day_position]))
            before = day_position * len(hours) + ramp_position
            after = before + 1
            record = extreme_record(
                engine=engine,
                selection_method=selection_method,
                rule="include_max_net_load_ramp_day",
                profile="<actual demand - available existing generation>",
                metadata=None,
                day_position=day_position,
                seasons=seasons,
                hour=f"{hours[ramp_position]}->{hours[ramp_position + 1]}",
                metric_value=float(ramp_values[day_position, ramp_position]),
                profile_in_clustering=(
                    all(column in selected_columns for column in demand_profiles)
                    and all(
                        column in selected_columns
                        for column in generation_profiles
                    )
                ),
                table=(
                    f"{demand_profile_table} + ExistingCapacity + "
                    "CapacityFactor"
                ),
                profile_count=len(demand_profiles) + len(generation_profiles),
                regions=" | ".join(region_group),
                periods=reference_period,
                demand_name=demand_commodity,
                tech=" | ".join(sorted(technology_labels)),
            )
            record.update(
                {
                    "ramp_direction": direction,
                    "metric_unit": "GW",
                    "demand_before_gw": float(demand_gw[before]),
                    "demand_after_gw": float(demand_gw[after]),
                    "generation_before_gw": float(generation_gw[before]),
                    "generation_after_gw": float(generation_gw[after]),
                    "net_load_before_gw": float(net_load_gw[before]),
                    "net_load_after_gw": float(net_load_gw[after]),
                }
            )
            records.append(record)
    return records


def parse_generation_technologies(config: dict[str, Any]) -> list[dict[str, str]]:
    parsed = []
    for item in config.get("technologies", []):
        if isinstance(item, str):
            parsed.append({"tech": item})
        elif isinstance(item, dict) and item.get("tech"):
            parsed.append({key: str(value) for key, value in item.items()})
        else:
            raise ValueError(
                "Each existing_generation technology must be a technology "
                "name or a mapping containing tech."
            )
    return parsed


def load_hourly_segfrac(
    database: Path,
    config: dict[str, Any],
    seasons: list[str],
    hours: list[str],
    canonical_season_format: str,
    base_year: int,
) -> np.ndarray:
    mode = config.get("mode", "database")
    if mode == "uniform_full_year":
        return np.full(
            len(seasons) * len(hours),
            1.0 / (len(seasons) * len(hours)),
        )
    if mode == "calendar":
        with sqlite3.connect(
            f"file:{database.resolve()}?mode=ro", uri=True
        ) as connection:
            season_weights = dict(
                connection.execute(
                    "SELECT season, segment_fraction FROM time_season"
                ).fetchall()
            )
            hour_weights = dict(
                connection.execute(
                    "SELECT tod, hours FROM time_of_day"
                ).fetchall()
            )
        missing_seasons = [season for season in seasons if season not in season_weights]
        missing_hours = [hour for hour in hours if hour not in hour_weights]
        if missing_seasons or missing_hours:
            raise ValueError(
                "Temoa v4 calendar weights are incomplete. Missing seasons: "
                f"{missing_seasons[:10]}; missing hours: {missing_hours[:10]}."
            )
        total_hour_weight = sum(float(hour_weights[hour]) for hour in hours)
        if total_hour_weight <= 0:
            raise ValueError("Temoa v4 time_of_day.hours must sum to a positive value.")
        values = np.asarray(
            [
                float(season_weights[season])
                * float(hour_weights[hour])
                / total_hour_weight
                for season in seasons
                for hour in hours
            ],
            dtype=float,
        )
        if np.any(values <= 0) or not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError(
                "Temoa v4 calendar weights must be positive and sum to 1; "
                f"found {values.sum()}."
            )
        return values
    if mode != "database":
        raise ValueError(
            "Net-load demand.segfrac.mode must be database, calendar, or "
            "uniform_full_year."
        )
    table = config.get("table", "SegFrac")
    season_column = config.get("season_column", "season_name")
    hour_column = config.get("hour_column", "time_of_day_name")
    value_column = config.get("value_column", "segfrac")
    values = np.full(len(seasons) * len(hours), np.nan)
    season_positions = {season: position for position, season in enumerate(seasons)}
    hour_positions = {hour: position for position, hour in enumerate(hours)}
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        mapping = build_season_mapping(
            connection,
            table,
            season_column,
            seasons,
            config.get("source_season_format", canonical_season_format),
            canonical_season_format,
            base_year,
        )
        rows = connection.execute(
            f'SELECT "{season_column}", "{hour_column}", "{value_column}" '
            f'FROM "{table}"'
        )
        for raw_season, hour, value in rows:
            season = mapping.get(raw_season)
            if season not in season_positions or hour not in hour_positions:
                continue
            position = (
                season_positions[season] * len(hours) + hour_positions[hour]
            )
            values[position] = float(value)
    if np.isnan(values).any() or np.any(values <= 0):
        raise ValueError(
            "Net-load ramp requires one positive SegFrac value for every hour."
        )
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"Net-load ramp requires SegFrac to sum to 1; found {values.sum()}."
        )
    return values


def load_annual_demand_gwh(
    connection: sqlite3.Connection,
    config: dict[str, Any],
    region: str,
    period: Any,
    commodity: str,
) -> float:
    table = config.get("annual_table", "Demand")
    region_column = config.get("region_column", "regions")
    period_column = config.get("period_column", "periods")
    commodity_column = config.get("commodity_column", "demand_comm")
    value_column = config.get("value_column", "demand")
    unit_column = config.get("unit_column", "demand_units")
    rows = connection.execute(
        f'SELECT "{value_column}", "{unit_column}" FROM "{table}" '
        f'WHERE "{region_column}" = ? AND "{period_column}" = ? '
        f'AND "{commodity_column}" = ?',
        (region, period, commodity),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError(
            f"Expected one annual demand row for region={region}, "
            f"period={period}, commodity={commodity}; found {len(rows)}."
        )
    value, unit = rows[0]
    return energy_to_gwh(float(value), str(unit))


def energy_to_gwh(value: float, unit: str) -> float:
    factors = {
        "PJ": 1_000_000 / 3_600,
        "TWH": 1_000.0,
        "GWH": 1.0,
        "MWH": 0.001,
    }
    normalized = normalize_unit(unit)
    if normalized not in factors:
        raise ValueError(
            f"Unsupported annual demand unit {unit!r}; use PJ, TWh, GWh, or MWh."
        )
    return value * factors[normalized]


def load_existing_capacity_gw(
    connection: sqlite3.Connection,
    config: dict[str, Any],
    region: str,
    technology: str,
) -> float:
    table = config.get("capacity_table", "ExistingCapacity")
    region_column = config.get("region_column", "regions")
    tech_column = config.get("tech_column", "tech")
    value_column = config.get("value_column", "exist_cap")
    unit_column = config.get("unit_column", "exist_cap_units")
    rows = connection.execute(
        f'SELECT "{value_column}", "{unit_column}" FROM "{table}" '
        f'WHERE "{region_column}" = ? AND "{tech_column}" = ?',
        (region, technology),
    ).fetchall()
    return sum(capacity_to_gw(float(value), str(unit)) for value, unit in rows)


def capacity_to_gw(value: float, unit: str) -> float:
    factors = {"GW": 1.0, "MW": 0.001, "KW": 0.000001}
    normalized = normalize_unit(unit)
    if normalized not in factors:
        raise ValueError(
            f"Unsupported existing-capacity unit {unit!r}; use GW, MW, or kW."
        )
    return value * factors[normalized]


def normalize_unit(unit: str) -> str:
    return unit.strip().upper().replace("(", "").replace(")", "").strip()


def build_pyomo_representatives(
    profiles: pd.DataFrame,
    selected_day_ids: list[int],
    hours_per_day: int,
) -> pd.DataFrame:
    frames = []
    for cluster, day_id in enumerate(selected_day_ids):
        start = (day_id - 1) * hours_per_day
        frame = profiles.iloc[start : start + hours_per_day].copy()
        frame.index = pd.MultiIndex.from_product(
            [[cluster], range(hours_per_day)],
            names=["cluster", "timestep"],
        )
        frames.append(frame)
    return pd.concat(frames)


def assign_days_to_pyomo_representatives(
    prepared,
    attributes,
    selected_day_ids: list[int],
) -> list[int]:
    feature_parts = []
    for attribute in attributes:
        if attribute.weight <= 0:
            continue
        values = prepared.daily_profiles[attribute.name]
        feature_parts.append(
            values * np.sqrt(float(attribute.weight))
        )
    features = np.concatenate(feature_parts, axis=1)
    day_to_position = {
        day_id: position
        for position, day_id in enumerate(prepared.day_labels)
    }
    representative_features = np.stack(
        [features[day_to_position[day_id]] for day_id in selected_day_ids]
    )
    assignments = []
    for values in features:
        distances = np.sum(
            (representative_features - values.reshape(1, -1)) ** 2,
            axis=1,
        )
        assignments.append(int(np.argmin(distances)))
    for cluster, day_id in enumerate(selected_day_ids):
        assignments[day_to_position[day_id]] = cluster
    return assignments


def calculate_pyomo_accuracy(
    profiles: pd.DataFrame,
    representatives: pd.DataFrame,
    cluster_assignments: list[int],
    cluster_weights: dict[int, float],
) -> AccuracySummary:
    n_days = len(cluster_assignments)
    hours_per_day = len(representatives.loc[0])
    source = profiles.to_numpy(dtype=float).reshape(
        n_days, hours_per_day, len(profiles.columns)
    )
    representative_values = np.stack(
        [
            representatives.loc[cluster].to_numpy(dtype=float)
            for cluster in cluster_weights
        ]
    )
    reconstructed = representative_values[
        np.asarray(cluster_assignments, dtype=int)
    ]
    errors = source - reconstructed
    rmse = np.sqrt(np.mean(errors**2, axis=(0, 1)))
    mae = np.mean(np.abs(errors), axis=(0, 1))

    duration_rmse = []
    target_length = n_days * hours_per_day
    for column_position in range(len(profiles.columns)):
        original = np.sort(source[:, :, column_position].reshape(-1))[::-1]
        values = []
        weights = []
        for cluster, weight in cluster_weights.items():
            values.extend(
                representative_values[cluster, :, column_position].tolist()
            )
            weights.extend([float(weight)] * hours_per_day)
        values_array = np.asarray(values, dtype=float)
        weights_array = np.asarray(weights, dtype=float)
        order = np.argsort(values_array)[::-1]
        sorted_values = values_array[order]
        cumulative = np.cumsum(weights_array[order]) / np.sum(weights_array)
        shares = np.arange(1, target_length + 1) / target_length
        positions = np.searchsorted(cumulative, shares, side="left")
        positions = np.minimum(positions, len(sorted_values) - 1)
        approximated = sorted_values[positions]
        duration_rmse.append(
            float(np.sqrt(np.mean((original - approximated) ** 2)))
        )

    index = pd.Index(profiles.columns, name="attribute")
    return AccuracySummary(
        rmse=pd.Series(rmse, index=index),
        mae=pd.Series(mae, index=index),
        rmse_duration=pd.Series(duration_rmse, index=index),
    )


def build_representation(data, Distribution, MinMaxMean):
    if data is None or isinstance(data, str):
        return data
    representation_type = data.get("type", "mean")
    if representation_type == "distribution":
        return Distribution(
            scope=data.get("scope", "cluster"),
            preserve_minmax=bool(data.get("preserve_minmax", False)),
        )
    if representation_type == "minmax_mean":
        return MinMaxMean(
            max_columns=data.get("max_columns", []),
            min_columns=data.get("min_columns", []),
        )
    return representation_type


def map_representative_source_days(
    representatives: pd.DataFrame,
    cluster_centers: tuple[int, ...] | None,
    seasons: list[str],
) -> dict[int, str]:
    cluster_ids = list(
        dict.fromkeys(representatives.index.get_level_values(0))
    )
    if cluster_centers is None or len(cluster_centers) != len(cluster_ids):
        raise ValueError(
            "TSAM did not return one historical source-day index per "
            "representative cluster. Use representation.type: medoid."
        )
    source_days = {
        cluster: seasons[int(cluster_centers[position])]
        for position, cluster in enumerate(cluster_ids)
    }
    if len(set(source_days.values())) != len(source_days):
        duplicates = sorted(
            {
                day
                for day in source_days.values()
                if list(source_days.values()).count(day) > 1
            }
        )
        raise ValueError(
            "Multiple clusters selected the same historical date, so unique "
            f"ACES season labels cannot be created: {duplicates}"
        )
    return source_days


def write_reduced_database(
    source_database: Path,
    config: dict[str, Any],
    base_directory: Path,
    representatives: pd.DataFrame,
    cluster_weights: dict[int, float],
    representative_source_days: dict[int, str],
    metadata: dict[str, ProfileMetadata],
    seasons: list[str],
    hours: list[str],
    cluster_assignments: list[int],
) -> Path:
    project = config["project"]
    output = resolve_path(base_directory, project["output_database"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not project.get("overwrite_output", False):
            raise FileExistsError(
                f"Output database already exists: {output}. "
                "Set project.overwrite_output: true to replace it."
            )
        output.unlink()
    template_setting = project.get("output_template_database")
    template_database = (
        resolve_path(base_directory, template_setting)
        if template_setting
        else source_database
    )
    if not template_database.exists():
        raise FileNotFoundError(
            f"Configured output template does not exist: {template_database}"
        )
    shutil.copy2(template_database, output)

    representatives = representatives.copy()
    if not isinstance(representatives.index, pd.MultiIndex):
        raise ValueError("TSAM representatives do not have cluster/timestep index.")
    cluster_ids = list(dict.fromkeys(representatives.index.get_level_values(0)))
    labels = {
        cluster: representative_source_days[cluster] for cluster in cluster_ids
    }
    weight_config = config["aces"]["weights"]
    total_days = len(seasons)
    if len(cluster_assignments) != total_days:
        raise ValueError(
            "Cluster assignments must contain one entry per source day; "
            f"found {len(cluster_assignments)} for {total_days} days."
        )
    unknown_clusters = sorted(
        {int(cluster) for cluster in cluster_assignments} - set(cluster_ids)
    )
    if unknown_clusters:
        raise ValueError(
            "Cluster assignments reference missing representative clusters: "
            f"{unknown_clusters}"
        )

    with sqlite3.connect(output) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        calendar = config["aces"]["calendar"]
        schema = config["aces"].get("schema", "aces_legacy")
        if schema == "temoa_v4":
            sequential = config["aces"]["sequential_time"]
            connection.execute(f'DELETE FROM "{sequential["table"]}"')
        connection.execute(f'DELETE FROM "{calendar["season_table"]}"')
        if weight_config.get("mode", "hourly_table") == "season_fraction":
            season_rows = [
                (
                    position,
                    labels[cluster],
                    float(cluster_weights[cluster]) / total_days,
                    f"Representative day weight={float(cluster_weights[cluster])}",
                )
                for position, cluster in enumerate(cluster_ids, start=1)
            ]
            connection.executemany(
                f'INSERT INTO "{calendar["season_table"]}" '
                f'("{weight_config["sequence_column"]}", '
                f'"{calendar["season_column"]}", '
                f'"{weight_config["value_column"]}", '
                f'"{weight_config["notes_column"]}") VALUES (?, ?, ?, ?)',
                season_rows,
            )
            width = max(3, len(str(total_days)))
            sequential_rows = [
                (
                    position,
                    f"S{position:0{width}d}",
                    labels[int(cluster)],
                    1.0 / total_days,
                    "Original chronological day mapped to representative day",
                )
                for position, cluster in enumerate(cluster_assignments, start=1)
            ]
            connection.executemany(
                f'INSERT INTO "{sequential["table"]}" '
                f'("{sequential["sequence_column"]}", '
                f'"{sequential["label_column"]}", '
                f'"{sequential["season_column"]}", '
                f'"{sequential["value_column"]}", '
                f'"{sequential["notes_column"]}") VALUES (?, ?, ?, ?, ?)',
                sequential_rows,
            )
            prune_v4_season_dependent_tables(
                connection=connection,
                selected_seasons=set(labels.values()),
                rewritten_tables={item.table for item in metadata.values()},
                excluded_tables={
                    calendar["season_table"],
                    sequential["table"],
                },
            )
        else:
            connection.executemany(
                f'INSERT INTO "{calendar["season_table"]}" '
                f'("{calendar["season_column"]}") VALUES (?)',
                [(labels[cluster],) for cluster in cluster_ids],
            )
            connection.execute(f'DELETE FROM "{weight_config["table"]}"')
            segfrac_rows = []
            for cluster in cluster_ids:
                weight = float(cluster_weights[cluster])
                for hour in hours:
                    segfrac_rows.append(
                        (
                            labels[cluster],
                            hour,
                            weight / (total_days * len(hours)),
                            f"Representative day weight={weight}",
                        )
                    )
            connection.executemany(
                f'INSERT INTO "{weight_config["table"]}" '
                f'("{weight_config["season_column"]}", '
                f'"{weight_config["hour_column"]}", '
                f'"{weight_config["value_column"]}", '
                f'"{weight_config["notes_column"]}") VALUES (?, ?, ?, ?)',
                segfrac_rows,
            )

        if weight_config.get("create_audit_table", True):
            audit_table = weight_config.get(
                "audit_table", "RepresentativeDayWeight"
            )
            connection.execute(f'DROP TABLE IF EXISTS "{audit_table}"')
            connection.execute(
                f'CREATE TABLE "{audit_table}" '
                "(representative_day TEXT PRIMARY KEY, weight REAL NOT NULL)"
            )
            connection.executemany(
                f'INSERT INTO "{audit_table}" VALUES (?, ?)',
                [
                    (labels[cluster], float(cluster_weights[cluster]))
                    for cluster in cluster_ids
                ],
            )

        table_groups: dict[str, list[tuple[str, ProfileMetadata]]] = {}
        for column, item in metadata.items():
            table_groups.setdefault(item.table, []).append((column, item))
        for table, items in table_groups.items():
            connection.execute(f'DELETE FROM "{table}"')
            insert_temporal_profiles(
                connection,
                table,
                items,
                representatives,
                cluster_ids,
                labels,
                cluster_weights,
                hours,
            )
        connection.commit()

    validate_output_database(output, config)
    if project.get("compact_output", True):
        print("Compacting reduced SQLite database...")
        with sqlite3.connect(output) as connection:
            connection.execute("VACUUM")
    return output


def prune_v4_season_dependent_tables(
    connection: sqlite3.Connection,
    selected_seasons: set[str],
    rewritten_tables: set[str],
    excluded_tables: set[str],
) -> None:
    """Remove source-date rows that cannot reference the reduced v4 calendar."""
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    ]
    placeholders = ", ".join("?" for _ in selected_seasons)
    for table in tables:
        if table in excluded_tables or table in rewritten_tables:
            continue
        columns = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        if "season" not in columns:
            continue
        if table.lower().startswith("output_"):
            connection.execute(f'DELETE FROM "{table}"')
        elif selected_seasons:
            connection.execute(
                f'DELETE FROM "{table}" WHERE "season" NOT IN ({placeholders})',
                tuple(sorted(selected_seasons)),
            )


def insert_temporal_profiles(
    connection: sqlite3.Connection,
    table: str,
    items: list[tuple[str, ProfileMetadata]],
    representatives: pd.DataFrame,
    cluster_ids: list[int],
    labels: dict[int, str],
    cluster_weights: dict[int, float],
    hours: list[str],
) -> None:
    first = items[0][1]
    insert_columns = (
        first.key_columns
        + [first.season_column, first.hour_column, first.value_column]
        + first.preserve_columns
    )
    placeholders = ", ".join("?" for _ in insert_columns)
    quoted = ", ".join(f'"{column}"' for column in insert_columns)
    query = f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
    rows = []
    for column, item in items:
        if column not in representatives.columns:
            continue
        values = representatives[column]
        denominator = None
        if item.output_rule == "weighted_and_normalized_to_one":
            denominator = sum(
                float(cluster_weights[cluster])
                * float(values.loc[cluster].sum())
                for cluster in cluster_ids
            )
            if denominator <= 0:
                raise ValueError(f"Cannot normalize nonpositive DSD profile: {column}")
        for cluster in cluster_ids:
            cluster_values = values.loc[cluster].to_numpy()
            for timestep, hour in enumerate(hours):
                value = float(cluster_values[timestep])
                if denominator is not None:
                    value = (
                        value * float(cluster_weights[cluster]) / denominator
                    )
                rows.append(
                    item.key_values
                    + (labels[cluster], hour, value)
                    + item.preserve_values
                )
    connection.executemany(query, rows)


def validate_output_database(database: Path, config: dict[str, Any]) -> None:
    validation = config.get("validation", {})
    weight_config = config["aces"]["weights"]
    with sqlite3.connect(database) as connection:
        if validation.get("require_segfrac_sum_one", True):
            total = connection.execute(
                f'SELECT SUM("{weight_config["value_column"]}") '
                f'FROM "{weight_config["table"]}"'
            ).fetchone()[0]
            tolerance = float(validation.get("segfrac_sum_tolerance", 1e-8))
            if total is None or abs(float(total) - 1.0) > tolerance:
                raise ValueError(f"Output SegFrac sum is {total}, expected 1.")
        if validation.get("require_dsd_sum_one", True):
            dsd_config = next(
                (
                    item
                    for item in config["aces"]["temporal_tables"]
                    if item.get("logical_table", item["table"])
                    == "DemandSpecificDistribution"
                    and item.get("enabled", True)
                ),
                None,
            )
            if dsd_config is not None:
                keys = dsd_config["key_columns"]
                quoted_keys = ", ".join(f'"{column}"' for column in keys)
                rows = connection.execute(
                    f'SELECT {quoted_keys}, SUM("{dsd_config["value_column"]}") '
                    f'FROM "{dsd_config["table"]}" GROUP BY {quoted_keys} '
                    f'HAVING ABS(SUM("{dsd_config["value_column"]}") - 1.0) > ?',
                    (float(validation.get("dsd_sum_tolerance", 1e-8)),),
                ).fetchall()
                if rows:
                    raise ValueError(
                        f"Output DSD profiles do not sum to 1: {rows[:5]}"
                    )
        if config["aces"].get("schema") == "temoa_v4":
            sequential = config["aces"]["sequential_time"]
            sequential_total = connection.execute(
                f'SELECT SUM("{sequential["value_column"]}") '
                f'FROM "{sequential["table"]}"'
            ).fetchone()[0]
            tolerance = float(validation.get("segfrac_sum_tolerance", 1e-8))
            if sequential_total is None or abs(float(sequential_total) - 1.0) > tolerance:
                raise ValueError(
                    "Output sequential-season fractions sum to "
                    f"{sequential_total}, expected 1."
                )
            foreign_key_errors = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if foreign_key_errors:
                raise ValueError(
                    "Output SQLite foreign-key validation failed: "
                    f"{foreign_key_errors[:5]}"
                )
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Output SQLite quick_check failed: {integrity}")


def write_audits(
    output_directory: Path,
    seasons: list[str],
    config: dict[str, Any],
    representative_source_days: dict[int, str],
    cluster_assignments: list[int],
    cluster_weights: dict[int, float],
    accuracy: AccuracySummary | Any,
    engine: str,
    forced_day_ids: list[int],
    objective_value: float | None,
    extreme_records: list[dict[str, Any]],
    selected_columns: list[str],
    column_weights: dict[str, float],
    metadata: dict[str, ProfileMetadata],
    iteration_history: pd.DataFrame | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(
        {
            "original_day": seasons,
            "cluster": cluster_assignments,
            "representative_day": [
                representative_source_days[cluster]
                for cluster in cluster_assignments
            ],
        }
    )
    assignment_counts = pd.Series(cluster_assignments).value_counts()
    forced_source_days = {
        record["source_day"]
        for record in extreme_records
        if record.get("source_day")
    }
    reason_by_day: dict[str, list[str]] = {}
    for record in extreme_records:
        source_day = record.get("source_day")
        if not source_day:
            continue
        reason = str(record["rule"])
        if record.get("profile"):
            reason = f'{reason}: {record["profile"]}'
        reason_by_day.setdefault(source_day, []).append(reason)

    weights = pd.DataFrame(
        {
            "cluster": list(cluster_weights),
            "source_day": [
                representative_source_days[cluster]
                for cluster in cluster_weights
            ],
            "weight": list(cluster_weights.values()),
            "nearest_assignment_count": [
                int(assignment_counts.get(cluster, 0))
                for cluster in cluster_weights
            ],
        }
    )
    weights["is_forced_extreme"] = weights["source_day"].isin(
        forced_source_days
    )
    weights["forced_by"] = weights["source_day"].map(
        lambda day: " ; ".join(
            dict.fromkeys(reason_by_day.get(day, []))
        )
    )

    extreme_columns = [
        "engine",
        "selection_method",
        "rule",
        "extreme_scope",
        "generation_group",
        "profile",
        "table",
        "regions",
        "periods",
        "demand_name",
        "tech",
        "profile_keys",
        "day_id",
        "source_day",
        "hour",
        "metric_value",
        "metric_unit",
        "ramp_direction",
        "demand_before_gw",
        "demand_after_gw",
        "generation_before_gw",
        "generation_after_gw",
        "net_load_before_gw",
        "net_load_after_gw",
        "demand_at_peak_gw",
        "generation_at_peak_gw",
        "net_load_at_peak_gw",
        "generation_period_gwh",
        "profile_in_clustering",
        "profile_count",
    ]
    extremes = pd.DataFrame(extreme_records, columns=extreme_columns)
    source_day_to_cluster = {
        source_day: cluster
        for cluster, source_day in representative_source_days.items()
    }
    if not extremes.empty:
        extremes["selected_as_representative"] = extremes[
            "source_day"
        ].isin(source_day_to_cluster)
        extremes["selected_cluster"] = extremes["source_day"].map(
            source_day_to_cluster
        )
        extremes["representative_weight"] = extremes[
            "selected_cluster"
        ].map(cluster_weights)
    else:
        extremes["selected_as_representative"] = pd.Series(dtype=bool)
        extremes["selected_cluster"] = pd.Series(dtype=float)
        extremes["representative_weight"] = pd.Series(dtype=float)

    profile_rows = []
    for column in selected_columns:
        item = metadata.get(column)
        keyed = (
            {
                name: value
                for name, value in zip(
                    item.key_columns, item.key_values
                )
            }
            if item is not None
            else {}
        )
        profile_rows.append(
            {
                "profile": column,
                "clustering_weight": float(column_weights[column]),
                "table": item.table if item is not None else "extra_attribute",
                "regions": keyed.get("regions", ""),
                "periods": keyed.get("periods", ""),
                "demand_name": keyed.get("demand_name", ""),
                "tech": keyed.get("tech", ""),
                "profile_keys": " | ".join(
                    f"{name}={value}" for name, value in keyed.items()
                ),
            }
        )
    profile_weights = pd.DataFrame(profile_rows)

    accuracy_table = pd.concat(
        {
            "rmse": accuracy.rmse,
            "mae": accuracy.mae,
            "rmse_duration": accuracy.rmse_duration,
        },
        axis=1,
    ).reset_index()
    run_summary = pd.DataFrame(
        [
            {
                "engine": engine,
                "requested_representative_days": config["clustering"][
                    "n_representative_days"
                ],
                "created_representative_days": len(cluster_weights),
                "original_days": len(seasons),
                "total_representative_weight": float(
                    sum(cluster_weights.values())
                ),
                "unique_forced_extreme_days": len(forced_source_days),
                "extreme_reason_rows": len(extremes),
                "pyomo_objective_value": objective_value,
                "pyomo_objective_definition": (
                    "Profile-weighted absolute error between full-year and "
                    "representative-day exceedance shares across duration-curve bins"
                ),
                "important_note": (
                    "Nearest assignment counts are diagnostic. ACES SegFrac "
                    "uses representative weights."
                ),
            }
        ]
    )
    configuration = pd.DataFrame(
        flatten_configuration(config), columns=["parameter", "value"]
    )

    assignments.to_csv(output_directory / "cluster_assignments.csv", index=False)
    weights.to_csv(output_directory / "representative_day_weights.csv", index=False)
    extremes.to_csv(
        output_directory / "extreme_day_provenance.csv", index=False
    )
    if config.get("outputs", {}).get("write_accuracy_metrics", True):
        accuracy_table.to_csv(
            output_directory / f"{engine}_accuracy.csv", index=False
        )
    if engine.startswith("pyomo"):
        forced = pd.DataFrame(
            {
                "day_id": forced_day_ids,
                "source_day": [
                    seasons[day_id - 1] for day_id in forced_day_ids
                ],
            }
        )
        forced.to_csv(output_directory / "pyomo_forced_days.csv", index=False)
        pd.DataFrame(
            [
                {
                    "engine": engine,
                    "objective_value": objective_value,
                    "n_forced_days": len(forced_day_ids),
                }
            ]
        ).to_csv(output_directory / "pyomo_summary.csv", index=False)
        if iteration_history is not None and not iteration_history.empty:
            iteration_history.to_csv(
                output_directory / "pyomo_iteration_history.csv", index=False
            )
    if config.get("outputs", {}).get("write_excel_audit", True):
        excel_path = output_directory / "representative_day_audit.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            run_summary.to_excel(
                writer, sheet_name="Run_Summary", index=False
            )
            weights.to_excel(
                writer, sheet_name="Representative_Days", index=False
            )
            extremes.to_excel(
                writer, sheet_name="Forced_Extremes", index=False
            )
            assignments.to_excel(
                writer, sheet_name="Day_Assignments", index=False
            )
            profile_weights.to_excel(
                writer, sheet_name="Profile_Weights", index=False
            )
            accuracy_table.to_excel(
                writer, sheet_name="Accuracy", index=False
            )
            if iteration_history is not None and not iteration_history.empty:
                iteration_history.to_excel(
                    writer, sheet_name="Pyomo_Iterations", index=False
                )
            configuration.to_excel(
                writer, sheet_name="Configuration", index=False
            )
            format_audit_workbook(writer)


def flatten_configuration(
    value: Any, prefix: str = ""
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_configuration(child, path))
    elif isinstance(value, list):
        if not value:
            rows.append((prefix, "[]"))
        else:
            for position, child in enumerate(value):
                rows.extend(
                    flatten_configuration(
                        child, f"{prefix}[{position}]"
                    )
                )
    else:
        rows.append((prefix, value))
    return rows


def format_audit_workbook(writer: pd.ExcelWriter) -> None:
    from openpyxl.styles import Font, PatternFill

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        if worksheet.max_row > 1 and worksheet.max_column > 0:
            worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
        for column_cells in worksheet.columns:
            width = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = min(max(width + 2, 12), 60)


def inspect_database(database: Path) -> pd.DataFrame:
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
            connection,
        )
        rows = []
        for table in tables["name"]:
            columns = pd.read_sql_query(f'PRAGMA table_info("{table}")', connection)
            rows.extend(
                {
                    "table": table,
                    "column": column["name"],
                    "type": column["type"],
                }
                for _, column in columns.iterrows()
            )
    return pd.DataFrame(rows)


def sort_seasons(values: list[str], date_format: str, year: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: pd.to_datetime(
            f"{year}-{value}", format=f"%Y-{date_format}"
        ),
    )


def parse_hour(value: str) -> int:
    return int(str(value).replace("H", ""))


def build_datetime_index(
    seasons: list[str], hours: list[str], calendar: dict[str, Any]
) -> pd.DatetimeIndex:
    timestamps = []
    year = int(calendar["base_year"])
    date_format = calendar["season_format"]
    for season in seasons:
        day = pd.to_datetime(
            f"{year}-{season}", format=f"%Y-{date_format}"
        )
        timestamps.extend(day + pd.Timedelta(hours=parse_hour(hour)) for hour in hours)
    return pd.DatetimeIndex(timestamps, name="timestamp")


def profile_name(table: str, key_values: tuple[Any, ...]) -> str:
    return "|".join([table, *(str(value) for value in key_values)])


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


if __name__ == "__main__":
    main()

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from run_aces_tsam import (
    ProfileMetadata,
    configure_database_schema,
    detect_database_schema,
    write_reduced_database,
)


def create_v4_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (
                element TEXT PRIMARY KEY, value INTEGER, notes TEXT
            );
            INSERT INTO metadata VALUES ('DB_MAJOR', 4, 'test');
            CREATE TABLE time_of_day (
                sequence INTEGER UNIQUE, tod TEXT PRIMARY KEY,
                hours REAL NOT NULL, notes TEXT
            );
            CREATE TABLE time_season (
                sequence INTEGER UNIQUE, season TEXT PRIMARY KEY,
                segment_fraction REAL NOT NULL, notes TEXT
            );
            CREATE TABLE time_season_sequential (
                sequence INTEGER UNIQUE, seas_seq TEXT PRIMARY KEY,
                season TEXT REFERENCES time_season(season),
                segment_fraction REAL NOT NULL, notes TEXT
            );
            CREATE TABLE demand_specific_distribution (
                region TEXT, period INTEGER,
                season TEXT REFERENCES time_season(season),
                tod TEXT REFERENCES time_of_day(tod), demand_name TEXT,
                dsd REAL, notes TEXT,
                PRIMARY KEY (region, period, season, tod, demand_name)
            );
            CREATE TABLE limit_seasonal_capacity_factor (
                region TEXT, season TEXT REFERENCES time_season(season),
                tech_or_group TEXT, operator TEXT, factor REAL,
                PRIMARY KEY (region, season, tech_or_group, operator)
            );
            INSERT INTO time_of_day VALUES
                (1, 'H00', 1, NULL), (2, 'H01', 1, NULL);
            INSERT INTO time_season VALUES
                (1, '01-01', 0.5, NULL), (2, '01-02', 0.5, NULL);
            INSERT INTO time_season_sequential VALUES
                (1, 'S001', '01-01', 0.5, NULL),
                (2, 'S002', '01-02', 0.5, NULL);
            INSERT INTO demand_specific_distribution VALUES
                ('R1', 2025, '01-01', 'H00', 'D_ELEC', 0.1, 'test'),
                ('R1', 2025, '01-01', 'H01', 'D_ELEC', 0.2, 'test'),
                ('R1', 2025, '01-02', 'H00', 'D_ELEC', 0.3, 'test'),
                ('R1', 2025, '01-02', 'H01', 'D_ELEC', 0.4, 'test');
            INSERT INTO limit_seasonal_capacity_factor VALUES
                ('R1', '01-01', 'T1', 'le', 0.8),
                ('R1', '01-02', 'T1', 'le', 0.9);
            """
        )


def base_config(output_name: str = "reduced.sqlite") -> dict:
    return {
        "project": {
            "input_database": "source.sqlite",
            "output_database": output_name,
            "output_template_database": None,
            "overwrite_output": True,
            "compact_output": False,
        },
        "aces": {
            "schema": "auto",
            "calendar": {
                "season_table": "time_season",
                "season_column": "t_season",
                "season_format": "%m-%d",
                "base_year": 2023,
                "hour_table": "time_of_day",
                "hour_column": "t_day",
                "hours_per_day": 2,
            },
            "weights": {"create_audit_table": True},
            "temporal_tables": [
                {
                    "table": "DemandSpecificDistribution",
                    "enabled": True,
                    "season_column": "season_name",
                    "source_season_format": "%d-%m",
                    "hour_column": "time_of_day_name",
                    "value_column": "dsd",
                    "key_columns": ["regions", "periods", "demand_name"],
                    "preserve_columns": ["dsd_notes"],
                    "output_rule": "weighted_and_normalized_to_one",
                }
            ],
        },
        "clustering": {"attributes": {}},
        "validation": {
            "require_segfrac_sum_one": True,
            "require_dsd_sum_one": True,
        },
    }


def test_schema_detection_and_v4_mapping(tmp_path: Path) -> None:
    database = tmp_path / "v4.sqlite"
    create_v4_database(database)

    assert detect_database_schema(database) == "temoa_v4"
    config = configure_database_schema(database, base_config())
    assert config["aces"]["schema"] == "temoa_v4"
    assert config["aces"]["calendar"]["season_column"] == "season"
    temporal = config["aces"]["temporal_tables"][0]
    assert temporal["table"] == "demand_specific_distribution"
    assert temporal["logical_table"] == "DemandSpecificDistribution"
    assert temporal["source_season_format"] == "%m-%d"


def test_legacy_schema_remains_config_driven(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE time_season (t_season TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO time_season VALUES ('01-01')")
    config = base_config()

    assert detect_database_schema(database) == "aces_legacy"
    configured = configure_database_schema(database, config)
    assert configured["aces"]["schema"] == "aces_legacy"
    assert configured["aces"]["calendar"]["season_column"] == "t_season"
    assert configured["aces"]["weights"]["mode"] == "hourly_table"
    assert configured["aces"]["temporal_tables"][0]["table"] == (
        "DemandSpecificDistribution"
    )


def test_explicit_schema_mismatch_fails(tmp_path: Path) -> None:
    database = tmp_path / "v4.sqlite"
    create_v4_database(database)
    config = base_config()
    config["aces"]["schema"] = "aces_legacy"

    with pytest.raises(ValueError, match="does not match detected schema"):
        configure_database_schema(database, config)


def test_v4_writer_updates_weights_profiles_and_sequence(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    create_v4_database(source)
    config = configure_database_schema(source, base_config())
    column = "DemandSpecificDistribution|R1|2025|D_ELEC"
    representatives = pd.DataFrame(
        {column: [0.1, 0.2]},
        index=pd.MultiIndex.from_product(
            [[0], [0, 1]], names=["cluster", "timestep"]
        ),
    )
    metadata = {
        column: ProfileMetadata(
            table="demand_specific_distribution",
            logical_table="DemandSpecificDistribution",
            key_columns=["region", "period", "demand_name"],
            key_values=("R1", 2025, "D_ELEC"),
            season_column="season",
            hour_column="tod",
            value_column="dsd",
            preserve_columns=["notes"],
            preserve_values=("test",),
            output_rule="weighted_and_normalized_to_one",
        )
    }

    output = write_reduced_database(
        source_database=source,
        config=config,
        base_directory=tmp_path,
        representatives=representatives,
        cluster_weights={0: 2.0},
        representative_source_days={0: "01-01"},
        metadata=metadata,
        seasons=["01-01", "01-02"],
        hours=["H00", "H01"],
        cluster_assignments=[0, 0],
    )

    with sqlite3.connect(output) as connection:
        assert connection.execute(
            "SELECT season, segment_fraction FROM time_season"
        ).fetchall() == [("01-01", 1.0)]
        assert connection.execute(
            "SELECT season FROM time_season_sequential ORDER BY sequence"
        ).fetchall() == [("01-01",), ("01-01",)]
        assert connection.execute(
            "SELECT ROUND(SUM(dsd), 12) FROM demand_specific_distribution"
        ).fetchone()[0] == 1.0
        assert connection.execute(
            "SELECT season, factor FROM limit_seasonal_capacity_factor"
        ).fetchall() == [("01-01", 0.8)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

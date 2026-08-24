-- Back up the database before running this script.
-- SQLite script for the ACES schema.

PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- Every hourly time slice receives an equal fraction.
-- For 366 days and 24 hours, each value is 1 / 8784.
DELETE FROM SegFrac;

INSERT INTO SegFrac (
    season_name,
    time_of_day_name,
    segfrac,
    segfrac_notes
)
SELECT
    season.t_season,
    hour.t_day,
    1.0 / (
        (SELECT COUNT(*) FROM time_season)
        * (SELECT COUNT(*) FROM time_of_day)
    ),
    'Uniform hourly fraction generated from time_season and time_of_day'
FROM time_season AS season
CROSS JOIN time_of_day AS hour;

-- Preserve each existing region/technology pair and its average cost.
DROP TABLE IF EXISTS temp._cost_keys;

CREATE TEMP TABLE _cost_keys AS
SELECT
    regions,
    tech,
    COALESCE(AVG(costvarvar), 1.0) AS average_cost,
    COALESCE(MAX(source), 'Synthetic random test data') AS source,
    COALESCE(MAX(eff_notes), '') AS notes
FROM CostVariableVariable
GROUP BY regions, tech;

-- Generate one test value for every region/technology/season/hour.
-- Values are randomly distributed within +/-10% of the existing average.
DELETE FROM CostVariableVariable;

INSERT INTO CostVariableVariable (
    regions,
    tech,
    season_name,
    time_of_day_name,
    costvarvar,
    source,
    eff_notes
)
SELECT
    key_data.regions,
    key_data.tech,
    season.t_season,
    hour.t_day,
    key_data.average_cost
        * (
            0.90
            + 0.20 * (ABS(RANDOM() % 1000001) / 1000000.0)
        ),
    key_data.source,
    'Synthetic test value within +/-10% of existing average. ' || key_data.notes
FROM _cost_keys AS key_data
CROSS JOIN time_season AS season
CROSS JOIN time_of_day AS hour;

DROP TABLE _cost_keys;
COMMIT;

-- Verification results:
SELECT COUNT(*) AS segfrac_rows, SUM(segfrac) AS segfrac_sum
FROM SegFrac;

SELECT COUNT(*) AS cost_variable_variable_rows
FROM CostVariableVariable;

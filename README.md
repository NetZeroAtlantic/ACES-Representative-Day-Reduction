# ACES Representative-Day Reduction

This project converts a full-year ACES SQLite database into a smaller ACES database containing weighted representative days. It supports two approaches:

- **TSAM clustering:** groups similar daily profiles and selects one historical day from each group.
- **Duration-curve optimization:** selects historical days and calculates their occurrence weights by minimizing duration-curve error. The optimization model is implemented with Pyomo.

The input database is never modified. The reduced database and audit files are written to `output/`.

## Project Layout

| Path | Purpose |
|---|---|
| `README.md` | Project setup, configuration, and method guide. |
| `aces_tsam_8760.yaml` | Main configuration file. Select the engine and edit all run settings here. |
| `run_aces_tsam.py` | Reads ACES tables, runs the selected method, and writes the reduced database. |
| `repday/` | Duration-curve optimization models, candidate reduction, preprocessing, metrics, and plots. |
| `extra_attributes/` | Optional hourly CSV or Excel files containing temperature, humidity, or other clustering inputs. |
| `data/` | Local input databases and test datasets. This folder is excluded from Git. |
| `output/` | Reduced databases and audit files. This folder is excluded from Git. |
| `scripts/inspect_aces_sqlite.py` | Lists tables and columns in an ACES SQLite database. |
| `scripts/generate_example_excel.py` | Generates an example hourly Excel dataset for the standalone `repday` package. |
| `scripts/run_wind_demand_example.py` | Example of using the standalone duration-curve method with Excel data. |
| `requirements.txt` | Python packages required by the main tool. |
| `repday_environment.yml` | Optional Conda environment definition. |
| `setup.py` | Package metadata for installing `repday` locally. |
| `.gitignore` | Prevents databases, outputs, caches, and local files from being uploaded to GitHub. |

## 1. Add the Input Database

Place the full-year ACES SQLite database in `data/`:

```text
Clustering/
  data/
    Offshore_wind_NS-NB.sqlite
```

Set the same path in `aces_tsam_8760.yaml`:

```yaml
project:
  input_database: data/Offshore_wind_NS-NB.sqlite
```

The database must contain the ACES calendar tables and the enabled temporal tables listed under `aces.temporal_tables`. The current calendar configuration expects 366 days and hourly labels `H00` through `H23`.

## 2. Install Dependencies

From the project directory:

```powershell
python -m pip install -r requirements.txt
```

The default environment includes TSAM, Pyomo, HiGHS, pandas, NumPy, scikit-learn, PyYAML, and Excel support. CPLEX or Gurobi may also be used when installed separately.

## 3. Validate Before Running

Validation reads the database and checks the selected engine configuration without creating an output database:

```powershell
python run_aces_tsam.py aces_tsam_8760.yaml --validate-only
```

Use this command after changing table names, profile names, extreme rules, or method settings.

## 4. Run the Reduction

```powershell
python run_aces_tsam.py aces_tsam_8760.yaml
```

The output database path is controlled by:

```yaml
project:
  output_database: output/Offshore_wind_NS-NB_repdays_actual_dates.sqlite
```

`overwrite_output: true` replaces an older output with the same name. It does not modify the input database.

## Choosing an Engine

Set `clustering.engine` to `tsam` or `pyomo`. Only the configuration block for the selected engine is used. Common profile inclusion, exclusion, and weight settings under `clustering.attributes` are used by both engines.

## TSAM Engine

TSAM groups the original days according to the selected hourly ACES profiles. `n_representative_days` is the number of normal clusters requested before additional extreme clusters are added.

### Supported Clustering Methods

| Method | How it groups days | Typical use |
|---|---|---|
| `hierarchical` | Repeatedly merges the most similar groups. | Recommended general-purpose method. Stable and does not need an optimization solver. |
| `kmeans` | Groups days around calculated average centers. | Fast grouping for large inputs. |
| `kmedoids` | Uses an optimization solver to select central historical days. | Direct optimization of historical cluster centers. Slower than hierarchical. |
| `kmaxoids` | Selects dissimilar days near the outside of the dataset. | Stress-condition or diversity studies rather than average-year accuracy. |
| `averaging` | Splits the year into sequential blocks instead of grouping by profile similarity. | Simple baseline only. |
| `contiguous` | Hierarchical clustering restricted to neighboring dates. | Seasonal continuity rather than grouping similar dates from different seasons. |

The ACES writer currently requires:

```yaml
representation:
  type: medoid
```

`medoid` writes one complete historical day for every cluster. TSAM also provides synthetic representations such as `mean`, `distribution`, `distribution_minmax`, and `minmax_mean`, but they are not supported by this ACES writer because a synthetic profile cannot be assigned to one historical source day.

All six TSAM clustering methods can be combined with `medoid`. Recommended combinations are:

```yaml
# Fast and stable
method: hierarchical
representation:
  type: medoid
```

```yaml
# Solver-based historical-day centers
method: kmedoids
representation:
  type: medoid
solver: highs
```

### TSAM Comparison Settings

| Setting | Effect |
|---|---|
| `normalize_column_means: true` | Prevents profiles with large numerical values from dominating the distance calculation. |
| `use_duration_curves: false` | Compares H00 with H00, H01 with H01, and so on. Set `true` to compare sorted daily values instead of hourly timing. |
| `include_period_sums: true` | Includes each 24-hour total when comparing days. Useful for preserving daily energy totals. |

For exact historical medoid values, keep `aggregate.preserve_column_means: false`.

### TSAM Extremes

| Rule | Selected day |
|---|---|
| `max_value` | Day containing the largest hourly value. Example: peak hourly demand. |
| `min_value` | Day containing the smallest hourly value. Example: minimum temperature. |
| `max_period` | Day with the largest 24-hour total. |
| `min_period` | Day with the smallest 24-hour total. Example: sustained low wind. |

The ACES runner supports two TSAM extreme methods:

| Method | Result |
|---|---|
| `append` | Adds each unique extreme as a separate representative, normally with weight 1. Final count can exceed `n_representative_days`. |
| `new_cluster` | Adds each unique extreme as a new center and allows similar days to join it. Final count can exceed `n_representative_days`, and the extreme cluster can have weight greater than 1. |

`replace` is not supported. TSAM can create a hybrid representative in which only the targeted profile comes from the extreme day. That replacement is not retained when the clustering is applied to the complete ACES profile set. The runner stops with a validation error if `replace` is selected.

If 15 normal days and five unique extreme days are configured, `append` or `new_cluster` can produce up to 20 representative days.

### TSAM Settings Required by the ACES Writer

```yaml
period_duration: 24
representation:
  type: medoid
segmentation:
  enabled: false
aggregate:
  preserve_column_means: false
```

If segmentation is enabled, `n_segments` must remain 24 because ACES expects `H00` through `H23`.

## Duration-Curve Optimization Engine

Set `clustering.engine: pyomo`. The model divides every selected profile into `n_bins` value thresholds. At each threshold it compares the share of full-year hours above the threshold with the share represented by the weighted selected days. It minimizes the weighted sum of the absolute differences.

Increasing `n_bins` describes the annual duration curves in more detail but increases model size and runtime.

### Solution Methods and Valid Combinations

| `solution_method` | `formulation` | What is optimized | Valid weight types |
|---|---|---|---|
| `opt` | `milp` | Selects exactly N historical days and their occurrence weights in one model. | Continuous or integer weights. Day selection always requires binary variables. |
| `opt` | `lp_fixed_days` | Uses the exact dates listed in `fixed_day_ids` and optimizes only their weights. | Continuous weights only. `fixed_day_ids` must contain exactly N unique IDs. |
| `hybrid_random_weighting` | `milp` | Randomly selects N historical days, optimizes their weights, repeats, and keeps the lowest-error result. | Continuous weights give an LP per iteration; integer weights give a MIP per iteration. |

Invalid combinations are rejected during validation:

- `hybrid_random_weighting` cannot use `lp_fixed_days`.
- `lp_fixed_days` cannot use integer weights.
- `sampled_candidate_pool_size` applies only to `hybrid_random_weighting`.
- Every forced day must be present in `fixed_day_ids` when using `lp_fixed_days`.

### Candidate Reduction

Candidate reduction is an optional prescreening step:

```text
366 original days -> candidate days -> N final representative days
```

| Method | Behavior |
|---|---|
| `none` | Uses all original days. `candidate_feature_mode` and `n_candidate_days` do not affect the result. |
| `kmeans_nearest_day` | Creates `n_candidate_days` k-means groups and keeps the real day nearest each center. |
| `ward_nearest_day` | Creates `n_candidate_days` Ward groups and keeps the real day nearest each center. |

`extreme_plus_kmeans` and `extreme_plus_ward` are legacy aliases. For ACES, use `kmeans_nearest_day` or `ward_nearest_day` and configure exact extreme profile names under `pyomo.extremes`.

When reduction is enabled, `n_candidate_days` is required and must be between `n_representative_days` and the number of original days.

Candidate feature modes:

| Mode | Day comparison used during prescreening |
|---|---|
| `chronological_daily_profile` | Compares the complete H00-H23 shapes. Recommended default. |
| `daily_duration_curve` | Compares each day's sorted values and ignores the hour at which each value occurred. |
| `hybrid_profile_and_duration` | Uses both chronological profiles and sorted daily values. More detailed and larger. |
| `summary_statistics` | Uses daily mean, minimum, maximum, standard deviation, range, and ramp statistics. Fastest and least detailed. |

### Forced Extremes

Pyomo extreme days count inside `n_representative_days`. If N is 15 and five unique extremes are forced, the method selects those five plus ten other days.

Supported controls are exact profile rules (`max_value`, `min_value`, `max_period`, and `min_period`), system rules (`include_peak_demand_day`, `include_min_wind_day`, and `include_max_daily_ramp_day`), and manual one-based IDs under `forced_day_ids`.

If multiple rules select the same date, it counts once but every reason is retained in the audit output.

## Profile Selection and Weights

The `clustering.attributes` section controls which ACES profiles influence representative-day selection:

- `include.tables`: temporal tables included in the comparison or optimization.
- `include.extra_attributes`: columns loaded from `extra_attributes/`.
- `exclude.regions`, `exclude.technologies`, and `exclude.demands`: remove groups from selection.
- `exclude.columns`: remove exact generated profile names.
- `table_default_weights`: default importance for every profile from a table.
- `column_weight_overrides`: importance assigned to one exact profile.

These values affect day selection only. They do not multiply the capacity factor, demand, price, or efficiency values written to the output database.

## Extra Attributes

Place each optional `.csv` or `.xlsx` file in `extra_attributes/`. It must contain a `timestamp` column matching every input hour:

```csv
timestamp,temperature,humidity
2024-01-01 00:00:00,-4.2,0.81
2024-01-01 01:00:00,-4.5,0.82
```

Enable the columns in the YAML:

```yaml
clustering:
  attributes:
    include:
      extra_attributes: [temperature, humidity]
```

Extra attributes influence day selection but are not written into ACES tables because they do not have a configured ACES destination table.

## Outputs

| Output | Purpose |
|---|---|
| Reduced `.sqlite` database | ACES database containing representative temporal rows. |
| `RepresentativeDayWeight` table | Raw occurrence weight assigned to each representative day. Weights sum to the number of original days. |
| `SegFrac` table | ACES fractions calculated from representative-day weights. Values sum to 1. |
| `representative_day_weights.csv` | Representative day, weight, nearest-assignment count, and extreme status. |
| `cluster_assignments.csv` | Original day, assigned cluster, and representative day. |
| `extreme_day_provenance.csv` | Extreme rule, profile, region, period, technology or demand, selected date, and value. |
| `<engine>_accuracy.csv` | RMSE, MAE, and duration-curve RMSE by profile. |
| `representative_day_audit.xlsx` | Run summary, representative days, forced extremes, assignments, profile weights, accuracy, and configuration. |
| `pyomo_forced_days.csv` | Forced dates for a Pyomo run. |
| `pyomo_summary.csv` | Pyomo objective value and forced-day count. |

For Pyomo runs, `weight` is optimized by the duration-curve model. `nearest_assignment_count` is a diagnostic count obtained by assigning each original day to its closest selected day. The ACES database uses `weight`, not `nearest_assignment_count`.

## Known Limitations

- TSAM output must use `medoid` while the database uses source-day labels.
- TSAM `replace` extremes are not supported; use `append` or `new_cluster`.
- TSAM periods and output segments must remain 24 hours for the current ACES schema.
- Extra attributes affect selection but are not written to the reduced SQLite database.
- Large input databases and all generated outputs are intentionally excluded from GitHub.

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
| `aces_tsam_v4_8760.yaml` | Detailed standalone PyPSA Canada configuration for the Temoa v4 schema. |
| `aces_pyomo_v4_subsystem_extremes_test.yaml` | PyPSA Canada test that selects physically weighted extremes independently for example regions. |
| `aces_pyomo_v4_smoke_test.yaml` | Small Pyomo v4 test using physically weighted system-wide extremes. |
| `aces_pyomo_v4_user_tech_extremes_test.yaml` | Small Pyomo v4 test using physical net load and user-selected existing-generation technologies. |
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

The database must contain the ACES calendar tables and the enabled temporal tables listed under `aces.temporal_tables`. A full-year database may contain 365 or 366 days; the current workflow expects hourly labels `H00` through `H23`.

### Legacy ACES and Temoa v4 schemas

The runner supports both the legacy ACES schema and Temoa database schema v4.
Set `aces.schema` to one of:

- `auto` to detect the schema from database metadata and table signatures.
- `aces_legacy` to require the legacy ACES schema.
- `temoa_v4` to require Temoa v4 and fail if another schema is supplied.

The logical profile names used by clustering settings remain stable across both
schemas (`DemandSpecificDistribution`, `CapacityFactorTech`, and so on). The
v4 adapter maps those names to lowercase v4 tables and columns, writes
representative weights to `time_season.segment_fraction`, and rebuilds
`time_season_sequential` from the chronological cluster assignments.

Please note that this repo is under continoius development and bug fixing

```powershell
python run_aces_tsam.py aces_tsam_v4_8760.yaml --validate-only
python run_aces_tsam.py aces_tsam_v4_8760.yaml
python -m temoa validate temoa_v4_validation.toml
```

The v4 configuration uses `extends: aces_tsam_8760.yaml`, so it inherits the
shared clustering settings while overriding only the database, output, schema,
solver, and dataset-specific extreme-day choices. Configuration inheritance is
recursive, and child mappings override matching parent values.

`temoa_v4_validation.toml` performs Temoa's own build-only validation against
the generated database using `representative_periods` sequencing. This final
check can also reveal unrelated static-data problems already present in the
source database.

### Date formats

`aces.calendar.season_format` defines the canonical format used by `time_season`, the internal hourly index, and the reduced output database. Each temporal table separately declares `source_season_format`, which describes how that source table stores its raw season labels. The runner parses every label with the declared source format and converts it to the canonical calendar before combining demand, capacity factors, costs, or other profiles.

For the supplied database, the correct mapping is:

```yaml
aces:
  calendar:
    season_format: "%m-%d"  # time_season uses month-day

  temporal_tables:
    - table: DemandSpecificDistribution
      source_season_format: "%d-%m"  # DSD uses day-month

    - table: CapacityFactorTech
      source_season_format: "%m-%d"  # CF uses month-day

    - table: CapacityFactorProcess
      source_season_format: "%m-%d"  # CF uses month-day
```


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
  output_database: output/yourdatabase.sqlite
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
| `include_max_daily_ramp_day` | One regional demand-shape ramp day for every configured region. Supports absolute, upward, or downward ramps. |
| `include_max_net_load_ramp_day` | One regional actual-demand minus available-existing-generation ramp day for every configured region. |

The first four rules are native TSAM profile extremes. The two ramp rules are calculated by the ACES runner before clustering. The runner gives each unique calculated date a temporary tiny-weight marker profile, passes those markers through TSAM `max_value`, and removes the markers before writing ACES tables or audit accuracy. If several regions or rules identify the same date, TSAM receives it once and every reason remains in `extreme_day_provenance.csv`.

The ACES runner supports two TSAM extreme methods:

| Method | Result |
|---|---|
| `append` | Adds each unique extreme as a separate representative, normally with weight 1. Final count can exceed `n_representative_days`. |
| `new_cluster` | Adds each unique extreme as a new center and allows similar days to join it. Final count can exceed `n_representative_days`, and the extreme cluster can have weight greater than 1. |

`replace` is not supported. TSAM can create a hybrid representative in which only the targeted profile comes from the extreme day. That replacement is not retained when the clustering is applied to the complete ACES profile set. The runner stops with a validation error if `replace` is selected.

Both custom ramp rules therefore use the selected TSAM `append` or `new_cluster` method. With `append`, each calculated date normally receives weight 1. With `new_cluster`, similar historical days may join it and increase its occurrence weight.

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

## Pyomo Duration-Curve Optimization Engine

Set `clustering.engine: pyomo`. The model divides every selected profile into `n_bins` value thresholds. At each threshold it compares the share of full-year hours above the threshold with the share represented by the weighted selected days. It minimizes the weighted sum of the absolute differences.

Increasing `n_bins` describes the annual duration curves in more detail but increases model size and runtime.

This engine has been exercised with the PyPSA Canada Temoa v4 database. The
small test configurations are:

| Configuration | Purpose |
|---|---|
| `aces_pyomo_v4_smoke_test.yaml` | Fast eight-day test using annual-demand-weighted peak demand and capacity-weighted minimum wind generation. |
| `aces_pyomo_v4_user_tech_extremes_test.yaml` | Fast eight-day test using actual annual demand, installed capacity, and user-selected wind and solar technologies. |

These are smoke tests, not recommended production-quality searches. Increase
the candidate count, random iterations, duration-curve bins, and possibly the
representative-day count before using the result for final modelling.

### Main Pyomo Options

| Option | Meaning and effect |
|---|---|
| `n_representative_days` | Total number of selected days. Pyomo forced extremes are included inside this number rather than added afterward. |
| `solution_method` | `opt` solves the requested selection in one optimization model. `hybrid_random_weighting` tests multiple randomly chosen day sets, optimizes their weights, and retains the best set. |
| `formulation` | `milp` allows the model or hybrid search to select days. `lp_fixed_days` accepts exactly the dates in `fixed_day_ids` and optimizes only their weights. |
| `use_integer_weights` | When `true`, every representative-day weight is a whole number of source days. When `false`, fractional occurrence weights are allowed. |
| `n_bins` | Number of thresholds used to approximate each annual duration curve. More bins improve detail but increase rows, variables, memory use, and runtime. Values around 10 are useful for smoke tests; production runs should compare several larger values. |
| `normalization` | Scales each profile before calculating errors so profiles with large numerical units do not dominate automatically. Supported modes are `minmax`, `zscore`, `robust`, and `none`. Profile weights still apply after normalization. |
| `random_seed` | Makes candidate reduction and hybrid day sampling reproducible. The same data, configuration, dependency versions, and seed should produce the same sampled combinations. |
| `fixed_day_ids` | One-based source-day positions used by `lp_fixed_days`; day 1 is the first season in the input calendar. |
| `forced_day_ids` | Additional one-based source-day positions that must be selected. Prefer named extreme rules when possible because their audit records explain why each day was forced. |
| `enforce_positive_weight` | Prevents a selected day from receiving zero weight. Usually keep this enabled. |
| `minimum_weight` | Minimum occurrence weight for every selected day when positive weights are enforced. A value of `1.0` means at least one source-day equivalent. |

Example lightweight settings:

```yaml
clustering:
  engine: pyomo
  n_representative_days: 8

  pyomo:
    solution_method: hybrid_random_weighting
    formulation: milp
    use_integer_weights: false
    n_bins: 10
    normalization: minmax
    n_random_iterations: 2
    random_seed: 40
    sampled_candidate_pool_size: 30

    candidate_reduction_method: ward_nearest_day
    candidate_feature_mode: chronological_daily_profile
    n_candidate_days: 30

    enforce_positive_weight: true
    minimum_weight: 1.0
```

For `hybrid_random_weighting`, `n_random_iterations` is the number of unique
day combinations to solve. Increasing it searches more possibilities but adds
approximately one weight-optimization solve per iteration.
`sampled_candidate_pool_size` optionally draws a smaller reproducible pool from
the candidate days before combinations are generated. It must contain enough
non-forced days to fill the requested representative-day count.

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

The exact-profile rules accept generated profile names such as
`CapacityFactorTech|AB_Central|E_WIND-ON_EX`:

| Rule | Day that is forced |
|---|---|
| `max_value` | Day containing the profile's highest individual hourly value. |
| `min_value` | Day containing the profile's lowest individual hourly value. |
| `max_period` | Day having the largest sum of its 24 hourly profile values. |
| `min_period` | Day having the smallest sum of its 24 hourly profile values. This is commonly used for a named wind-capacity-factor profile. |

Each listed profile is evaluated independently. Listing wind profiles for 19
regions can therefore force as many as 19 different days. Duplicate dates count
only once. Validation stops if the number of unique forced dates is greater
than `n_representative_days`.

Calculated Pyomo extremes use the explicit `system_wide` and `subsystem`
sections documented below. Both use annual-demand weighting and existing-
capacity weighting; the scope determines whether selected regions are summed
or evaluated independently. Calculated rules placed directly under
`pyomo.extremes` are rejected; this prevents an old configuration from running
with unclear region-aggregation behavior.

### Complete Pyomo Extreme-Option Reference

Pyomo extremes are divided into two deliberately different categories:

1. **Profile-specific extremes** inspect each exact profile independently.
2. **Physically weighted calculated extremes** can be evaluated either for one
   combined system or independently for every selected regional subsystem.

All unique dates selected by these rules count inside
`n_representative_days`. If several profiles, regions, technology groups, or
rules select the same date, the date occupies one place while every reason is
retained in the audit.

#### Profile-specific extremes

```yaml
extremes:
  enabled: true
  selection_mode: force_within_count
  max_value: []
  min_value: []
  max_period: []
  min_period: []
```

| Common field | Allowed value and meaning |
|---|---|
| `enabled` | `true` activates the four exact-profile lists. Use `true` whenever any exact profile is listed. |
| `selection_mode` | Pyomo currently supports only `force_within_count`. Forced dates consume places inside `n_representative_days`. |
| `max_value` | List of exact profiles. Force the day containing each profile's maximum hourly value. |
| `min_value` | List of exact profiles. Force the day containing each profile's minimum hourly value. |
| `max_period` | List of exact profiles. Force the day with each profile's largest 24-hour sum. |
| `min_period` | List of exact profiles. Force the day with each profile's smallest 24-hour sum. |

These four lists never combine or average profiles. Every list entry is a
separate calculation. Profile identifiers are case-sensitive and use the
canonical form printed by the runner. For example:

```yaml
max_value:
  - "DemandSpecificDistribution|AB_Central|2025|D_ELEC"
  - "DemandSpecificDistribution|ON_South|2025|D_ELEC"
min_period:
  - "CapacityFactorTech|AB_Central|E_WIND-ON_EX"
```

The first `max_value` entry selects the maximum-hour day for the AB_Central
profile; the second independently selects the maximum-hour day for ON_South.
They are not summed into a Canada-wide profile. Similarly, `min_period` above
uses only the named AB_Central wind profile. If the entries select different
dates, both dates are forced.

#### Calculated extremes: system-wide versus subsystem

The same six calculated rules are available under two explicit scopes:

| Scope | Meaning of `regions: all` | Number of results per enabled rule |
|---|---|---|
| `system_wide` | Sum all regions first, then search the combined physical profile. | Normally one date. |
| `subsystem` | Search each region independently; regions are never added together. | Up to one date per region, or per region and minimum-generation group. |

A region list limits the calculation without changing its scope. For example,
`system_wide.regions: [AB_Central, AB_South]` searches the sum of those two
regions, while `subsystem.regions: [AB_Central, AB_South]` searches each of the
two regions separately.

All calculated demand quantities are physical and annual-demand weighted:

```text
hourly demand (GW)
  = annual demand (GWh) × hourly DSD / (hourly segment fraction × calendar hours)
```

All calculated generation quantities are capacity weighted:

```text
available generation (GW) = existing capacity (GW) × hourly capacity factor
```

Existing-capacity vintages for a region and technology are summed. A missing or
zero existing capacity contributes zero and does not require a capacity-factor
profile. The calculations represent available generation, not optimized model
dispatch.

The calculated rules are:

| Rule | Selected event | Units | Technology handling |
|---|---|---|---|
| `include_peak_demand_day` | Maximum hourly actual demand. | GW | None. |
| `include_min_wind_day` | Minimum 24-hour available wind generation. | GWh | Listed wind technologies are summed. |
| `include_peak_net_load_day` | Maximum hourly demand minus available generation. | GW | Listed technologies are summed. |
| `include_min_generation_period` | Minimum 24-hour available generation. | GWh | Each named technology group is evaluated separately. |
| `include_max_daily_ramp_day` | Largest within-day hourly change in actual demand. | GW | None. |
| `include_max_net_load_ramp_day` | Largest within-day hourly change in net load. | GW | Listed technologies are summed. |

Ramp `direction` can be `absolute`, `upward`, or `downward`. Ramps compare
H00→H01 through H22→H23; they do not compare H23 with the next day's H00.

#### Complete system-wide example

```yaml
system_wide:
  enabled: true
  reference_period: first_model_period
  regions: all

  demand:
    annual_table: demand
    region_column: region
    period_column: period
    commodity_column: commodity
    value_column: demand
    unit_column: units
    commodity: D_ELEC
    profile_table: DemandSpecificDistribution
    segfrac:
      mode: calendar

  existing_generation:
    capacity_table: existing_capacity
    region_column: region
    tech_column: tech
    value_column: capacity
    unit_column: units
    capacity_factor_table: CapacityFactorTech

  include_peak_demand_day:
    enabled: true

  include_min_wind_day:
    enabled: true
    technologies:
      - tech: E_WIND-ON_EX
        capacity_factor_table: CapacityFactorTech

  include_peak_net_load_day:
    enabled: true
    technologies:
      - tech: E_WIND-ON_EX
      - tech: E_SOLPV-CEN_EX

  # Each group is a separate extreme. Keep only wind_plus_solar if one combined
  # minimum-generation day is desired.
  include_min_generation_period:
    enabled: true
    groups:
      - name: wind
        technologies: [E_WIND-ON_EX]
      - name: solar
        technologies: [E_SOLPV-CEN_EX]
      - name: wind_plus_solar
        technologies: [E_WIND-ON_EX, E_SOLPV-CEN_EX]

  include_max_daily_ramp_day:
    enabled: true
    direction: absolute

  include_max_net_load_ramp_day:
    enabled: true
    direction: absolute
    technologies: [E_WIND-ON_EX, E_SOLPV-CEN_EX]
```

This example can select one date for peak system demand, one for minimum system
wind, one for peak system net load, three separate minimum-generation dates,
one system demand-ramp date, and one system net-load-ramp date. Coincident dates
are deduplicated.

#### Complete subsystem example

The subsystem block uses the same database mappings and rule names. Its only
semantic difference is that every selected region is evaluated independently:

```yaml
subsystem:
  enabled: true
  reference_period: first_model_period
  regions: all  # Or [AB_Central, ON_South, QC_East].
  demand: ...
  existing_generation: ...

  include_peak_demand_day:
    enabled: true

  include_min_wind_day:
    enabled: true
    technologies: [E_WIND-ON_EX]

  include_peak_net_load_day:
    enabled: true
    technologies: [E_WIND-ON_EX, E_SOLPV-CEN_EX]

  include_min_generation_period:
    enabled: true
    groups:
      - name: wind
        technologies: [E_WIND-ON_EX]
      - name: solar
        technologies: [E_SOLPV-CEN_EX]

  include_max_daily_ramp_day:
    enabled: true
    direction: absolute

  include_max_net_load_ramp_day:
    enabled: true
    direction: absolute
    technologies: [E_WIND-ON_EX, E_SOLPV-CEN_EX]
```

With `regions: all`, peak demand can select one day for every region, peak net
load can select one day for every region, and each minimum-generation group can
select one day for every region that has positive capacity in that group.
Regions with no positive capacity for a minimum-generation group are skipped
because an all-zero series has no unique minimum-generation event. A region
with zero capacity for one listed net-load technology still works: that
technology contributes zero to the region's net load.

Supported annual-demand units are PJ, TWh, GWh, and MWh. Supported existing-
capacity units are GW, MW, and kW. Use `segfrac.mode: calendar` for Temoa v4,
`database` for a valid legacy hourly SegFrac table, or `uniform_full_year` only
when every source hour is equally weighted.

#### Sparse demand-profile safeguards

Some trade-demand DSD profiles are zero on most days. If none of their nonzero
days is selected, the reduced profile cannot be normalized to one. Force one
nonzero day for each such profile with `max_value`, for example:

```yaml
max_value:
  - "DemandSpecificDistribution|AB_South|2025|D_ELEC_EXPORT_AB_SOUTH_US_TRADE"
  - "DemandSpecificDistribution|QC_East|2025|D_ELEC_EXPORT_QC_EAST_US_TRADE"
  - "DemandSpecificDistribution|SK|2025|D_ELEC_EXPORT_SK_US_TRADE"
```

These are data-coverage safeguards. They are additional forced reasons and are
not part of the physical peak-demand, wind, or net-load equations.

### Solver Options

| Option | Meaning |
|---|---|
| `solver.name` | Pyomo solver name. `highs` is the recommended portable default. `cplex` and `gurobi` require separate installations and licences. |
| `solver.time_limit_seconds` | Optional solver time limit. Set `null` for no configured limit. |
| `solver.mip_gap` | Relative optimality-gap target for mixed-integer runs. It has no practical effect on a continuous LP that already solves to optimality. |
| `solver.threads` | Optional solver-thread limit. Set `null` to use the solver default. |
| `solver.tee` | When `true`, print the solver log. Use `false` for quieter normal runs. |

CPLEX 20.1's command-line interface can solve the model under modern Python,
but some Windows/Pyomo combinations fail afterward while reopening a quoted
temporary log path. The bundled smoke tests use HiGHS to avoid this interface
problem. This does not prevent using a newer compatible CPLEX installation.

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
| `pyomo_iteration_history.csv` | For hybrid random weighting, every unique tested day combination, solver outcome, objective value and per-profile contribution, optimized weights, and reconstruction errors. |

For Pyomo runs, `weight` is optimized by the duration-curve model. `nearest_assignment_count` is a diagnostic count obtained by assigning each original day to its closest selected day. The ACES database uses `weight`, not `nearest_assignment_count`.

For `hybrid_random_weighting`, complete selected-day combinations are sampled without replacement: an individual day may appear in several iterations, but the same combination is never solved twice. The Excel audit includes the same history in the `Pyomo_Iterations` sheet. It records chronological RMSE and MAE, duration-curve RMSE, MAE, and normalized RMSE, peak error, annual-energy error, optimized weights, per-profile objective contributions, and the total Pyomo objective for every successful iteration. The winning iteration is explicitly marked; the total Pyomo objective remains the metric used to choose it.

## Known Limitations

- TSAM output must use `medoid` while the database uses source-day labels.
- TSAM `replace` extremes are not supported; use `append` or `new_cluster`.
- TSAM periods and output segments must remain 24 hours for the current ACES schema.
- Extra attributes affect selection but are not written to the reduced SQLite database.
- Large input databases and all generated outputs are intentionally excluded from GitHub.

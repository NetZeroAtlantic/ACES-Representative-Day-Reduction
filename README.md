# ACES 8760 Representative-Day Clustering

This tool reads the native ACES SQLite schema, reduces 8,760/8,784 hourly values into weighted representative days with TSAM or Pyomo, and writes a reduced ACES database. Representative seasons use selected historical dates, such as `01-17`, rather than generated names such as `R01`.

## Files

- `aces_tsam_8760.yaml`: the only configuration file.
- `run_aces_tsam.py`: ACES representative-day runner for TSAM and Pyomo.
- `extra_attributes/`: optional timestamped CSV or Excel inputs.
- `data/`: place the original 8,760/8,784-hour ACES SQLite database here.
- `output/`: reduced SQLite database and audit CSV files.

## Test

Install dependencies in the active environment:

```powershell
python -m pip install -r requirements.txt
```

Validate the database and YAML without clustering:

```powershell
python run_aces_tsam.py aces_tsam_8760.yaml --validate-only
```

Run the complete clustering and database conversion:

```powershell
python run_aces_tsam.py aces_tsam_8760.yaml
```

The configured output is `output/Offshore_wind_NS-NB_repdays_actual_dates.sqlite`. `overwrite_output: true` makes the command repeatable and never modifies the input database.

## Configure

Edit `aces_tsam_8760.yaml`:

- `project.input_database`: input ACES SQLite path.
- `clustering.n_representative_days`: normal cluster count.
- `clustering.engine`: `tsam` or `pyomo`.
- `clustering.tsam.cluster.method`: `hierarchical`, `kmeans`, `kmedoids`, `kmaxoids`, `averaging`, or `contiguous`.
- `clustering.tsam.cluster.representation.type`: output profile representation.
- `clustering.attributes`: included ACES tables and profile weights.
- `clustering.tsam.extremes`: peak-load, low-wind, and other forced extreme days.
- `clustering.pyomo.solution_method`: `opt` or `hybrid_random_weighting`.
- `clustering.pyomo.extremes`: exact profile-based extreme days forced inside the requested Pyomo count.

With `extremes.method: new_cluster`, forced extremes are added to the requested normal count. For example, 12 normal clusters can produce 14 final representative days. Use `append` or `new_cluster` in the current ACES runner. Standard TSAM `replace` is not currently preserved when the clustering is transferred to the complete database profile matrix.

For Pyomo, forced extremes always count inside `n_representative_days`. With 15 requested days and five unique forced extremes, Pyomo keeps those five and chooses ten more. `opt` chooses days and weights together. `hybrid_random_weighting` repeatedly samples fixed day sets, optimizes only their occurrence weights, and keeps the lowest duration-curve error.

## Extra Attributes

Put a `.csv` or `.xlsx` file in `extra_attributes/`. It must contain one `timestamp` column and 8,784 hourly rows:

```csv
timestamp,temperature,humidity
2024-01-01 00:00:00,-4.2,0.81
2024-01-01 01:00:00,-4.5,0.82
```

Then list the feature names under:

```yaml
clustering:
  attributes:
    include:
      extra_attributes: [temperature, humidity]
```

Extra attributes influence day selection but are not written into ACES tables.

## Outputs

- Reduced ACES SQLite database using each selected historical date as its season label.
- `RepresentativeDayWeight` table whose weights sum to the original 366 days.
- `SegFrac` values whose total is 1.
- DSD profiles normalized to 1 after applying representative-day weights.
- `cluster_assignments.csv`, `representative_day_weights.csv`, and engine-specific accuracy CSV.
- Pyomo runs also create `pyomo_forced_days.csv` and `pyomo_summary.csv`.
- `representative_day_audit.xlsx` for both engines, with representative weights and assignment counts, forced-extreme provenance, ACES region/period/demand/technology details, profile weights, accuracy, and the complete run configuration.

## Future: TSAM Hybrid Replace

This feature is intentionally deferred. Standard TSAM `replace` can create a hybrid representative in which a configured extreme profile comes from its extreme historical day while other profiles remain from the normal cluster medoid.

Example:

```text
Representative cluster center: 01-28
NL demand profile:             01-24 peak-demand day
Wind and solar profiles:       01-28 normal medoid day
```

The initial `tsam.aggregate()` result contains this hybrid representation. The current ACES runner subsequently calls `result.clustering.apply(profiles)` to apply the same clustering to every ACES temporal profile. TSAM cannot reproduce the column-specific `replace` injection through `apply()`, so the transferred result returns to the normal medoid values.

The future implementation should introduce an explicit configuration option rather than silently changing standard behaviour:

```yaml
clustering:
  tsam:
    extremes:
      method: replace
      preserve_hybrid_result: true
```

Required implementation work:

1. Retain the initial TSAM hybrid representatives instead of discarding them.
2. Combine hybrid values with temporal profiles that were not used during clustering.
3. Track provenance separately for every representative/profile combination.
4. Avoid claiming that a hybrid representative is one complete historical date.
5. Define a stable hybrid label, such as `HYB01`, or store both a cluster label and per-profile source dates.
6. Extend the Excel audit with normal-medoid date, extreme-source date, affected profile, rule, region, period, technology or demand, and replacement value.
7. Validate that weights sum to the original year, SegFrac sums to one, DSD remains normalized, and every temporal ACES table has consistent representative labels.

Acceptance criteria:

- Configured extreme profiles retain their complete extreme-day H00–H23 values.
- Non-targeted profiles retain the intended normal cluster representation.
- Multiple extreme rules affecting one cluster are recorded without losing provenance.
- The reduced SQLite database never labels a hybrid profile as a single real historical day.
- `append` and `new_cluster` behaviour remains unchanged.
- Without `preserve_hybrid_result: true`, the runner rejects TSAM `replace` with a clear compatibility error.

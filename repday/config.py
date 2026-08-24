from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class AttributeConfig:
    name: str
    column: str
    weight: float = 1.0
    normalize: bool = True
    normalization: str = "minmax"  # minmax, zscore, robust, none
    active: bool = True


@dataclass
class ClusteringConfig:
    n_representative_days: int = 12
    hours_per_day: int = 24
    timestamp_column: Optional[str] = "timestamp"
    day_id_column: Optional[str] = None
    require_full_days: bool = True
    use_integer_weights: bool = True
    formulation: str = "milp"  # "milp" or "lp_fixed_days"
    fixed_day_ids: list[int] | None = None
    n_bins: int = 5
    forced_day_ids: list[int] | None = None
    # -------------------------
    # Candidate reduction
    # -------------------------
    candidate_reduction_method: str = "none"
    candidate_feature_mode: str = "chronological_daily_profile"
    n_candidate_days: int | None = None
    include_extreme_days: bool = False
    include_peak_demand_day: bool = True
    include_min_wind_day: bool = True
    include_max_daily_ramp_day: bool = False
    random_seed: int = 42
    force_extreme_days_in_final_selection: bool = True
    enforce_positive_weight_for_selected_days: bool = True
    min_weight_if_selected: float = 1.0
    # -------------------------
    # Solution method
    # -------------------------
    solution_method: str = "opt"   # "opt" or "hybrid_random_weighting"
    n_random_iterations: int = 50
    sampled_candidate_pool_size: Optional[int] = None


@dataclass
class SolverConfig:
    solver_name: str = "cplex"
    tee: bool = False
    timelimit_seconds: Optional[int] = 300
    mipgap: Optional[float] = 0.001
    threads: Optional[int] = None


@dataclass
class OutputConfig:
    output_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    save_csvs: bool = True


@dataclass
class RunConfig:
    excel_path: str
    attributes: List[AttributeConfig]
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return {
            "excel_path": self.excel_path,
            "attributes": [asdict(a) for a in self.attributes],
            "clustering": asdict(self.clustering),
            "solver": asdict(self.solver),
            "output": asdict(self.output),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

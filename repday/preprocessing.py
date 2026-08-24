from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List
import numpy as np
import pandas as pd
from .config import AttributeConfig


@dataclass
class PreparedData:
    hourly: pd.DataFrame
    day_labels: list[int]
    daily_profiles: Dict[str, np.ndarray]
    daily_duration_curves: Dict[str, np.ndarray]
    original_duration_curves: Dict[str, np.ndarray]
    normalization_params: Dict[str, Dict[str, float]]


class Normalizer:
    def __init__(self) -> None:
        self.params: Dict[str, Dict[str, float]] = {}

    def fit_transform(self, s: pd.Series, method: str, key: str) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        x = x.interpolate(limit_direction="both").ffill().bfill()
        method = method.lower()

        if method == "none":
            self.params[key] = {"method": "none"}
            return x

        if method == "minmax":
            x_min = float(x.min())
            x_max = float(x.max())
            denom = x_max - x_min
            if math.isclose(denom, 0.0):
                denom = 1.0
            self.params[key] = {"method": "minmax", "min": x_min, "max": x_max}
            return (x - x_min) / denom

        if method == "zscore":
            mean = float(x.mean())
            std = float(x.std(ddof=0))
            if math.isclose(std, 0.0):
                std = 1.0
            self.params[key] = {"method": "zscore", "mean": mean, "std": std}
            return (x - mean) / std

        if method == "robust":
            median = float(x.median())
            q1 = float(x.quantile(0.25))
            q3 = float(x.quantile(0.75))
            iqr = q3 - q1
            if math.isclose(iqr, 0.0):
                iqr = 1.0
            self.params[key] = {"method": "robust", "median": median, "iqr": iqr}
            return (x - median) / iqr

        raise ValueError(f"Unsupported normalization method: {method}")

    def inverse_transform(self, s: pd.Series, key: str) -> pd.Series:
        params = self.params[key]
        method = params["method"]
        if method == "none":
            return s
        if method == "minmax":
            return s * (params["max"] - params["min"]) + params["min"]
        if method == "zscore":
            return s * params["std"] + params["mean"]
        if method == "robust":
            return s * params["iqr"] + params["median"]
        raise ValueError(f"Unsupported normalization method: {method}")


def prepare_daily_structures(
    hourly: pd.DataFrame,
    attributes: List[AttributeConfig],
    hours_per_day: int = 24,
    require_full_days: bool = True,
) -> PreparedData:
    df = hourly.copy()

    if require_full_days:
        counts = df.groupby("day_id").size()
        full_days = counts[counts == hours_per_day].index.tolist()
        df = df[df["day_id"].isin(full_days)].copy()

    normalizer = Normalizer()
    active = [a for a in attributes if a.active]

    for attr in active:
        if attr.column not in df.columns:
            raise KeyError(f"Missing input column: {attr.column}")
        if attr.normalize:
            df[f"{attr.name}__norm"] = normalizer.fit_transform(df[attr.column], attr.normalization, attr.name)
        else:
            df[f"{attr.name}__norm"] = pd.to_numeric(df[attr.column], errors="coerce").astype(float)
            df[f"{attr.name}__norm"] = df[f"{attr.name}__norm"].interpolate(limit_direction="both").ffill().bfill()
            normalizer.params[attr.name] = {"method": "none"}

    day_labels = sorted(df["day_id"].unique().tolist())
    daily_profiles: Dict[str, np.ndarray] = {}
    daily_duration_curves: Dict[str, np.ndarray] = {}
    original_duration_curves: Dict[str, np.ndarray] = {}

    for attr in active:
        col = f"{attr.name}__norm"
        pivot = df.pivot(index="day_id", columns="hour_in_day", values=col).sort_index()
        if pivot.shape[1] != hours_per_day:
            raise ValueError(f"Attribute '{attr.name}' does not have {hours_per_day} hours for every day.")
        arr = pivot.to_numpy(dtype=float)
        daily_profiles[attr.name] = arr
        daily_duration_curves[attr.name] = np.sort(arr, axis=1)[:, ::-1]
        original_duration_curves[attr.name] = np.sort(df[col].to_numpy(dtype=float))[::-1]

    return PreparedData(
        hourly=df,
        day_labels=day_labels,
        daily_profiles=daily_profiles,
        daily_duration_curves=daily_duration_curves,
        original_duration_curves=original_duration_curves,
        normalization_params=normalizer.params,
    )

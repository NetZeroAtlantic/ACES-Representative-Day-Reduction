from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def duration_curve_metrics(original: np.ndarray, approx: np.ndarray) -> Dict[str, float]:
    if len(original) != len(approx):
        raise ValueError("Original and approximated arrays must have the same length.")
    mae = float(np.mean(np.abs(original - approx)))
    rmse = float(np.sqrt(np.mean((original - approx) ** 2)))
    denom = float(np.max(original) - np.min(original))
    nrmse = rmse / denom if denom != 0 else 0.0
    peak_error = float(abs(np.max(original) - np.max(approx)))
    annual_energy_error = float(abs(np.sum(original) - np.sum(approx)))
    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "peak_error": peak_error,
        "annual_energy_error": annual_energy_error,
    }


def build_metrics_table(original_curves: Dict[str, np.ndarray], approx_curves: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, original in original_curves.items():
        stats = duration_curve_metrics(original, approx_curves[name])
        rows.append({"attribute": name, **stats})
    return pd.DataFrame(rows)

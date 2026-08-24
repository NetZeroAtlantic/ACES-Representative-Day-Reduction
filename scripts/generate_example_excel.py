from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def generate_example_year(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01 00:00:00", periods=8760, freq="H")
    hour = ts.hour.to_numpy()
    dayofyear = ts.dayofyear.to_numpy()
    dow = ts.dayofweek.to_numpy()

    # Wind: seasonal + synoptic variability + noise, clipped to [0,1]
    seasonal_wind = 0.45 + 0.18 * np.cos(2 * np.pi * (dayofyear - 20) / 365.0)
    intraday_wind = 0.06 * np.sin(2 * np.pi * (hour + 3) / 24.0)
    synoptic = 0.12 * np.sin(2 * np.pi * np.arange(len(ts)) / (24.0 * 5.5))
    wind = seasonal_wind + intraday_wind + synoptic + rng.normal(0, 0.08, len(ts))
    wind = np.clip(wind, 0.0, 1.0)

    # Demand: winter higher, morning/evening peaks, lower on weekends
    winter = 700 + 150 * np.cos(2 * np.pi * (dayofyear - 15) / 365.0)
    morning_peak = 90 * np.exp(-((hour - 8) / 2.8) ** 2)
    evening_peak = 150 * np.exp(-((hour - 19) / 3.5) ** 2)
    base_load = 460 + 30 * np.sin(2 * np.pi * (hour - 6) / 24.0)
    weekend_adj = np.where(dow >= 5, -80, 0)
    demand = winter + base_load + morning_peak + evening_peak + weekend_adj + rng.normal(0, 22, len(ts))
    demand = np.clip(demand, 350, None)

    return pd.DataFrame({
        "timestamp": ts,
        "wind": np.round(wind, 4),
        "demand": np.round(demand, 2),
    })


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(parents=True, exist_ok=True)

    df = generate_example_year()
    df.to_excel(out / "example_wind_demand_8760.xlsx", index=False)
    df[["timestamp", "wind"]].to_excel(out / "example_wind_8760.xlsx", index=False)
    df[["timestamp", "demand"]].to_excel(out / "example_demand_8760.xlsx", index=False)

    print(f"Created example workbooks in {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import pandas as pd


class DataError(Exception):
    pass


def load_hourly_profiles(
    excel_path: str,
    timestamp_column: str | None = "timestamp",
    day_id_column: str | None = None,
    hours_per_day: int = 24,
    sheet_name=0,
) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    if df.empty:
        raise DataError("Input Excel file is empty.")

    if timestamp_column and timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        df["day_id"] = ((df[timestamp_column] - df[timestamp_column].iloc[0]).dt.total_seconds() // 86400).astype(int) + 1
        df["hour_in_day"] = df.groupby("day_id").cumcount()
        return df

    if day_id_column and day_id_column in df.columns:
        df = df.copy()
        df["day_id"] = df[day_id_column].astype(int)
        df["hour_in_day"] = df.groupby("day_id").cumcount()
        return df

    if len(df) % hours_per_day != 0:
        raise DataError(
            "Could not infer day structure. Provide either timestamp_column or day_id_column."
        )

    df = df.copy().reset_index(drop=True)
    df["day_id"] = (df.index // hours_per_day) + 1
    df["hour_in_day"] = df.index % hours_per_day
    return df

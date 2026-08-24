from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering

from .config import AttributeConfig
from .preprocessing import PreparedData


@dataclass
class CandidateReductionResult:
    candidate_day_ids: List[int]
    feature_matrix: pd.DataFrame
    extreme_day_ids: List[int]
    cluster_representative_day_ids: List[int]


def _get_active_attributes(attributes: List[AttributeConfig]) -> List[AttributeConfig]:
    return [a for a in attributes if a.active]


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def build_daily_feature_matrix(
    prepared: PreparedData,
    attributes: List[AttributeConfig],
    feature_mode: str = "chronological_daily_profile",
) -> pd.DataFrame:
    """
    Build one feature vector per day.

    feature_mode options:
    - chronological_daily_profile
    - daily_duration_curve
    - hybrid_profile_and_duration
    - summary_statistics
    """
    active_attributes = _get_active_attributes(attributes)
    day_ids = list(prepared.day_labels)

    rows = []
    for i, day_id in enumerate(day_ids):
        row = {"day_id": day_id}

        for attr in active_attributes:
            daily = prepared.daily_profiles[attr.name][i, :].astype(float)

            if feature_mode == "chronological_daily_profile":
                vec = daily

            elif feature_mode == "daily_duration_curve":
                vec = np.sort(daily)[::-1]

            elif feature_mode == "hybrid_profile_and_duration":
                vec = np.concatenate([daily, np.sort(daily)[::-1]])

            elif feature_mode == "summary_statistics":
                ramps = np.diff(daily)
                vec = np.array(
                    [
                        np.mean(daily),
                        np.min(daily),
                        np.max(daily),
                        np.std(daily),
                        np.max(daily) - np.min(daily),
                        np.mean(np.abs(ramps)) if len(ramps) > 0 else 0.0,
                        np.max(ramps) if len(ramps) > 0 else 0.0,
                        np.min(ramps) if len(ramps) > 0 else 0.0,
                    ],
                    dtype=float,
                )
            else:
                raise ValueError(f"Unsupported feature_mode: {feature_mode}")

            for j, value in enumerate(vec):
                row[f"{attr.name}_f{j+1}"] = float(value)

        rows.append(row)

    return pd.DataFrame(rows)


def select_extreme_days(
    prepared: PreparedData,
    attributes: List[AttributeConfig],
    include_peak_demand_day: bool = True,
    include_min_wind_day: bool = True,
    include_max_daily_ramp_day: bool = False,
) -> List[int]:
    """
    Select a few extreme days as guaranteed candidates.
    Returns unique actual day_ids.
    """
    active_attributes = _get_active_attributes(attributes)
    day_ids = list(prepared.day_labels)
    selected: List[int] = []

    attr_names = {a.name for a in active_attributes}

    if include_peak_demand_day and "demand" in attr_names:
        demand_daily_max = prepared.daily_profiles["demand"].max(axis=1)
        idx = int(np.argmax(demand_daily_max))
        selected.append(day_ids[idx])

    if include_min_wind_day and "wind" in attr_names:
        wind_daily_mean = prepared.daily_profiles["wind"].mean(axis=1)
        idx = int(np.argmin(wind_daily_mean))
        selected.append(day_ids[idx])

    if include_max_daily_ramp_day and "demand" in attr_names:
        ramps = np.abs(np.diff(prepared.daily_profiles["demand"], axis=1))
        daily_max_ramp = ramps.max(axis=1)
        idx = int(np.argmax(daily_max_ramp))
        selected.append(day_ids[idx])

    return sorted(set(selected))


def _nearest_day_to_center(feature_df: pd.DataFrame, center: np.ndarray) -> int:
    X = feature_df.drop(columns=["day_id"]).to_numpy(dtype=float)
    distances = np.sum((X - center.reshape(1, -1)) ** 2, axis=1)
    idx = int(np.argmin(distances))
    return int(feature_df.iloc[idx]["day_id"])


def _nearest_day_in_cluster(
    feature_df: pd.DataFrame,
    cluster_indices: np.ndarray,
) -> int:
    sub = feature_df.iloc[cluster_indices].copy()
    X = sub.drop(columns=["day_id"]).to_numpy(dtype=float)
    center = X.mean(axis=0)
    distances = np.sum((X - center.reshape(1, -1)) ** 2, axis=1)
    idx_local = int(np.argmin(distances))
    return int(sub.iloc[idx_local]["day_id"])


def select_candidate_days_kmeans(
    feature_df: pd.DataFrame,
    n_candidate_days: int,
    random_seed: int = 42,
) -> List[int]:
    """
    Use k-means, then pick the actual historical day nearest each centroid.
    """
    if n_candidate_days >= len(feature_df):
        return sorted(feature_df["day_id"].astype(int).tolist())

    X = feature_df.drop(columns=["day_id"]).to_numpy(dtype=float)

    model = KMeans(
        n_clusters=n_candidate_days,
        random_state=random_seed,
        n_init=20,
    )
    model.fit(X)

    reps = []
    available_day_ids = feature_df["day_id"].astype(int).to_numpy()
    for center in model.cluster_centers_:
        distances = np.sum((X - center.reshape(1, -1)) ** 2, axis=1)
        for position in np.argsort(distances):
            day_id = int(available_day_ids[position])
            if day_id not in reps:
                reps.append(day_id)
                break

    return sorted(set(reps))


def select_candidate_days_ward(
    feature_df: pd.DataFrame,
    n_candidate_days: int,
) -> List[int]:
    """
    Use Ward hierarchical clustering, then pick one actual day nearest the cluster mean.
    """
    if n_candidate_days >= len(feature_df):
        return sorted(feature_df["day_id"].astype(int).tolist())

    X = feature_df.drop(columns=["day_id"]).to_numpy(dtype=float)
    model = AgglomerativeClustering(
        n_clusters=n_candidate_days,
        linkage="ward",
    )
    labels = model.fit_predict(X)

    reps = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_indices = np.where(labels == cluster_id)[0]
        reps.append(_nearest_day_in_cluster(feature_df, cluster_indices))

    return sorted(set(reps))


def reduce_candidate_days(
    prepared: PreparedData,
    attributes: List[AttributeConfig],
    method: str = "none",
    feature_mode: str = "chronological_daily_profile",
    n_candidate_days: Optional[int] = None,
    include_extreme_days: bool = False,
    include_peak_demand_day: bool = True,
    include_min_wind_day: bool = True,
    include_max_daily_ramp_day: bool = False,
    random_seed: int = 42,
) -> CandidateReductionResult:
    """
    Main entry point.

    method options:
    - none
    - kmeans_nearest_day
    - ward_nearest_day
    - extreme_plus_kmeans
    - extreme_plus_ward
    """
    all_day_ids = sorted(int(d) for d in prepared.day_labels)
    feature_df = build_daily_feature_matrix(
        prepared=prepared,
        attributes=attributes,
        feature_mode=feature_mode,
    )

    extreme_day_ids: List[int] = []
    cluster_reps: List[int] = []

    if method == "none" or n_candidate_days is None or n_candidate_days >= len(all_day_ids):
        return CandidateReductionResult(
            candidate_day_ids=all_day_ids,
            feature_matrix=feature_df,
            extreme_day_ids=[],
            cluster_representative_day_ids=[],
        )

    if include_extreme_days or method in {"extreme_plus_kmeans", "extreme_plus_ward"}:
        extreme_day_ids = select_extreme_days(
            prepared=prepared,
            attributes=attributes,
            include_peak_demand_day=include_peak_demand_day,
            include_min_wind_day=include_min_wind_day,
            include_max_daily_ramp_day=include_max_daily_ramp_day,
        )

    remaining_slots = max(0, n_candidate_days - len(extreme_day_ids))
    feature_df_remaining = feature_df[~feature_df["day_id"].isin(extreme_day_ids)].copy()

    if remaining_slots == 0:
        candidate_ids = sorted(set(extreme_day_ids))
        return CandidateReductionResult(
            candidate_day_ids=candidate_ids,
            feature_matrix=feature_df,
            extreme_day_ids=sorted(extreme_day_ids),
            cluster_representative_day_ids=[],
        )

    if method in {"kmeans_nearest_day", "extreme_plus_kmeans"}:
        cluster_reps = select_candidate_days_kmeans(
            feature_df=feature_df_remaining,
            n_candidate_days=remaining_slots,
            random_seed=random_seed,
        )

    elif method in {"ward_nearest_day", "extreme_plus_ward"}:
        cluster_reps = select_candidate_days_ward(
            feature_df=feature_df_remaining,
            n_candidate_days=remaining_slots,
        )

    else:
        raise ValueError(f"Unsupported candidate reduction method: {method}")

    candidate_ids = sorted(set(extreme_day_ids + cluster_reps))

    return CandidateReductionResult(
        candidate_day_ids=candidate_ids,
        feature_matrix=feature_df,
        extreme_day_ids=sorted(extreme_day_ids),
        cluster_representative_day_ids=sorted(cluster_reps),
    )

"""Feature engineering for the RF intervention model.

FEATURE_NAMES is the frozen, ordered feature list. Everything downstream
(training, prediction, intervention counterfactuals) must produce columns in
exactly this order -- a reordering bug here would silently corrupt
predictions, so FEATURE_NAMES is saved alongside the trained model
(see app.ml.train_rf) rather than re-derived at load time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.grid.join import _equirect_project

BUCKET_NAMES: list[str] = ["road", "sidewalk", "built", "vegetation", "sky", "other"]

FEATURE_NAMES: list[str] = [
    # base surface-composition fractions
    "road",
    "sidewalk",
    "built",
    "vegetation",
    "sky",
    "other",
    # derived
    "impervious_fraction",
    "green_view_index",
    "sky_view_factor_proxy",
    "vegetation_to_impervious_ratio",
    "person_count",
    "vehicle_count",
    # context (hour as sin/cos, NOT a raw integer -- 23:00 and 00:00 must be close)
    "hour_sin",
    "hour_cos",
    "solar_wm2",
    "wind_ms",
    "relative_humidity",
    # spatial
    "distance_to_aoi_centre_m",
]

_RATIO_EPSILON = 1e-3  # avoids divide-by-zero when impervious_fraction ~ 0


def _aoi_center(bbox: str) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


def build_features(df: pd.DataFrame, bbox: str) -> pd.DataFrame:
    """Compute the frozen feature matrix from a training-set-shaped dataframe.

    Required input columns: road, sidewalk, built, vegetation, sky, other,
    impervious_fraction, green_view_index, sky_view_factor_proxy,
    person_count, bicycle_count, car_count, motorcycle_count, bus_count,
    truck_count, hour (0-23), solar_wm2, wind_ms, relative_humidity, lat, lon.

    Works on any row count, including a single row (used by
    app.ml.intervention for baseline-vs-counterfactual prediction).
    """
    out = pd.DataFrame(index=df.index)

    for name in BUCKET_NAMES:
        out[name] = df[name].astype(float)

    out["impervious_fraction"] = df["impervious_fraction"].astype(float)
    out["green_view_index"] = df["green_view_index"].astype(float)
    out["sky_view_factor_proxy"] = df["sky_view_factor_proxy"].astype(float)
    out["vegetation_to_impervious_ratio"] = df["vegetation"].astype(float) / (
        df["impervious_fraction"].astype(float) + _RATIO_EPSILON
    )

    out["person_count"] = df["person_count"].astype(float)
    out["vehicle_count"] = (
        df["bicycle_count"].astype(float)
        + df["car_count"].astype(float)
        + df["motorcycle_count"].astype(float)
        + df["bus_count"].astype(float)
        + df["truck_count"].astype(float)
    )

    hour_frac = df["hour"].astype(float)
    angle = 2 * np.pi * hour_frac / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["solar_wm2"] = df["solar_wm2"].astype(float)
    out["wind_ms"] = df["wind_ms"].astype(float)
    out["relative_humidity"] = df["relative_humidity"].astype(float)

    center_lat, center_lon = _aoi_center(bbox)
    x_m, y_m = _equirect_project(df["lat"].to_numpy(), df["lon"].to_numpy(), center_lat, center_lon)
    out["distance_to_aoi_centre_m"] = np.hypot(x_m, y_m)

    return out[FEATURE_NAMES]

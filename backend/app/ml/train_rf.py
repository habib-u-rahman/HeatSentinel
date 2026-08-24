"""Trains the RF intervention model, reporting an honest spatial-block CV score
alongside a (leakier) random-split CV score so the gap between them is
visible rather than hidden.

CRITICAL: neighbouring sample points are spatially autocorrelated (nearby
streets tend to share surface composition and near-identical temperatures),
so a random K-fold split leaks -- a held-out point's near-duplicate neighbour
is often still in the training fold, producing a fake-high R2. Spatial block
CV (holding out whole AOI blocks) is the honest score.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from app.ml.features import FEATURE_NAMES, build_features

logger = logging.getLogger(__name__)

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
N_BLOCK_ROWS = 4  # a 4x4 grid of spatial blocks over the AOI
N_BLOCK_COLS = 4
SPATIAL_CV_SPLITS = 4  # the 16 blocks are grouped into this many held-out folds
RANDOM_CV_SPLITS = 5
PERMUTATION_REPEATS = 20


def _assign_blocks(lats: np.ndarray, lons: np.ndarray, bbox: str) -> np.ndarray:
    """Assign each row to one of N_BLOCK_ROWS x N_BLOCK_COLS spatial blocks over bbox."""
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    lon_span = max(max_lon - min_lon, 1e-12)
    lat_span = max(max_lat - min_lat, 1e-12)

    col = np.clip(((lons - min_lon) / lon_span * N_BLOCK_COLS).astype(int), 0, N_BLOCK_COLS - 1)
    row = np.clip(((lats - min_lat) / lat_span * N_BLOCK_ROWS).astype(int), 0, N_BLOCK_ROWS - 1)
    return row * N_BLOCK_COLS + col


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def tree_spread_interval(
    model: RandomForestRegressor, X: pd.DataFrame, low_pct: float = 5.0, high_pct: float = 95.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (mean, low, high) from the spread across individual trees' predictions --
    a quantile-style uncertainty estimate from tree disagreement, not a fitted quantile model."""
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
    tree_preds = np.array([tree.predict(X_arr) for tree in model.estimators_])  # (n_trees, n_rows)
    mean = tree_preds.mean(axis=0)
    low = np.percentile(tree_preds, low_pct, axis=0)
    high = np.percentile(tree_preds, high_pct, axis=0)
    return mean, low, high


def train_and_evaluate(training_df: pd.DataFrame, output_path: Optional[Path] = None) -> dict:
    """Train the RF, evaluate under both CV protocols, save the model bundle if
    output_path is given, and return everything needed to print/inspect the run."""
    if "bbox" not in training_df.columns:
        raise ValueError("training_df must carry a 'bbox' column (see app.ml.dataset.build_training_set)")
    bbox = training_df["bbox"].iloc[0]
    if (training_df["bbox"] != bbox).any():
        raise ValueError("training_df has rows from more than one bbox -- this is not supported")

    if "source" not in training_df.columns:
        raise ValueError("training_df must carry a 'source' column")
    sources = training_df["source"].unique()
    if len(sources) > 1:
        raise ValueError(f"training_df mixes sources {list(sources)} -- refusing to train on mixed LIVE/FIXTURE data")
    source = sources[0]

    X = build_features(training_df, bbox)
    y = training_df["temp_c"].to_numpy()
    blocks = _assign_blocks(training_df["lat"].to_numpy(), training_df["lon"].to_numpy(), bbox)

    model = RandomForestRegressor(**RF_PARAMS)

    random_cv = KFold(n_splits=RANDOM_CV_SPLITS, shuffle=True, random_state=42)
    random_preds = cross_val_predict(model, X, y, cv=random_cv)
    random_scores = _score(y, random_preds)

    n_unique_blocks = len(np.unique(blocks))
    spatial_splits = min(SPATIAL_CV_SPLITS, n_unique_blocks)
    if spatial_splits < 2:
        raise ValueError(
            f"Only {n_unique_blocks} distinct spatial block(s) present -- need >= 2 for spatial-block CV. "
            "The training set's points are too spatially clustered."
        )
    spatial_cv = GroupKFold(n_splits=spatial_splits)
    spatial_preds = cross_val_predict(model, X, y, cv=spatial_cv, groups=blocks)
    spatial_scores = _score(y, spatial_preds)

    baseline_preds = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=spatial_cv, groups=blocks)
    baseline_scores = _score(y, baseline_preds)

    # Permutation importance on one genuinely held-out spatial fold -- NOT the
    # impurity-based .feature_importances_, which is biased toward
    # high-cardinality features, and NOT evaluated on data the model was
    # fit on.
    train_idx, test_idx = next(iter(spatial_cv.split(X, y, groups=blocks)))
    fold_model = RandomForestRegressor(**RF_PARAMS)
    fold_model.fit(X.iloc[train_idx], y[train_idx])
    perm = permutation_importance(
        fold_model, X.iloc[test_idx], y[test_idx], n_repeats=PERMUTATION_REPEATS, random_state=42
    )
    permutation_importances = sorted(
        zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std), key=lambda t: -t[1]
    )

    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(X, y)

    metadata = {
        "n_rows": len(training_df),
        "source": str(source),
        "bbox": bbox,
        "feature_names": FEATURE_NAMES,
        "date_range": (
            [str(training_df["observed_at"].min()), str(training_df["observed_at"].max())]
            if "observed_at" in training_df.columns
            else None
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scores": {
            "random_split": random_scores,
            "spatial_block": spatial_scores,
            "baseline_mean": baseline_scores,
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final_model, "feature_names": FEATURE_NAMES, "metadata": metadata}, output_path)
        logger.info("rf_model_saved path=%s n_rows=%d source=%s", output_path, len(training_df), source)

    return {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "metadata": metadata,
        "random_split_scores": random_scores,
        "spatial_block_scores": spatial_scores,
        "baseline_scores": baseline_scores,
        "permutation_importances": permutation_importances,
    }

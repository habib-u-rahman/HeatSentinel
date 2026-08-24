"""Predicted temperature delta from street-level surface interventions --
the headline feature.

HONESTY CONSTRAINT: our RF is trained on segmentation-class fractions (road,
vegetation, built, ...). Albedo-only interventions (cool roofs, reflective
pavement) change surface REFLECTIVITY, which those features literally cannot
represent -- painting a road white doesn't change its "road" class fraction.
For those we do NOT fake a model prediction; we apply a clearly-labelled
literature-derived offset instead. Every DeltaTResult carries a `method`
field ("model_prediction" vs "literature_offset") so the API/UI never blur
the two together.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from app.ml.features import BUCKET_NAMES, build_features
from app.vision import metrics as vision_metrics

logger = logging.getLogger(__name__)

MODEL_PATH = Path("../models/rf_intervention.pkl")

# Bucket-fraction deltas applied directly to the surface-composition fractions
# (renormalized to sum to 1.0, clipped at 0 first so no bucket goes negative).
# deltas=None marks an intervention our segmentation features cannot see at
# all (see module docstring) -- those use literature_offset_c instead.
INTERVENTIONS: dict[str, dict] = {
    "plant_street_trees": {
        "deltas": {"vegetation": 0.15, "road": -0.10, "sky": -0.05},
        "literature_offset_c": None,
        "source": "typical street-tree planting program (new canopy over road/sky view); illustrative "
        "planning-scale composition change, not a measured retrofit",
    },
    "green_wall": {
        "deltas": {"vegetation": 0.08, "built": -0.08},
        "literature_offset_c": None,
        "source": "green facade retrofit over building walls; illustrative planning-scale composition change",
    },
    "pocket_park": {
        "deltas": {"vegetation": 0.30, "road": -0.20, "sidewalk": -0.10},
        "literature_offset_c": None,
        "source": "converting a small paved lot to a pocket park; illustrative planning-scale composition change",
    },
    "cool_roof": {
        "deltas": None,
        "literature_offset_c": -1.5,
        "source": "cool/reflective roofing raises albedo ~0.15-0.30; ~1-2C local air-temp reduction reported in "
        "the urban-heat literature (e.g. Akbari et al. 2001; Santamouris 2014 meta-analysis range) -- NOT "
        "visible to our segmentation features, so this is a literature offset, not a model prediction",
    },
    "reflective_pavement": {
        "deltas": None,
        "literature_offset_c": -1.0,
        "source": "reflective 'cool pavement' coatings raise road albedo; ~0.5-1.5C local air-temp reduction at "
        "pedestrian height reported (e.g. Middel et al. 2020 field studies) -- NOT visible to our segmentation "
        "features, so this is a literature offset, not a model prediction",
    },
}

LITERATURE_OFFSET_UNCERTAINTY_C = 0.5  # a coarse +/- band around a literature point estimate, NOT a fitted interval


@dataclass
class DeltaTResult:
    intervention: str
    value: float  # predicted temperature change in C (negative = cooling)
    confidence_interval: tuple[float, float]
    method: str  # "model_prediction" or "literature_offset"
    n_training_rows: int
    data_source: str  # "live" or "fixture" -- the underlying MODEL's training data source
    detail: str = ""


@lru_cache(maxsize=4)
def _load_model_bundle(path: str) -> dict:
    return joblib.load(path)


def _apply_deltas(buckets: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    """Apply bucket-fraction deltas, clip at 0, renormalize to sum to 1.0."""
    updated = dict(buckets)
    for name, delta in deltas.items():
        updated[name] = max(0.0, updated.get(name, 0.0) + delta)

    total = sum(updated.values())
    if total <= 0:
        raise ValueError(f"Intervention deltas {deltas} zeroed out every bucket -- nothing left to renormalize")
    return {name: value / total for name, value in updated.items()}


def _row_from_profile(surface_profile: dict, context: dict, buckets: dict[str, float]) -> pd.DataFrame:
    """One-row, dataset.py-shaped dataframe for build_features(), with buckets (and their
    derived metrics, recomputed from app.vision.metrics for consistency) overridden --
    used to build both the baseline and the post-intervention feature rows."""
    row = dict(surface_profile)
    row.update(buckets)
    row["impervious_fraction"] = vision_metrics.impervious_fraction(buckets)
    row["green_view_index"] = vision_metrics.green_view_index(buckets)
    row["sky_view_factor_proxy"] = vision_metrics.sky_view_factor_proxy(buckets)
    row["hour"] = context["hour"]
    row["solar_wm2"] = context["solar_wm2"]
    row["wind_ms"] = context["wind_ms"]
    row["relative_humidity"] = context["relative_humidity"]
    return pd.DataFrame([row])


def predict_delta_t(
    surface_profile: dict,
    intervention_name: str,
    context: dict,
    bbox: str,
    model_path: Optional[Path] = None,
) -> DeltaTResult:
    """Predicted temperature change from applying intervention_name to surface_profile
    under meteorological context (hour, solar_wm2, wind_ms, relative_humidity).
    """
    if intervention_name not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention {intervention_name!r}; choose one of {sorted(INTERVENTIONS)}")
    spec = INTERVENTIONS[intervention_name]

    bundle = _load_model_bundle(str(model_path) if model_path is not None else str(MODEL_PATH))
    metadata = bundle["metadata"]
    n_training_rows = metadata["n_rows"]
    data_source = metadata["source"].lower()

    if spec["deltas"] is None:
        value = spec["literature_offset_c"]
        return DeltaTResult(
            intervention=intervention_name,
            value=value,
            confidence_interval=(value - LITERATURE_OFFSET_UNCERTAINTY_C, value + LITERATURE_OFFSET_UNCERTAINTY_C),
            method="literature_offset",
            n_training_rows=0,  # not derived from the model at all
            data_source=data_source,
            detail=spec["source"],
        )

    model = bundle["model"]
    base_buckets = {name: surface_profile[name] for name in BUCKET_NAMES}
    transformed_buckets = _apply_deltas(base_buckets, spec["deltas"])

    # Individual tree estimators aren't fitted with column-name metadata (only
    # the outer RandomForestRegressor is), so predict on plain arrays to avoid
    # a spurious "fitted without feature names" warning -- column ORDER still
    # matches FEATURE_NAMES via build_features, which is what actually matters.
    base_features = build_features(_row_from_profile(surface_profile, context, base_buckets), bbox).to_numpy()
    transformed_features = build_features(_row_from_profile(surface_profile, context, transformed_buckets), bbox).to_numpy()

    # Paired per-tree difference: the same ensemble predicts both the baseline
    # and the counterfactual, so a given tree's idiosyncratic bias cancels out
    # of the delta instead of inflating the spread of two independent CIs.
    base_tree_preds = np.array([tree.predict(base_features)[0] for tree in model.estimators_])
    transformed_tree_preds = np.array([tree.predict(transformed_features)[0] for tree in model.estimators_])
    deltas_per_tree = transformed_tree_preds - base_tree_preds

    value = float(deltas_per_tree.mean())
    low, high = float(np.percentile(deltas_per_tree, 5)), float(np.percentile(deltas_per_tree, 95))

    return DeltaTResult(
        intervention=intervention_name,
        value=value,
        confidence_interval=(low, high),
        method="model_prediction",
        n_training_rows=n_training_rows,
        data_source=data_source,
        detail=spec["source"],
    )


def rank_interventions(
    surface_profile: dict, context: dict, bbox: str, model_path: Optional[Path] = None
) -> list[DeltaTResult]:
    """All interventions, sorted by predicted cooling (most negative delta first)."""
    results = [predict_delta_t(surface_profile, name, context, bbox, model_path=model_path) for name in INTERVENTIONS]
    return sorted(results, key=lambda r: r.value)


def _demo() -> None:
    training_path = Path("../data/raw/training_set.parquet")
    if not training_path.exists():
        raise SystemExit(
            f"No training set at {training_path} -- run scripts/build_training_set.py and scripts/train_rf.py first"
        )

    df = pd.read_parquet(training_path)
    sample = df.iloc[0]
    bbox = sample["bbox"]

    surface_profile = {name: float(sample[name]) for name in BUCKET_NAMES}
    surface_profile.update(
        {
            "person_count": float(sample["person_count"]),
            "bicycle_count": float(sample["bicycle_count"]),
            "car_count": float(sample["car_count"]),
            "motorcycle_count": float(sample["motorcycle_count"]),
            "bus_count": float(sample["bus_count"]),
            "truck_count": float(sample["truck_count"]),
            "lat": float(sample["lat"]),
            "lon": float(sample["lon"]),
        }
    )
    context = {
        "hour": float(sample["hour"]),
        "solar_wm2": float(sample["solar_wm2"]),
        "wind_ms": float(sample["wind_ms"]),
        "relative_humidity": float(sample["relative_humidity"]),
    }

    print(f"sample point: lat={sample['lat']:.5f} lon={sample['lon']:.5f} observed temp_c={sample['temp_c']:.1f}")
    print(f"surface: {[(n, round(surface_profile[n], 3)) for n in BUCKET_NAMES]}\n")

    for result in rank_interventions(surface_profile, context, bbox):
        ci_low, ci_high = result.confidence_interval
        print(
            f"{result.intervention:22s} delta={result.value:+.2f}C "
            f"[{ci_low:+.2f}, {ci_high:+.2f}] method={result.method:17s} "
            f"n_training_rows={result.n_training_rows} source={result.data_source}"
        )
        if result.detail:
            print(f"    {result.detail}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Rank street-level heat interventions for a sample point")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

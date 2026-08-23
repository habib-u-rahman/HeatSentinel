from __future__ import annotations

BucketMap = dict[str, float]

# Weights for thermal_load_score: how much each surface bucket contributes to
# daytime urban heat load. Hand-tuned starting point, meant to be adjusted once
# we have ground-truth surface-temperature data to fit against.
THERMAL_LOAD_WEIGHTS: dict[str, float] = {
    "road": 0.35,  # asphalt has the lowest albedo of common street surfaces; the dominant urban-heat-island driver
    "built": 0.25,  # building/wall/fence thermal mass absorbs heat by day and re-radiates it into the evening
    "sidewalk": 0.15,  # impervious like road, but usually lighter-coloured and narrower -> smaller per-fraction effect
    "sky": 0.15,  # proxy for unobstructed sky view -> less shading -> more direct solar gain reaching the ground
    "vegetation": -0.30,  # shade + evapotranspiration cooling; the strongest negative (cooling) contributor
    "other": 0.0,  # unclassified pixels (people, vehicles, poles, signs, ...) treated as heat-neutral
}


def green_view_index(buckets: BucketMap) -> float:
    """Fraction of the image occupied by vegetation/terrain. Higher = more street greenery."""
    return buckets["vegetation"]


def sky_view_factor_proxy(buckets: BucketMap) -> float:
    """Fraction of the image occupied by open sky — a cheap 2D proxy for the true hemispherical sky view factor."""
    return buckets["sky"]


def impervious_fraction(buckets: BucketMap) -> float:
    """Fraction of the image covered by impervious surfaces: road + sidewalk + built."""
    return buckets["road"] + buckets["sidewalk"] + buckets["built"]


def thermal_load_score(buckets: BucketMap) -> float:
    """Weighted composite of the surface buckets in [0, 1]; higher = hotter.

    Computes a raw weighted sum over THERMAL_LOAD_WEIGHTS, then min-max
    normalizes it using the weights' own [min, max] range. Because bucket
    fractions form a simplex (non-negative, sum to 1), that range is exactly
    the raw score's theoretical bound, so the result always lands in [0, 1]
    even after the weights above get retuned.
    """
    raw = sum(weight * buckets[name] for name, weight in THERMAL_LOAD_WEIGHTS.items())
    w_min = min(THERMAL_LOAD_WEIGHTS.values())
    w_max = max(THERMAL_LOAD_WEIGHTS.values())
    normalized = (raw - w_min) / (w_max - w_min)
    return max(0.0, min(1.0, normalized))

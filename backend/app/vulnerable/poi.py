"""OSM points-of-interest as a documented PROXY for vulnerable populations.

HONESTY CONSTRAINT: no real population/census dataset exists for this AOI.
These are OpenStreetMap points of interest (schools, clinics, bus stops, ...)
used as a structural PROXY for where vulnerable people are likely to be --
NEVER present this as measured population data. Every consumer of this
module (the API, alerts) must carry population_proxy=True and
POPULATION_PROXY_NOTE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("../data/raw/vulnerable_pois.parquet")

POPULATION_PROXY_NOTE = (
    "No real population or census dataset exists for this AOI. These are OpenStreetMap "
    "points of interest used as a documented PROXY for where vulnerable people are likely "
    "to be -- NOT measured population or census data."
)

# category -> (OSM key, OSM value, vulnerability_group, weight, rationale). weight
# is a 0-1 judgment call for RANKING relative heat vulnerability -- not a
# calibrated epidemiological risk multiplier.
POI_CATEGORIES: dict[str, dict] = {
    "kindergarten": {
        "osm_key": "amenity",
        "osm_value": "kindergarten",
        "vulnerability_group": "children",
        "weight": 1.0,
        "rationale": "youngest children have the least-developed thermoregulation and highest surface-area-to-mass ratio (WHO heat-health guidance)",
    },
    "school": {
        "osm_key": "amenity",
        "osm_value": "school",
        "vulnerability_group": "children",
        "weight": 0.8,
        "rationale": "school-age children are more heat-vulnerable than adults, though less than infants/toddlers (WHO heat-health guidance)",
    },
    "hospital": {
        "osm_key": "amenity",
        "osm_value": "hospital",
        "vulnerability_group": "patients",
        "weight": 1.0,
        "rationale": "hospital patients/visitors include acutely ill people with reduced heat tolerance and mobility",
    },
    "clinic": {
        "osm_key": "amenity",
        "osm_value": "clinic",
        "vulnerability_group": "patients",
        "weight": 0.7,
        "rationale": "outpatient clinics see a vulnerable population, typically less acute than inpatient hospital care",
    },
    "doctors": {
        "osm_key": "amenity",
        "osm_value": "doctors",
        "vulnerability_group": "patients",
        "weight": 0.6,
        "rationale": "individual medical practices; similar rationale to clinics, at smaller scale",
    },
    "nursing_home": {
        "osm_key": "social_facility",
        "osm_value": "nursing_home",
        "vulnerability_group": "elderly",
        "weight": 1.0,
        "rationale": "elderly residents have reduced thermoregulation and are frequently on medications that impair heat response (CDC heat-health guidance)",
    },
    "social_facility": {
        "osm_key": "amenity",
        "osm_value": "social_facility",
        "vulnerability_group": "elderly",
        "weight": 0.7,
        "rationale": "broader social-care facilities; assumed to skew toward elderly/vulnerable populations absent a more specific subtype tag",
    },
    "bus_station": {
        "osm_key": "amenity",
        "osm_value": "bus_station",
        "vulnerability_group": "waiting_exposure",
        "weight": 0.6,
        "rationale": "people waiting for transit have prolonged, involuntary outdoor heat exposure with little control over shade",
    },
    "bus_stop": {
        "osm_key": "highway",
        "osm_value": "bus_stop",
        "vulnerability_group": "waiting_exposure",
        "weight": 0.5,
        "rationale": "same rationale as bus_station; typically less shelter/shade infrastructure than a full station",
    },
    "construction": {
        "osm_key": "landuse",
        "osm_value": "construction",
        "vulnerability_group": "outdoor_workers",
        "weight": 0.9,
        "rationale": "sustained heavy outdoor labour is the highest ACGIH/OSHA work-intensity category for heat stress",
    },
    "marketplace": {
        "osm_key": "amenity",
        "osm_value": "marketplace",
        "vulnerability_group": "informal_workers",
        "weight": 0.7,
        "rationale": "informal/market vendors work outdoors for extended hours with little control over their heat exposure",
    },
}

POI_COLUMNS = ["poi_id", "name", "category", "vulnerability_group", "weight", "lat", "lon"]


class PoiFetchError(Exception):
    """Raised when Overpass can't be reached, or returns nothing usable, after retries."""


def _build_overpass_tags() -> dict[str, list[str]]:
    tags: dict[str, set[str]] = {}
    for spec in POI_CATEGORIES.values():
        tags.setdefault(spec["osm_key"], set()).add(spec["osm_value"])
    return {key: sorted(values) for key, values in tags.items()}


def _classify(row: pd.Series) -> Optional[str]:
    # nursing_home is a MORE SPECIFIC subtag of amenity=social_facility -- check
    # it first so a tagged nursing home doesn't fall back to the generic category.
    if row.get("social_facility") == "nursing_home":
        return "nursing_home"
    amenity = row.get("amenity")
    if isinstance(amenity, str) and amenity in POI_CATEGORIES:
        return amenity
    if row.get("highway") == "bus_stop":
        return "bus_stop"
    if row.get("landuse") == "construction":
        return "construction"
    return None


@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_raw_features(bbox: str):
    """The single Overpass request for every category at once (Overpass rate-limits
    aggressively -- one AOI-wide query, not one per category)."""
    import osmnx as ox

    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    tags = _build_overpass_tags()
    logger.info("overpass_poi_fetch bbox=%s tags=%s", bbox, tags)
    try:
        return ox.features_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat), tags=tags)
    except Exception as exc:
        raise PoiFetchError(f"Overpass POI fetch failed for bbox={bbox}: {exc}") from exc


def fetch_pois(bbox: str, output_path: Path = DEFAULT_OUTPUT_PATH, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (or load from cache) OSM POIs for bbox, classified into the
    vulnerability categories above. Never re-hits Overpass if output_path
    already exists, unless force_refresh=True.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force_refresh:
        logger.info("poi_cache_hit path=%s", output_path)
        return pd.read_parquet(output_path)

    raw = _fetch_raw_features(bbox)
    if raw.empty:
        logger.warning("overpass returned zero features for bbox=%s", bbox)

    records = []
    for (element_type, osm_id), row in raw.iterrows():
        category = _classify(row)
        if category is None:
            continue
        geometry = row.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        centroid = geometry.centroid
        spec = POI_CATEGORIES[category]
        name = row.get("name")
        records.append(
            {
                "poi_id": f"{element_type}/{osm_id}",
                "name": name if isinstance(name, str) else None,
                "category": category,
                "vulnerability_group": spec["vulnerability_group"],
                "weight": spec["weight"],
                "lat": float(centroid.y),
                "lon": float(centroid.x),
            }
        )

    pois_df = pd.DataFrame(records, columns=POI_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pois_df.to_parquet(output_path, index=False)
    logger.info("poi_fetch_complete bbox=%s n_pois=%d", bbox, len(pois_df))
    return pois_df

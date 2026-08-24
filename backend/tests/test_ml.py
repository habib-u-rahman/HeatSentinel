from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for
from app.ml.dataset import build_training_set
from app.ml.features import BUCKET_NAMES, FEATURE_NAMES, build_features
from app.ml.intervention import INTERVENTIONS, _apply_deltas, predict_delta_t, rank_interventions
from app.ml.train_rf import _assign_blocks, train_and_evaluate

BBOX = "-74.020,40.700,-73.995,40.726"


# --- shared helpers ------------------------------------------------------------


def _make_grid(cells: list[tuple[float, float, float]], bbox: str = BBOX, observed_at=None) -> TemperatureGrid:
    return TemperatureGrid(
        cells=[TemperatureCell(lat=lat, lon=lon, temp_c=temp, cell_id=cell_id_for(lat, lon)) for lat, lon, temp in cells],
        bbox=bbox,
        granularity_m=100,
        observed_at=observed_at or datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
        source=GridSource.FIXTURE,
    )


class _StubOpenMeteoClient:
    def fetch_hourly(self, lat, lon, start, end):
        return pd.DataFrame(
            {
                "time": ["2026-07-15T14:00", "2026-07-15T15:00", "2026-07-15T16:00"],
                "temperature_2m": [33.0, 34.0, 33.5],
                "relative_humidity_2m": [55.0, 50.0, 52.0],
                "wind_speed_10m": [2.0, 2.5, 2.2],
                "shortwave_radiation": [700.0, 800.0, 750.0],
            }
        )


def _make_synthetic_training_df(n: int = 300, seed: int = 0, bbox: str = BBOX) -> pd.DataFrame:
    """A clean synthetic dataset where temp_c is a strongly monotonic function of
    impervious_fraction/vegetation -- exactly the physical claim the RF should learn,
    used to test the model pipeline without depending on the real fixture/join scripts."""
    rng = np.random.default_rng(seed)
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    lats = rng.uniform(min_lat, max_lat, n)
    lons = rng.uniform(min_lon, max_lon, n)

    vegetation = rng.uniform(0.0, 0.6, n)
    road = rng.uniform(0.1, 0.6, n)
    remainder = np.clip(1.0 - vegetation - road, 0.0, None)
    sidewalk = remainder * 0.2
    built = remainder * 0.5
    sky = remainder * 0.2
    other = remainder * 0.1
    total = vegetation + road + sidewalk + built + sky + other
    vegetation, road, sidewalk, built, sky, other = (b / total for b in (vegetation, road, sidewalk, built, sky, other))

    impervious_fraction = road + sidewalk + built
    hour = rng.uniform(0, 24, n)
    solar_wm2 = rng.uniform(0, 900, n)
    wind_ms = rng.uniform(0.5, 5, n)
    relative_humidity = rng.uniform(30, 80, n)
    noise = rng.normal(0, 0.3, n)

    temp_c = 28 + 6 * impervious_fraction - 6 * vegetation + 0.002 * solar_wm2 + noise

    return pd.DataFrame(
        {
            "point_id": [f"p{i}" for i in range(n)],
            "lat": lats,
            "lon": lons,
            "road": road,
            "sidewalk": sidewalk,
            "built": built,
            "vegetation": vegetation,
            "sky": sky,
            "other": other,
            "impervious_fraction": impervious_fraction,
            "green_view_index": vegetation,
            "sky_view_factor_proxy": sky,
            "person_count": rng.integers(0, 5, n),
            "bicycle_count": rng.integers(0, 3, n),
            "car_count": rng.integers(0, 10, n),
            "motorcycle_count": rng.integers(0, 2, n),
            "bus_count": rng.integers(0, 2, n),
            "truck_count": rng.integers(0, 2, n),
            "hour": hour,
            "solar_wm2": solar_wm2,
            "wind_ms": wind_ms,
            "relative_humidity": relative_humidity,
            "temp_c": temp_c,
            "source": "FIXTURE",
            "bbox": bbox,
            "observed_at": pd.Timestamp("2026-07-15T15:00:00Z"),
        }
    )


def _sample_profile() -> dict:
    return {
        "road": 0.4,
        "sidewalk": 0.1,
        "built": 0.2,
        "vegetation": 0.2,
        "sky": 0.05,
        "other": 0.05,
        "person_count": 1.0,
        "bicycle_count": 0.0,
        "car_count": 2.0,
        "motorcycle_count": 0.0,
        "bus_count": 0.0,
        "truck_count": 0.0,
        "lat": 40.713,
        "lon": -74.008,
    }


def _sample_context() -> dict:
    return {"hour": 15.0, "solar_wm2": 800.0, "wind_ms": 2.0, "relative_humidity": 45.0}


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    df = _make_synthetic_training_df(n=400, seed=0)
    path = tmp_path_factory.mktemp("model") / "rf_intervention.pkl"
    train_and_evaluate(df, output_path=path)
    return path


# --- dataset ---------------------------------------------------------------------


def test_build_training_set_yields_expected_row_count(tmp_path):
    n_points = 10
    rng = np.random.default_rng(3)
    lats = 40.71 + rng.uniform(-0.005, 0.005, n_points)
    lons = -74.00 + rng.uniform(-0.005, 0.005, n_points)
    point_ids = [f"p{i}" for i in range(n_points)]

    sample_points_df = pd.DataFrame(
        {
            "point_id": point_ids,
            "lat": lats,
            "lon": lons,
            "nearest_edge_key": [f"{i}_{i + 1}_0" for i in range(n_points)],
            "grid_cell_id": ["r0_c0"] * n_points,
        }
    )
    sample_points_path = tmp_path / "sample_points.parquet"
    sample_points_df.to_parquet(sample_points_path, index=False)

    # only 8 of the 10 points get a surface profile (2 dropped: "no image")
    surface_df = pd.DataFrame(
        {
            "point_id": point_ids[:8],
            "lat": lats[:8],
            "lon": lons[:8],
            "road": [0.4] * 8,
            "sidewalk": [0.1] * 8,
            "built": [0.2] * 8,
            "vegetation": [0.2] * 8,
            "sky": [0.05] * 8,
            "other": [0.05] * 8,
            "impervious_fraction": [0.7] * 8,
            "green_view_index": [0.2] * 8,
            "sky_view_factor_proxy": [0.05] * 8,
            "thermal_load_score": [0.6] * 8,
            "person_count": [0] * 8,
            "bicycle_count": [0] * 8,
            "car_count": [1] * 8,
            "motorcycle_count": [0] * 8,
            "bus_count": [0] * 8,
            "truck_count": [0] * 8,
        }
    )
    surface_profiles_path = tmp_path / "surface_profiles.parquet"
    surface_df.to_parquet(surface_profiles_path, index=False)

    # grid only covers 6 of those 8 points within max_dist_m (2 more dropped: "too far")
    grid_cells = [(lats[i], lons[i], 30.0 + i) for i in range(6)]
    grid_cells.append((41.5, -75.5, 20.0))  # far away, irrelevant to any point
    grid = _make_grid(grid_cells, observed_at=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc))

    training_df, report_lines = build_training_set(
        sample_points_path,
        surface_profiles_path,
        [grid],
        openmeteo_client=_StubOpenMeteoClient(),
        max_dist_m=50.0,
        bbox=BBOX,
        min_rows=1,
    )

    assert len(training_df) == 6
    assert (training_df["source"] == "FIXTURE").all()
    assert "temp_c" in training_df.columns
    assert any("10 -> 8" in line for line in report_lines)
    assert any("8 -> 6" in line for line in report_lines)


def test_build_training_set_raises_on_mixed_sources(tmp_path):
    sample_points_path = tmp_path / "sample_points.parquet"
    pd.DataFrame({"point_id": ["p0"], "lat": [40.71], "lon": [-74.0], "nearest_edge_key": ["1_2_0"], "grid_cell_id": ["r0_c0"]}).to_parquet(
        sample_points_path, index=False
    )
    surface_profiles_path = tmp_path / "surface_profiles.parquet"
    pd.DataFrame(
        {
            "point_id": ["p0"],
            "lat": [40.71],
            "lon": [-74.0],
            "road": [0.4],
            "sidewalk": [0.1],
            "built": [0.2],
            "vegetation": [0.2],
            "sky": [0.05],
            "other": [0.05],
            "impervious_fraction": [0.7],
            "green_view_index": [0.2],
            "sky_view_factor_proxy": [0.05],
            "thermal_load_score": [0.6],
            "person_count": [0],
            "bicycle_count": [0],
            "car_count": [1],
            "motorcycle_count": [0],
            "bus_count": [0],
            "truck_count": [0],
        }
    ).to_parquet(surface_profiles_path, index=False)

    live_grid = _make_grid([(40.71, -74.0, 30.0)])
    live_grid = live_grid.model_copy(update={"source": GridSource.LIVE})
    fixture_grid = _make_grid([(40.71, -74.0, 31.0)])

    from app.ml.dataset import MixedSourceTrainingSetError

    with pytest.raises(MixedSourceTrainingSetError):
        build_training_set(sample_points_path, surface_profiles_path, [live_grid, fixture_grid], openmeteo_client=_StubOpenMeteoClient())


# --- features ---------------------------------------------------------------------


def test_feature_order_is_stable_across_calls():
    df = _make_synthetic_training_df(n=5, seed=1)
    f1 = build_features(df, BBOX)
    f2 = build_features(df, BBOX)
    assert list(f1.columns) == FEATURE_NAMES
    pd.testing.assert_frame_equal(f1, f2)


# --- spatial CV ---------------------------------------------------------------------


def test_spatial_cv_holds_out_whole_blocks():
    df = _make_synthetic_training_df(n=200, seed=2)
    blocks = _assign_blocks(df["lat"].to_numpy(), df["lon"].to_numpy(), BBOX)
    n_unique = len(np.unique(blocks))
    cv = GroupKFold(n_splits=min(4, n_unique))

    for train_idx, test_idx in cv.split(df, groups=blocks):
        train_blocks = set(blocks[train_idx])
        test_blocks = set(blocks[test_idx])
        assert train_blocks.isdisjoint(test_blocks)


# --- intervention ---------------------------------------------------------------------


def test_transformed_profile_buckets_sum_to_one():
    buckets = {"road": 0.4, "sidewalk": 0.1, "built": 0.2, "vegetation": 0.2, "sky": 0.05, "other": 0.05}
    transformed = _apply_deltas(buckets, INTERVENTIONS["pocket_park"]["deltas"])
    assert sum(transformed.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in transformed.values())


def test_predict_delta_t_noop_returns_near_zero(trained_model_path, monkeypatch):
    monkeypatch.setitem(INTERVENTIONS, "noop_test", {"deltas": {}, "literature_offset_c": None, "source": "test-only"})
    result = predict_delta_t(_sample_profile(), "noop_test", _sample_context(), BBOX, model_path=trained_model_path)
    assert result.value == pytest.approx(0.0, abs=0.1)
    assert result.method == "model_prediction"


def test_more_vegetation_never_increases_predicted_temp(trained_model_path):
    result = predict_delta_t(_sample_profile(), "pocket_park", _sample_context(), BBOX, model_path=trained_model_path)
    assert result.value <= 0.2  # allow a little model noise slack, but no clear increase


def test_rank_interventions_sorted_by_cooling(trained_model_path):
    results = rank_interventions(_sample_profile(), _sample_context(), BBOX, model_path=trained_model_path)
    values = [r.value for r in results]
    assert values == sorted(values)
    assert {r.intervention for r in results} == set(INTERVENTIONS)


def test_fixture_sourced_results_are_labelled(trained_model_path):
    model_result = predict_delta_t(_sample_profile(), "pocket_park", _sample_context(), BBOX, model_path=trained_model_path)
    assert model_result.data_source == "fixture"
    assert model_result.method == "model_prediction"
    assert model_result.n_training_rows == 400

    literature_result = predict_delta_t(_sample_profile(), "cool_roof", _sample_context(), BBOX, model_path=trained_model_path)
    assert literature_result.method == "literature_offset"
    assert literature_result.n_training_rows == 0
    assert literature_result.data_source == "fixture"

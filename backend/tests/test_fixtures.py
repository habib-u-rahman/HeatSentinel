from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.fortyguard.fixtures import _diurnal_offset_c, generate_grid, generate_series
from app.grid.join import _equirect_project
from app.grid.schema import GridSource

BBOX = "-74.020,40.700,-73.995,40.726"


def test_diurnal_peak_hotter_than_trough():
    assert _diurnal_offset_c(15.0) > _diurnal_offset_c(5.0)
    assert _diurnal_offset_c(15.0) == pytest.approx(5.0)
    assert _diurnal_offset_c(5.0) == pytest.approx(-5.0)


def test_diurnal_curve_is_continuous_at_joins():
    # a small step either side of the peak/trough shouldn't jump discontinuously
    assert _diurnal_offset_c(14.99) == pytest.approx(_diurnal_offset_c(15.0), abs=0.01)
    assert _diurnal_offset_c(5.01) == pytest.approx(_diurnal_offset_c(5.0), abs=0.01)


def test_generate_grid_is_fixture_sourced():
    grid = generate_grid(BBOX, 150, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc), seed=0)
    assert grid.source == GridSource.FIXTURE
    assert len(grid.cells) > 0


def test_uhi_gradient_center_hotter_than_edge():
    grid = generate_grid(BBOX, 100, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc), seed=0)
    lats, lons, temps = grid.to_arrays()

    min_lon, min_lat, max_lon, max_lat = (float(x) for x in BBOX.split(","))
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    x_m, y_m = _equirect_project(lats, lons, center_lat, center_lon)
    dist_from_center = np.hypot(x_m, y_m)

    # average temp in the innermost decile of cells vs. the outermost decile
    order = np.argsort(dist_from_center)
    decile = max(1, len(order) // 10)
    inner_mean = temps[order[:decile]].mean()
    outer_mean = temps[order[-decile:]].mean()

    assert inner_mean > outer_mean


def test_surface_coupling_direction():
    surface_profiles = pd.DataFrame(
        {
            "lat": [40.712, 40.718],
            "lon": [-74.010, -74.003],
            "impervious_fraction": [0.95, 0.05],
            "vegetation": [0.02, 0.85],
        }
    )
    grid = generate_grid(
        BBOX, 50, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc), seed=0, surface_profiles=surface_profiles
    )
    lats, lons, temps = grid.to_arrays()

    dist_to_impervious = np.hypot(lats - 40.712, lons - (-74.010))
    dist_to_vegetated = np.hypot(lats - 40.718, lons - (-74.003))

    near_impervious_temp = temps[np.argmin(dist_to_impervious)]
    near_vegetated_temp = temps[np.argmin(dist_to_vegetated)]

    assert near_impervious_temp > near_vegetated_temp


def test_generate_series_length_and_reuses_noise_texture():
    start = datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 15, 3, 0, tzinfo=timezone.utc)
    series = generate_series(BBOX, start, end, freq="1h", granularity_m=200, seed=7)

    assert len(series) == 4  # 00:00, 01:00, 02:00, 03:00
    assert all(g.source == GridSource.FIXTURE for g in series)
    assert [g.observed_at.hour for g in series] == [0, 1, 2, 3]

    # same seed => the day-to-day spatial texture (grid minus its scalar diurnal
    # offset) is identical across timestamps, only the diurnal component moves.
    _, _, temps_0 = series[0].to_arrays()
    _, _, temps_1 = series[1].to_arrays()
    diffs = temps_1 - temps_0
    assert diffs.std() < 1e-9  # a pure scalar shift, not independent re-randomization

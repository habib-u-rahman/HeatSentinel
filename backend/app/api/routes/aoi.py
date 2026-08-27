"""POST /api/aoi/build, GET /api/aoi/build/{job_id}, GET /api/aoi/current --
on-demand new-city AOI builds (app/pipeline/build_aoi.py has the actual
pipeline: real OSM street network, real Mapillary photos, real CV analysis,
real OSM POIs -- no fabricated data at any stage).

One build at a time, single process: a module-level job registry + a
background thread via asyncio.to_thread, not FastAPI BackgroundTasks (which
only ties work to one request's lifecycle -- the separate polling GET still
needs somewhere to read progress from) and not a queue/Redis (this is a
single-demo-user app; the GIL makes simple dict reads safe enough, and a lock
around writes costs nothing extra).

Single active AOI, not a multi-workspace system: app.state holds exactly one
city's data at a time, matching how the rest of the app already reads it
(see app.api.deps.get_aoi_bbox). A completed build atomically swaps app.state
in _run_job below -- the same loading code app.main's lifespan uses at startup.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    AoiBuildProgress,
    AoiBuildQueuedResponse,
    AoiBuildRequest,
    AoiBuildResultSchema,
    AoiBuildStatusResponse,
    AoiCurrentResponse,
)
from app.core.geo import MAX_AREA_KM2, bbox_around, bbox_area_km2
from app.ingest.geocode import NominatimClient
from app.pipeline.build_aoi import AoiBuildResult, NoImageryError, run_aoi_build, slugify
from app.routing.graph import _annotate_edges

logger = logging.getLogger(__name__)

router = APIRouter()

# Real coverage below this is still shown (never hidden/fabricated up), just
# flagged as degraded rather than presented as a normal success.
DEGRADED_COVERAGE_THRESHOLD_PCT = 25.0
AOIS_BASE_DIR = Path("../data/aois")


@dataclass
class AoiBuildJob:
    job_id: str
    status: str = "queued"  # queued | running | done | failed
    stage: Optional[str] = None
    message: Optional[str] = None
    progress_current: int = 0
    progress_total: int = 0
    result: Optional[AoiBuildResult] = None
    error: Optional[str] = None


_jobs: dict[str, AoiBuildJob] = {}
_jobs_lock = threading.Lock()
_active_job_id: Optional[str] = None


def _run_job(app, job: AoiBuildJob, bbox: str, city_name: str, n_points: int, aoi_dir: Path) -> None:
    """Runs on a worker thread (via asyncio.to_thread) -- the pipeline itself
    is blocking (OSM/Overpass/Mapillary network calls + CPU-bound CV
    inference), so this must never run directly on the event loop.
    """
    global _active_job_id
    job.status = "running"

    def on_progress(stage: str, message: str, current: int, total: int) -> None:
        with _jobs_lock:
            job.stage = stage
            job.message = message
            job.progress_current = current
            job.progress_total = total

    try:
        result = run_aoi_build(bbox, city_name, n_points, aoi_dir, on_progress=on_progress)
    except NoImageryError as exc:
        with _jobs_lock:
            job.status = "failed"
            job.error = str(exc)
        logger.warning("aoi_build_no_imagery job_id=%s city=%s", job.job_id, city_name)
        return
    except Exception as exc:
        logger.exception("aoi_build_failed job_id=%s city=%s", job.job_id, city_name)
        with _jobs_lock:
            job.status = "failed"
            job.error = f"Build failed: {exc}"
        return
    finally:
        with _jobs_lock:
            if _active_job_id == job.job_id:
                _active_job_id = None

    # Success -- swap app.state now, using the exact same load pattern
    # app.main's lifespan uses at startup, so every route (which reads
    # app.state via app.api.deps) picks up the new city immediately.
    import osmnx as ox

    graph = ox.load_graphml(result.graph_path)
    _annotate_edges(graph)
    degraded = result.mapillary_coverage_pct < DEGRADED_COVERAGE_THRESHOLD_PCT

    app.state.graph = graph
    app.state.sample_points_df = pd.read_parquet(result.sample_points_path)
    app.state.surface_profiles_df = pd.read_parquet(result.surface_profiles_path)
    app.state.vulnerable_pois_df = pd.read_parquet(result.vulnerable_pois_path)
    app.state.aoi_bbox = result.bbox
    app.state.city_name = result.city_name
    app.state.aoi_built_at = datetime.now(timezone.utc)
    app.state.aoi_meta = {
        "n_sample_points": result.n_sample_points,
        "n_with_imagery": result.n_with_imagery,
        "mapillary_coverage_pct": result.mapillary_coverage_pct,
        "n_pois": result.n_pois,
        "degraded": degraded,
    }

    with _jobs_lock:
        job.status = "done"
        job.result = result
        job.stage = "done"
        job.message = f"{city_name} is ready."
    logger.info(
        "aoi_build_complete job_id=%s city=%s coverage_pct=%.1f degraded=%s",
        job.job_id,
        city_name,
        result.mapillary_coverage_pct,
        degraded,
    )


def _result_to_schema(result: AoiBuildResult) -> AoiBuildResultSchema:
    return AoiBuildResultSchema(
        aoi_bbox=result.bbox,
        city_name=result.city_name,
        n_sample_points=result.n_sample_points,
        n_with_imagery=result.n_with_imagery,
        mapillary_coverage_pct=result.mapillary_coverage_pct,
        n_pois=result.n_pois,
        degraded=result.mapillary_coverage_pct < DEGRADED_COVERAGE_THRESHOLD_PCT,
    )


@router.post("/aoi/build", response_model=AoiBuildQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
async def post_aoi_build(body: AoiBuildRequest, request: Request) -> AoiBuildQueuedResponse:
    global _active_job_id
    with _jobs_lock:
        if _active_job_id is not None and _jobs[_active_job_id].status in ("queued", "running"):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="A build is already in progress -- wait for it to finish (or fail) before starting another.",
            )

    if body.bbox:
        bbox = body.bbox
        city_name = body.query or "Custom area"
    elif body.query:
        with NominatimClient() as geocoder:
            match = geocoder.geocode(body.query)
        if match is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Couldn't find a location for {body.query!r}. Try a more specific name, or pass an explicit bbox.",
            )
        bbox = bbox_around(match.lat, match.lon, body.radius_km * 1000)
        city_name = match.display_name.split(",")[0].strip() or body.query
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Provide either 'query' (a place name) or an explicit 'bbox'.")

    area_km2 = bbox_area_km2(bbox)
    if area_km2 > MAX_AREA_KM2:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Requested area is {area_km2:.1f} km2, which exceeds the {MAX_AREA_KM2} km2 cap for a live build. Use a smaller radius_km.",
        )

    job_id = uuid.uuid4().hex
    job = AoiBuildJob(job_id=job_id, progress_total=body.n_points)
    # job_id suffix guarantees a unique dir even when slugify(city_name) collapses to
    # the same fallback (e.g. two different non-Latin-script city names both -> "city").
    aoi_dir = AOIS_BASE_DIR / f"{slugify(city_name)}-{job_id[:8]}"

    with _jobs_lock:
        _jobs[job_id] = job
        _active_job_id = job_id

    app = request.app
    asyncio.create_task(asyncio.to_thread(_run_job, app, job, bbox, city_name, body.n_points, aoi_dir))

    return AoiBuildQueuedResponse(job_id=job_id, status="queued", bbox=bbox, city_name=city_name, area_km2=area_km2)


@router.get("/aoi/build/{job_id}", response_model=AoiBuildStatusResponse)
async def get_aoi_build_status(job_id: str) -> AoiBuildStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown job_id {job_id!r}")

    return AoiBuildStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        message=job.message,
        progress=AoiBuildProgress(current=job.progress_current, total=job.progress_total),
        result=_result_to_schema(job.result) if job.result is not None else None,
        error=job.error,
    )


@router.get("/aoi/current", response_model=AoiCurrentResponse)
async def get_aoi_current(request: Request) -> AoiCurrentResponse:
    state = request.app.state
    meta = getattr(state, "aoi_meta", None) or {}
    return AoiCurrentResponse(
        city_name=state.city_name,
        aoi_bbox=state.aoi_bbox,
        built_at=getattr(state, "aoi_built_at", None),
        n_sample_points=meta.get("n_sample_points"),
        n_with_imagery=meta.get("n_with_imagery"),
        mapillary_coverage_pct=meta.get("mapillary_coverage_pct"),
        n_pois=meta.get("n_pois"),
        degraded=meta.get("degraded", False),
    )

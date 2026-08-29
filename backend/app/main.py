"""HeatSentinel HTTP API.

Preloads the walk graph + RF model ONCE at startup (see lifespan) -- cold
loading a 7,000+ edge graph inside a request handler would time out. Run
from the backend/ directory:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from contextlib import asynccontextmanager
from pathlib import Path

# osmnx eagerly imports matplotlib.pyplot at package-import time purely to
# expose plotting helpers (ox.plot_graph etc.) that this app never calls --
# confirmed costing ~23MB (measured: 100.5MB vs 77.1MB) on a Render
# free-tier instance where every MB matters against the 512MB ceiling.
# Stubbing before any app/osmnx import intercepts it safely: osmnx's own
# import still succeeds (verified), it just never gets a real plotting
# backend, which is fine since nothing here calls osmnx's plot functions.
if "matplotlib" not in sys.modules:
    _mpl_stub = types.ModuleType("matplotlib")
    _mpl_stub.pyplot = types.ModuleType("matplotlib.pyplot")
    _mpl_stub.colors = types.ModuleType("matplotlib.colors")
    _mpl_stub.cm = types.ModuleType("matplotlib.cm")
    sys.modules["matplotlib"] = _mpl_stub
    sys.modules["matplotlib.pyplot"] = _mpl_stub.pyplot
    sys.modules["matplotlib.colors"] = _mpl_stub.colors
    sys.modules["matplotlib.cm"] = _mpl_stub.cm

import joblib
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import (
    NoGridAvailableError,
    OutOfAOIError,
    ServiceNotReadyError,
    UnknownInterventionError,
    UnknownPointIdError,
)
from app.api.routes import alerts, aoi, grid, health, intervention, points, route, vulnerable
from app.config import get_settings
from app.ingest.openmeteo import OpenMeteoClient
from app.routing.graph import _annotate_edges
from app.routing.router import RouteNotFoundError, SnapDistanceExceededError

logger = logging.getLogger(__name__)

# backend/app/main.py -> backend/app -> backend -> repo root. Anchored on
# __file__ (not a "../" relative path) so these resolve correctly regardless
# of the process's working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONNECTED_GRAPH_FILENAME = "walk_graph_connected.graphml"
MODEL_PATH = _PROJECT_ROOT / "models" / "rf_intervention.pkl"
SAMPLE_POINTS_PATH = _PROJECT_ROOT / "data" / "raw" / "sample_points.parquet"
SURFACE_PROFILES_PATH = _PROJECT_ROOT / "data" / "raw" / "surface_profiles.parquet"
VULNERABLE_POIS_PATH = _PROJECT_ROOT / "data" / "raw" / "vulnerable_pois.parquet"
STATIC_IMAGES_DIR = _PROJECT_ROOT / "data" / "street_images"
STATIC_OVERLAYS_DIR = _PROJECT_ROOT / "data" / "overlays"
STATIC_CACHE_MAX_AGE_S = 3600

# Extra allowed origins (e.g. the deployed frontend URL) come from the
# CORS_EXTRA_ORIGINS env var (comma-separated) so a new frontend deployment
# doesn't require a code change -- just an env var + redeploy.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
] + [origin.strip() for origin in os.environ.get("CORS_EXTRA_ORIGINS", "").split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # The active AOI -- the `.env` Rawalpindi default until a build_aoi.py run
    # (see app/api/routes/aoi.py) swaps these in-place. Every route reads
    # request.app.state.aoi_bbox, never settings.AOI_BBOX directly, so a
    # completed build actually takes effect without a process restart.
    app.state.aoi_bbox = settings.AOI_BBOX
    app.state.city_name = settings.CITY_NAME
    app.state.aoi_built_at = None  # None = still the .env default, not a live build
    app.state.aoi_meta = None  # AoiBuildResult-shaped dict once a build has run

    graph_path = Path(settings.CACHE_DIR) / CONNECTED_GRAPH_FILENAME
    if graph_path.exists():
        import osmnx as ox

        graph = ox.load_graphml(graph_path)
        _annotate_edges(graph)
        app.state.graph = graph
        logger.info("startup: loaded walk graph nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())
    else:
        app.state.graph = None
        logger.warning("startup: no connected graph cache at %s -- run scripts/build_graph.py", graph_path)

    if MODEL_PATH.exists():
        app.state.model_bundle = joblib.load(MODEL_PATH)
        logger.info("startup: loaded RF model n_rows=%d", app.state.model_bundle["metadata"]["n_rows"])
    else:
        app.state.model_bundle = None
        logger.warning("startup: no RF model at %s -- run scripts/train_rf.py", MODEL_PATH)

    app.state.sample_points_df = pd.read_parquet(SAMPLE_POINTS_PATH) if SAMPLE_POINTS_PATH.exists() else None
    app.state.surface_profiles_df = pd.read_parquet(SURFACE_PROFILES_PATH) if SURFACE_PROFILES_PATH.exists() else None

    if VULNERABLE_POIS_PATH.exists():
        app.state.vulnerable_pois_df = pd.read_parquet(VULNERABLE_POIS_PATH)
        logger.info("startup: loaded vulnerable POIs n=%d", len(app.state.vulnerable_pois_df))
    else:
        app.state.vulnerable_pois_df = None
        logger.warning("startup: no vulnerable POI cache at %s -- run scripts/fetch_pois.py", VULNERABLE_POIS_PATH)

    app.state.openmeteo_client = OpenMeteoClient()

    yield

    app.state.openmeteo_client.close()


app = FastAPI(title="HeatSentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # grid payloads are large

# Static street-level images + segmentation overlays, for the demo. Directories
# are created (possibly empty) so the mount never fails even before any
# images/overlays have been fetched/generated -- individual missing files still
# 404 normally. mkdir is a no-op when the dir already exists (true whenever
# these are bundled into the deployment), so this stays safe even on a
# read-only serverless filesystem; it only actually attempts a write when the
# dir is genuinely missing, which OSError-guards against crashing app startup.
try:
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    logger.warning("static_dir_mkdir_failed -- read-only filesystem and dirs not bundled")
app.mount("/static/images", StaticFiles(directory=STATIC_IMAGES_DIR), name="static_images")
app.mount("/static/overlays", StaticFiles(directory=STATIC_OVERLAYS_DIR), name="static_overlays")


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("request path=%s method=%s status=%d elapsed_ms=%.1f", request.url.path, request.method, response.status_code, elapsed_ms)
    return response


@app.middleware("http")
async def _static_cache_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = f"public, max-age={STATIC_CACHE_MAX_AGE_S}"
    return response


# --- domain exception -> clean 4xx/503 JSON, never a raw traceback -------------


def _error_handler(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


app.add_exception_handler(OutOfAOIError, _error_handler(status.HTTP_400_BAD_REQUEST))
app.add_exception_handler(SnapDistanceExceededError, _error_handler(status.HTTP_400_BAD_REQUEST))
app.add_exception_handler(NoGridAvailableError, _error_handler(status.HTTP_400_BAD_REQUEST))
app.add_exception_handler(RouteNotFoundError, _error_handler(status.HTTP_404_NOT_FOUND))
app.add_exception_handler(UnknownPointIdError, _error_handler(status.HTTP_404_NOT_FOUND))
app.add_exception_handler(UnknownInterventionError, _error_handler(status.HTTP_404_NOT_FOUND))
app.add_exception_handler(ServiceNotReadyError, _error_handler(status.HTTP_503_SERVICE_UNAVAILABLE))


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error"})


app.include_router(health.router, prefix="/api")
app.include_router(grid.router, prefix="/api")
app.include_router(route.router, prefix="/api")
app.include_router(points.router, prefix="/api")
app.include_router(intervention.router, prefix="/api")
app.include_router(vulnerable.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(aoi.router, prefix="/api")

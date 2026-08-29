from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root. Anchored on
# __file__ rather than a "../" relative path so CACHE_DIR resolves correctly
# regardless of the process's working directory (e.g. serverless runtimes
# that don't cd into backend/ before starting the app).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    FORTYGUARD_API_KEY: str
    FORTYGUARD_BASE_URL: str = "https://api.fortyguard.com/v1"
    GROQ_API_KEY: str
    CITY_NAME: str
    AOI_BBOX: str  # "min_lon,min_lat,max_lon,max_lat"
    GRID_GRANULARITY_M: int = 100
    CACHE_DIR: Path = _PROJECT_ROOT / "data" / "cache"
    MAPILLARY_TOKEN: str = ""  # optional: only needed by scripts/fetch_images.py
    ALLOW_FIXTURE_DATA: bool = True  # dev default; flip to False before a demo so app.grid.store refuses to load synthetic FIXTURE grids


@lru_cache
def get_settings() -> Settings:
    return Settings()

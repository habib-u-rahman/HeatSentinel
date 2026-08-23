from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    FORTYGUARD_API_KEY: str
    FORTYGUARD_BASE_URL: str = "https://api.fortyguard.com/v1"
    GROQ_API_KEY: str
    CITY_NAME: str
    AOI_BBOX: str  # "min_lon,min_lat,max_lon,max_lat"
    GRID_GRANULARITY_M: int = 100
    CACHE_DIR: Path = Path("../data/cache")
    MAPILLARY_TOKEN: str = ""  # optional: only needed by scripts/fetch_images.py
    ALLOW_FIXTURE_DATA: bool = True  # dev default; flip to False before a demo so app.grid.store refuses to load synthetic FIXTURE grids


@lru_cache
def get_settings() -> Settings:
    return Settings()

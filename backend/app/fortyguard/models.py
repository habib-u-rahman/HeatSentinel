from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[list[float]]]


class DateTimeFilter(BaseModel):
    start_date: str
    start_time: str
    filter_type: str


class HeatmapSubmitRequest(BaseModel):
    polygon_aoi: GeoJSONPolygon
    date_time: DateTimeFilter
    granularity: int


class HeatmapSubmitResponse(BaseModel):
    """Schema is unconfirmed — kept permissive. VERIFY AGAINST DOCS."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def activity_id(self) -> Optional[str]:
        return self.data.get("activity_id")


class HeatmapPollResponse(BaseModel):
    """Schema is unconfirmed — kept permissive. VERIFY AGAINST DOCS."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def status(self) -> Optional[str]:
        return self.data.get("status")

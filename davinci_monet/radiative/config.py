"""Pydantic configuration models for radiative analysis."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

VALID_PLOT_TYPES = frozenset(
    {
        "toa_event_fields",
        "anomaly_maps",
        "surface_flux",
        "sw_vs_aod_scatter",
        "daily_correlation",
        "spatial_comparison",
        "site_timeseries",
        "surface_impact",
        "surface_dimming_timeseries",
        "method_comparison",
    }
)


class EventConfig(BaseModel):
    """Smoke/aerosol event definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    start_time: datetime
    end_time: datetime
    domain: tuple[float, float, float, float]  # (west, east, south, north)
    background_window: int = 3
    peak_date: date | None = None  # if None, auto-select from event window

    @model_validator(mode="after")
    def _check_time_order(self) -> EventConfig:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class CeresConfig(BaseModel):
    """CERES satellite data configuration."""

    model_config = ConfigDict(extra="forbid")

    product: Literal["ebaf", "syn1deg"]
    source: Literal["local", "opendap"]
    files: str | None = None
    variables: list[str] | None = None

    @model_validator(mode="after")
    def _check_local_has_files(self) -> CeresConfig:
        if self.source == "local" and not self.files:
            raise ValueError("files is required when source is local")
        return self


class Merra2Config(BaseModel):
    """MERRA-2 reanalysis configuration."""

    model_config = ConfigDict(extra="forbid")

    files: str
    smoke_species: list[str] = ["OCEXTTAU", "BCEXTTAU"]


class SiteConfig(BaseModel):
    """A named observation site with coordinates."""

    model_config = ConfigDict(extra="forbid")

    name: str  # display name, e.g. "Missoula MT"
    latitude: float
    longitude: float
    aeronet_id: str  # AERONET site name, e.g. "Missoula"


class AeronetConfig(BaseModel):
    """AERONET ground station configuration."""

    model_config = ConfigDict(extra="forbid")

    files: str
    sites: list[str] | None = None


class SurfaceImpactConfig(BaseModel):
    """Surface radiative impact estimation configuration."""

    model_config = ConfigDict(extra="forbid")

    method: list[Literal["merra2", "semi_empirical"]] = ["merra2", "semi_empirical"]
    ssa: float = 0.92
    asymmetry: float = 0.65


class RadiativeConfig(BaseModel):
    """Top-level radiative analysis configuration."""

    model_config = ConfigDict(extra="forbid")

    event: EventConfig
    ceres: CeresConfig
    merra2: Merra2Config | None = None
    aeronet: AeronetConfig | None = None
    sites: list[SiteConfig] | None = None
    surface_impact: SurfaceImpactConfig | None = None
    plots: list[str]
    output_dir: str

    @model_validator(mode="after")
    def _check_constraints(self) -> RadiativeConfig:
        if self.surface_impact is not None and self.merra2 is None:
            raise ValueError("surface_impact requires merra2 configuration")
        invalid = set(self.plots) - VALID_PLOT_TYPES
        if invalid:
            raise ValueError(
                f"Invalid plot types: {invalid}. Valid: {sorted(VALID_PLOT_TYPES)}"
            )
        return self

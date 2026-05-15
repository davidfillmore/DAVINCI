"""Pydantic configuration schema for the PlumeSentinel add-on workflow."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class GridConfig(BaseModel):
    """Gridding specification for swath-to-grid binning."""

    resolution: float
    lon_range: list[float]
    lat_range: list[float]
    min_obs_count: int = 1


class ProjectionConfig(BaseModel):
    """Map projection specification."""

    type: str
    central_longitude: float = 0.0
    central_latitude: float = 0.0


class GibsBackgroundConfig(BaseModel):
    """NASA GIBS WMTS background tile specification."""

    type: str  # "gibs_wmts"
    layer: str
    date: str


class InputSpec(BaseModel, extra="allow"):
    """Specification for a single data input (GOES, HMS, MODIS, etc.)."""

    type: str  # goes_truecolor, hms_smoke, modis_l2_aod
    file: str | None = None
    files: list[str] | None = None
    gamma: float = 1.8
    variable: str | None = None
    valid_range: list[float] | None = None
    grid: GridConfig | None = None


class PlotSpec(BaseModel, extra="allow"):
    """Specification for a single plot output."""

    type: str
    background: str | GibsBackgroundConfig | None = None
    overlays: list[str] | None = None
    field: str | None = None
    extent: list[float] | None = None
    projection: ProjectionConfig | None = None
    title: str | None = None
    cmap: str | None = None
    alpha: float = 0.7
    colorbar_label: str | None = None

    @field_validator("background", mode="before")
    @classmethod
    def parse_background(cls, v):  # noqa: ANN001, ANN201
        """Parse dict background into GibsBackgroundConfig."""
        if isinstance(v, dict):
            return GibsBackgroundConfig(**v)
        return v


class MqttConfig(BaseModel):
    """MQTT broker connection and publish parameters."""

    broker: str = "broker.hivemq.com"
    topic: str
    port: int = 1883
    qos: int = 0


class BulletinConfig(BaseModel):
    """Configuration for the bulletin generation stage."""

    template: str | None = None
    output_filename: str = "bulletin.txt"
    model: str = "claude-sonnet-4-6"
    include_images: bool = False
    api_key_env: str = "ANTHROPIC_API_KEY"
    mqtt: MqttConfig | None = None
    on_error: str = "warn"

    @field_validator("on_error")
    @classmethod
    def _on_error_must_be_warn(cls, v: str) -> str:
        if v != "warn":
            raise ValueError(
                'on_error currently only supports "warn"; "fail" is reserved'
            )
        return v

    @field_validator("mqtt", mode="before")
    @classmethod
    def _parse_mqtt(cls, v):  # noqa: ANN001, ANN201
        if isinstance(v, dict):
            return MqttConfig(**v)
        return v


class PlumeSentinelConfig(BaseModel):
    """Top-level configuration for the PlumeSentinel add-on workflow."""

    inputs: dict[str, InputSpec]
    plots: dict[str, PlotSpec]
    bulletin: BulletinConfig | None = None

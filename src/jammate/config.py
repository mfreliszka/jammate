"""Central typed configuration.

All tunable values live here instead of being scattered across modules. Sections are added by
the implementation steps that need them; keep each one small and flat.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioConfig(BaseModel):
    """Audio stream settings. 48 kHz is the expected system sample rate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_device_name: str | None = Field(
        default=None, description="Device name or unique substring; None = system default."
    )
    output_device_name: str | None = Field(
        default=None, description="Device name or unique substring; None = system default."
    )
    sample_rate_hz: int = Field(default=48_000, gt=0)
    block_size_frames: int = Field(default=128, gt=0)
    fallback_block_size_frames: int = Field(default=256, gt=0)
    input_channel_index: int = Field(
        default=0, ge=0, description="Zero-based channel of the input device carrying the bass."
    )
    monitor_gain: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Linear gain applied to the bass monitor signal."
    )
    input_ring_seconds: float = Field(
        default=2.0,
        gt=0.0,
        le=60.0,
        description="Capacity of the ring buffer between the audio callback and analysis.",
    )

    @property
    def input_ring_capacity_frames(self) -> int:
        return round(self.sample_rate_hz * self.input_ring_seconds)

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.fallback_block_size_frames < self.block_size_frames:
            raise ValueError("fallback_block_size_frames must be >= block_size_frames")
        if self.input_ring_capacity_frames < self.fallback_block_size_frames:
            raise ValueError("input ring must hold at least one fallback-sized block")
        return self


class JamMateConfig(BaseModel):
    """Root configuration object passed to subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio: AudioConfig = Field(default_factory=AudioConfig)

"""Central typed configuration.

All tunable values live here instead of being scattered across modules. Sections are added by
the implementation steps that need them; keep each one small and flat.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Instrument = Literal["bass", "guitar"]

# Default pitch-detection range per instrument (§15, D7): bass from below a 4-string's low E1,
# guitar from drop-C (C2) up to the 24th fret (E6).
INSTRUMENT_PITCH_RANGES_HZ: dict[Instrument, tuple[float, float]] = {
    "bass": (35.0, 400.0),
    "guitar": (60.0, 1400.0),
}


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


class AnalysisConfig(BaseModel):
    """Analysis worker, pitch detection and note stabilization (Step 2.2, §15, §25, §26)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: Instrument = Field(
        default="bass", description="Selects the default pitch range (D7)."
    )
    pitch_min_hz: float | None = Field(
        default=None, gt=0.0, description="Overrides the instrument's lowest detectable pitch."
    )
    pitch_max_hz: float | None = Field(
        default=None, gt=0.0, description="Overrides the instrument's highest detectable pitch."
    )
    hop_frames: int = Field(
        default=480, gt=0, description="Frames between pitch estimates (480 = 10 ms at 48 kHz)."
    )
    level_gate_dbfs: float = Field(
        default=-45.0,
        le=0.0,
        description="Window RMS below this is silence. Above ringing unmuted strings (-50 dBFS).",
    )
    new_note_gate_db: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "A new note must be within this many dB of the recent playing level. Rejects "
            "harmonics left by muting, open strings ringing after a fretted note is released, "
            "and touch noise. It never ends the current note."
        ),
    )
    playing_level_release_db_per_second: float = Field(
        default=3.0,
        gt=0.0,
        description="How fast the recent playing level falls back after a loud note.",
    )
    yin_threshold: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="YIN aperiodicity threshold: voiced only below it (confidence = 1 - value).",
    )
    min_note_seconds: float = Field(
        default=0.07,
        gt=0.0,
        description="A pitch must be stable this long to become a note (skips slide stops).",
    )
    stability_cents: float = Field(
        default=35.0,
        gt=0.0,
        description="Max pitch spread over `min_note_seconds`; more is a slide, bend or glide.",
    )
    tuning_tolerance_cents: float = Field(
        default=40.0,
        gt=0.0,
        lt=50.0,
        description="A pitch further than this from every note is between notes: no new note.",
    )
    hysteresis_cents: float = Field(
        default=70.0,
        gt=0.0,
        lt=100.0,
        description="The current note is kept while the pitch stays within this many cents.",
    )
    octave_confirm_seconds: float = Field(
        default=0.15,
        gt=0.0,
        description="An octave jump from the current note must be stable this long (suppression).",
    )
    silence_hold_seconds: float = Field(
        default=0.06,
        ge=0.0,
        description="Unvoiced this long ends the note (bridges brief dropouts between picks).",
    )
    max_backlog_seconds: float = Field(
        default=0.1,
        gt=0.0,
        description="When unread input exceeds this, the worker skips to the latest audio (§23).",
    )

    @property
    def pitch_range_hz(self) -> tuple[float, float]:
        default_min_hz, default_max_hz = INSTRUMENT_PITCH_RANGES_HZ[self.instrument]
        return (self.pitch_min_hz or default_min_hz, self.pitch_max_hz or default_max_hz)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        min_hz, max_hz = self.pitch_range_hz
        if min_hz >= max_hz:
            raise ValueError(f"pitch range must have min < max, got {min_hz}-{max_hz} Hz")
        if self.hysteresis_cents <= self.tuning_tolerance_cents:
            raise ValueError("hysteresis_cents must be larger than tuning_tolerance_cents")
        return self


class JamMateConfig(BaseModel):
    """Root configuration object passed to subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio: AudioConfig = Field(default_factory=AudioConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def _check_pitch_range_fits_sample_rate(self) -> Self:
        if self.analysis.pitch_range_hz[1] >= self.audio.sample_rate_hz / 4:
            raise ValueError("pitch_max_hz must be below a quarter of the sample rate")
        return self

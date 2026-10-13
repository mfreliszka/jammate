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


class OnsetConfig(BaseModel):
    """Onset detection (Step 2.3, §27)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold_mads: float = Field(
        default=3.0,
        gt=0.0,
        description="Adaptive threshold: median + this many MADs of the recent detection values.",
    )
    min_detection: float = Field(
        default=6.0,
        gt=0.0,
        description="Absolute floor of the detection function, so steady noise never triggers.",
    )
    threshold_window_seconds: float = Field(
        default=0.5, gt=0.0, description="History used for the adaptive threshold."
    )
    level_rise_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of the hop level rise (dB) added to the spectral flux.",
    )
    refractory_seconds: float = Field(
        default=0.08,
        gt=0.0,
        description="No second onset within this time (one pick can register twice ~35 ms apart).",
    )
    min_level_dbfs: float = Field(
        default=-50.0,
        le=0.0,
        description="The window RMS just after an onset must reach this (rejects handling noise).",
    )


class IntensityConfig(BaseModel):
    """How hard the player is playing, 0.0-1.0 (Step 2.3, §47).

    The level mappings are absolute, so they depend on the interface gain; the defaults suit
    playing peaks around -6 dBFS.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quiet_dbfs: float = Field(default=-50.0, description="Window RMS that counts as 0.")
    loud_dbfs: float = Field(default=-6.0, description="Window RMS that counts as 1.")
    attack_quiet_dbfs: float = Field(default=-40.0, description="Attack peak that counts as 0.")
    attack_loud_dbfs: float = Field(default=0.0, description="Attack peak that counts as 1.")
    full_density_onsets_per_second: float = Field(
        default=8.0, gt=0.0, description="Note density that counts as 1 (16ths at 120 BPM)."
    )
    window_seconds: float = Field(
        default=2.0, gt=0.0, description="Attacks in this recent window give strength and density."
    )
    level_weight: float = Field(default=0.5, ge=0.0)
    attack_weight: float = Field(default=0.25, ge=0.0)
    density_weight: float = Field(default=0.25, ge=0.0)
    rise_seconds: float = Field(
        default=0.2, gt=0.0, description="Smoothing time constant while intensity rises."
    )
    fall_seconds: float = Field(
        default=1.0, gt=0.0, description="Smoothing time constant while intensity falls."
    )
    report_step: float = Field(
        default=0.05, gt=0.0, le=1.0, description="Emit an update when it moved this much."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.quiet_dbfs >= self.loud_dbfs or self.attack_quiet_dbfs >= self.attack_loud_dbfs:
            raise ValueError("each quiet level must be below its loud level")
        if self.level_weight + self.attack_weight + self.density_weight <= 0:
            raise ValueError("at least one intensity weight must be positive")
        return self


class PlayingStateConfig(BaseModel):
    """Whether the player is playing, and why not (Step 2.3, §70). Uses `level_gate_dbfs`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hold_seconds: float = Field(
        default=1.0, gt=0.0, description="An attack keeps the state 'playing' this long."
    )
    gate_hysteresis_db: float = Field(
        default=3.0,
        ge=0.0,
        description="Once above the level gate, the level must fall this far below it to leave.",
    )
    noise_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Above the gate this long with no pitch, no attack and a steady level: noise.",
    )
    noise_level_range_db: float = Field(
        default=6.0, gt=0.0, description="'Steady': the level varied less than this."
    )


class TempoConfig(BaseModel):
    """Tempo estimation and locking from onsets (Step 2.4, §28, §29)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_bpm: float = Field(default=50.0, gt=0.0, description="Slowest tempo ever reported.")
    max_bpm: float = Field(default=200.0, gt=0.0, description="Fastest tempo ever reported.")
    preferred_bpm: float = Field(
        default=100.0,
        gt=0.0,
        description=(
            "Before a lock, tempos are reported in the octave around this (71-141 BPM): onsets "
            "fit half and double the tempo equally well, so the octave decides (D9)."
        ),
    )
    window_seconds: float = Field(
        default=10.0, gt=0.0, description="Onsets older than this give no tempo evidence."
    )
    recency_seconds: float = Field(
        default=6.0,
        gt=0.0,
        description="Time constant of the onset weights: recent onsets count most.",
    )
    min_onsets: int = Field(
        default=4, ge=3, description="Onsets in the window needed for an estimate."
    )
    timing_tolerance_seconds: float = Field(
        default=0.025,
        gt=0.0,
        description="How far off a beat grid an onset can be and still count as on it.",
    )
    max_median_interval_seconds: float = Field(
        default=1.8,
        gt=0.0,
        description="Sparser onsets (median gap above this) give no estimate: no pulse to hear.",
    )
    agreement_percent: float = Field(
        default=3.0,
        gt=0.0,
        lt=20.0,
        description="Two tempo estimates within this many percent agree.",
    )
    min_clarity: float = Field(
        default=0.6,
        gt=0.0,
        lt=1.0,
        description="Only an estimate at least this clear can lock or keep the tempo.",
    )
    lock_seconds: float = Field(
        default=1.5,
        ge=0.0,
        description="Estimates must agree this long, and `lock_estimates` times, to lock.",
    )
    lock_estimates: int = Field(default=4, ge=2)
    lock_significance: float = Field(
        default=6.0,
        ge=0.0,
        description=(
            "To lock, the latest estimate's effective onset count x clarity^2 must reach this. "
            "A few random onsets always fit some grid; this asks for enough of them."
        ),
    )
    unlock_seconds: float = Field(
        default=3.0,
        gt=0.0,
        description="A locked tempo is dropped after this long of disagreeing estimates.",
    )
    adaptation: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="A locked tempo moves this fraction of the way to each agreeing estimate.",
    )
    confidence_rate: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Confidence moves this fraction of the way to each estimate's support.",
    )
    report_bpm_step: float = Field(
        default=0.1, gt=0.0, description="Emit an update when the tempo moved this much."
    )
    report_confidence_step: float = Field(
        default=0.05, gt=0.0, le=1.0, description="Emit an update when confidence moved this much."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_bpm < 2 * self.min_bpm:
            raise ValueError("max_bpm must be at least twice min_bpm (one full tempo octave)")
        if not self.min_bpm <= self.preferred_bpm <= self.max_bpm:
            raise ValueError("preferred_bpm must lie between min_bpm and max_bpm")
        return self


class AnalysisConfig(BaseModel):
    """Analysis worker, pitch detection and note stabilization (Step 2.2, §15, §25, §26).

    The pitch settings are the top-level fields; later analysis stages have their own sections.
    """

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
            "A new note or onset must be within this many dB of the recent playing level. "
            "Rejects harmonics left by muting, open strings ringing after a fretted note is "
            "released, and touch noise. It never ends the current note."
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
    onset: OnsetConfig = Field(default_factory=OnsetConfig)
    intensity: IntensityConfig = Field(default_factory=IntensityConfig)
    playing: PlayingStateConfig = Field(default_factory=PlayingStateConfig)
    tempo: TempoConfig = Field(default_factory=TempoConfig)

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


class MusicalStateConfig(BaseModel):
    """The Musical State (Step 2.5, §36)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recent_note_count: int = Field(
        default=8, ge=1, le=64, description="How many recent notes the state keeps."
    )


class JamMateConfig(BaseModel):
    """Root configuration object passed to subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio: AudioConfig = Field(default_factory=AudioConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    state: MusicalStateConfig = Field(default_factory=MusicalStateConfig)

    @model_validator(mode="after")
    def _check_pitch_range_fits_sample_rate(self) -> Self:
        if self.analysis.pitch_range_hz[1] >= self.audio.sample_rate_hz / 4:
            raise ValueError("pitch_max_hz must be below a quarter of the sample rate")
        return self

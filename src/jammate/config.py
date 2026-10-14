"""Central typed configuration.

All tunable values live here instead of being scattered across modules. Sections are added by
the implementation steps that need them; keep each one small and flat.
"""

import logging
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

type Instrument = Literal["bass", "guitar"]
type StyleFamily = Literal["rock", "stoner_rock", "blues", "funk", "metal"]
type ModelSize = Literal["mrt2_small", "mrt2_base"]
type MeterName = Literal["4/4", "3/4", "6/8"]

METER_BEATS_PER_BAR: dict[MeterName, int] = {
    "4/4": 4,
    "3/4": 3,
    "6/8": 6,
}

MODEL_SIZE_LABELS: dict[ModelSize, str] = {
    "mrt2_small": "Low Quality (mrt2_small)",
    "mrt2_base": "High Quality (mrt2_base)",
}

INSTRUMENT_LABELS: dict[Instrument, str] = {
    "guitar": "Electric Guitar",
    "bass": "Bass Guitar",
}

DEFAULT_KIT_DIR = Path(__file__).resolve().parents[2] / "assets" / "drums"
DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "recordings"
DEFAULT_PREFERENCES_PATH = (
    Path.home() / "Library" / "Application Support" / "JamMate" / "settings.json"
)

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
    monitor_muted: bool = Field(default=False, description="Mute the bass monitor output bus.")
    drums_gain: float = Field(
        default=0.8, ge=0.0, le=4.0, description="Linear gain applied to the drum sampler bus."
    )
    drums_muted: bool = Field(default=False, description="Mute the drum sampler output bus.")
    ai_gain: float = Field(
        default=0.7, ge=0.0, le=4.0, description="Linear gain applied to the AI accompaniment bus."
    )
    ai_muted: bool = Field(default=False, description="Mute the AI accompaniment output bus.")
    soft_clip_threshold: float = Field(
        default=0.85,
        gt=0.0,
        lt=1.0,
        description="Linear amplitude up to this level; above it, a tanh knee limits at 1.0.",
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
    manual_bpm: float | None = Field(
        default=None,
        description="When set, locks tempo at this BPM immediately and bypasses estimation (§58).",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_bpm < 2 * self.min_bpm:
            raise ValueError("max_bpm must be at least twice min_bpm (one full tempo octave)")
        if not self.min_bpm <= self.preferred_bpm <= self.max_bpm:
            raise ValueError("preferred_bpm must lie between min_bpm and max_bpm")
        if self.manual_bpm is not None and not self.min_bpm <= self.manual_bpm <= self.max_bpm:
            raise ValueError("manual_bpm must lie between min_bpm and max_bpm")
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
    key_window_seconds: float = Field(
        default=8.0,
        gt=0.0,
        description="Rolling musical window in seconds for key estimation (Step 5.2, §145).",
    )
    onset: OnsetConfig = Field(default_factory=OnsetConfig)
    intensity: IntensityConfig = Field(default_factory=IntensityConfig)
    playing: PlayingStateConfig = Field(default_factory=PlayingStateConfig)
    tempo: TempoConfig = Field(default_factory=TempoConfig)
    key: "KeyConfig" = Field(default_factory=lambda: KeyConfig())

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


class KeyConfig(BaseModel):
    """Key and tonal-center estimation over a rolling musical window (Step 5.2, §90, §145)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_seconds: float = Field(
        default=8.0,
        gt=0.0,
        description="Exponential decay time constant for the 12-bin pitch-class histogram.",
    )
    downbeat_weight: float = Field(
        default=1.5,
        ge=1.0,
        description="Weight multiplier for notes landing near beat-1 downbeats or phrase starts.",
    )
    phrase_end_silence_seconds: float = Field(
        default=0.25,
        gt=0.0,
        description="Silence gap that qualifies the preceding note as a phrase-ending tonic.",
    )
    phrase_end_weight: float = Field(
        default=1.3,
        ge=1.0,
        description="Weight multiplier applied to a stable note before a phrase-ending silence.",
    )
    chord_tone_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Relative histogram weight given to non-root chord tones from Step 5.1.",
    )
    min_confidence: float = Field(
        default=0.45,
        gt=0.0,
        le=1.0,
        description="Minimum harmonic confidence required to report or switch a key.",
    )
    switch_margin: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="Score margin a new key must exceed the current key by to switch (§145).",
    )
    report_confidence_step: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Emit a KeyUpdate when harmonic confidence changes by at least this much.",
    )
    manual_key: str | None = Field(
        default=None,
        description="Optional manual key override, e.g. 'E minor' or 'C major' (§58, §145).",
    )

    @model_validator(mode="after")
    def _normalize_manual_key(self) -> Self:
        if self.manual_key is not None:
            from jammate.analysis.key import parse_key_string

            cleaned = self.manual_key.strip()
            if not cleaned or cleaned.lower() == "auto":
                object.__setattr__(self, "manual_key", None)
            else:
                _pc, key_name, mode = parse_key_string(cleaned)
                object.__setattr__(self, "manual_key", f"{key_name} {mode}")
        return self


class MusicalStateConfig(BaseModel):
    """The Musical State (Step 2.5, §36)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recent_note_count: int = Field(
        default=8, ge=1, le=64, description="How many recent notes the state keeps."
    )
    instrument: Instrument = Field(
        default="bass", description="Active instrument for register classification (Step 5.2)."
    )


class RiffConfig(BaseModel):
    """Bar-quantized riff detection and session memory (Step 5.3, §41, §42, §98, §146)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps_per_bar: int = Field(
        default=16, ge=4, le=64, description="Subdivision steps per bar for riff fingerprinting."
    )
    riff_lock_similarity: float = Field(
        default=0.78,
        gt=0.0,
        le=1.0,
        description="Minimum similarity score to lock or continue a riff (§146).",
    )
    riff_change_threshold: float = Field(
        default=0.58,
        gt=0.0,
        le=1.0,
        description="Similarity score below which a playing bar flags a riff change (§146).",
    )
    max_riffs: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum number of riffs stored in bounded SessionMemory (§146).",
    )
    rhythm_weight: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Weight of rhythmic cosine similarity."
    )
    pitch_weight: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Weight of pitch-class contour similarity."
    )
    root_weight: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight of bar root pitch-class agreement."
    )

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if self.riff_change_threshold >= self.riff_lock_similarity:
            raise ValueError("riff_change_threshold must be less than riff_lock_similarity")
        if self.rhythm_weight + self.pitch_weight + self.root_weight <= 0.0:
            raise ValueError("at least one similarity weight must be positive")
        return self


class BeatClockConfig(BaseModel):
    """The internal musical clock and its phase-locked loop (Step 3.1, §31-§33, §149)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meter: MeterName = Field(
        default="4/4",
        description="Time signature / meter ('4/4', '3/4', or '6/8', Step 6.1, §149).",
    )
    beats_per_bar: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Beats per bar (4 for 4/4, 3 for 3/4, 6 for 6/8).",
    )
    phase_correction_gain: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fraction of an onset's timing error applied to the clock anchor (§33).",
    )
    max_phase_correction_seconds: float = Field(
        default=0.008,
        gt=0.0,
        description=(
            "Maximum phase correction applied from a single onset (< 10 ms hop so beat progress "
            "never steps backwards)."
        ),
    )
    tempo_smoothing: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Fraction of the way the clock's BPM moves toward a new locked estimate (§33).",
    )
    alignment_tolerance_seconds: float = Field(
        default=0.03,
        gt=0.0,
        description="Timing tolerance when scoring downbeat candidates from recent onsets.",
    )
    reanchor_error_beats: float = Field(
        default=0.35,
        gt=0.1,
        lt=0.5,
        description="Beat phase error above which an onset counts toward a persistent re-anchor.",
    )
    reanchor_onsets: int = Field(
        default=6,
        ge=2,
        description="Consecutive large-error onsets required before re-anchoring the clock.",
    )
    reanchor_seconds: float = Field(
        default=1.5,
        gt=0.0,
        description="Minimum duration of persistent large phase error before re-anchoring.",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_meter_and_beats_per_bar(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        if "meter" in copied and "beats_per_bar" not in copied:
            meter = copied["meter"]
            if meter in METER_BEATS_PER_BAR:
                copied["beats_per_bar"] = METER_BEATS_PER_BAR[meter]
        elif "beats_per_bar" in copied and "meter" not in copied:
            bpb = copied["beats_per_bar"]
            if bpb == 3:
                copied["meter"] = "3/4"
            elif bpb == 6:
                copied["meter"] = "6/8"
            elif bpb == 4:
                copied["meter"] = "4/4"
        return copied


class DrumConfig(BaseModel):
    """Drum sampler, pattern library, scheduler and decision settings (Steps 3.2-3.4, §43-§49)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kit_dir: Path = Field(
        default=DEFAULT_KIT_DIR, description="Directory containing the kit WAV files."
    )
    max_voices: int = Field(
        default=16, ge=1, le=64, description="Preallocated voice pool size in the drum sampler."
    )
    event_ring_capacity: int = Field(
        default=256,
        ge=16,
        description="Capacity of the scheduled drum event ring between scheduler and callback.",
    )
    style: StyleFamily = Field(
        default="rock", description="Active drum pattern family (Step 3.3, §44)."
    )
    lookahead_seconds: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Look-ahead window for scheduling future drum hits into the ring (§78).",
    )
    scheduler_interval_seconds: float = Field(
        default=0.005,
        gt=0.0,
        le=0.1,
        description="Wake interval of the drum scheduler worker thread.",
    )
    blues_swing: float = Field(
        default=1.0 / 3.0,
        ge=0.0,
        le=0.5,
        description="Swing amount for the blues pattern family (1/3 = triplet shuffle).",
    )
    entry_tempo_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum tempo confidence required once locked before drums enter (§79).",
    )
    low_mid_intensity_threshold: float = Field(
        default=0.35,
        gt=0.0,
        lt=1.0,
        description="Intensity threshold between low and mid pattern tiers (§48).",
    )
    mid_high_intensity_threshold: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
        description="Intensity threshold between mid and high pattern tiers (§48).",
    )
    intensity_hysteresis: float = Field(
        default=0.05,
        ge=0.0,
        lt=0.25,
        description="Hysteresis deadband around each intensity tier boundary (§48).",
    )
    fill_every_bars: int = Field(
        default=8, ge=1, description="Play a 1-bar drum fill every Nth bar while playing (§49)."
    )
    silence_hold_bars: float = Field(
        default=2.0,
        gt=0.0,
        description="Bars of silence before the drummer drops to a lighter holding groove (§71).",
    )
    silence_stop_bars: float = Field(
        default=8.0,
        gt=0.0,
        description="Bars of silence before the drummer stops playing (§71).",
    )
    timing_offset_seconds: float = Field(
        default=0.0,
        ge=-0.1,
        le=0.1,
        description="Signed offset (s) applied to scheduled drum hit frames for feel tuning.",
    )
    adaptive_accents_enabled: bool = Field(
        default=True,
        description="Lock kick and hat patterns to recurring player onset accents (§93, §147).",
    )
    accent_lock_threshold: float = Field(
        default=0.65,
        gt=0.0,
        le=1.0,
        description="Normalized 16-step accent score required to lock a kick hit (§147).",
    )
    accent_window_bars: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of recent bars accumulated by AccentTracker (§147).",
    )
    accent_decay: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description="Exponential bar-to-bar decay factor for AccentTracker.",
    )
    kick_min_step_gap: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Minimum 16th-step distance between adaptive kicks to prevent machine-gunning.",
    )
    humanize_enabled: bool = Field(
        default=True,
        description="Apply deterministic microtiming, velocity shaping, and ghost notes (§147).",
    )
    humanize_timing_ms: float = Field(
        default=4.0,
        ge=0.0,
        le=20.0,
        description="Max microtiming offset in ms (±humanize_timing_ms, 0 on beat 1) (§147).",
    )
    humanize_velocity_variation: float = Field(
        default=0.06,
        ge=0.0,
        le=0.25,
        description="Max deterministic per-hit velocity variation (±humanize_velocity_variation).",
    )

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if (
            self.low_mid_intensity_threshold + self.intensity_hysteresis
            >= self.mid_high_intensity_threshold - self.intensity_hysteresis
        ):
            raise ValueError("intensity tier thresholds overlap after hysteresis")
        if self.silence_hold_bars >= self.silence_stop_bars:
            raise ValueError("silence_hold_bars must be less than silence_stop_bars")
        return self


class AccompanimentConfig(BaseModel):
    """Generative AI accompaniment settings (Steps 4.1-4.3, §50-§52, §74, §82)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable AI accompaniment (uses MRT2 when True, NullAccompaniment when False).",
    )
    model_size: ModelSize = Field(
        default="mrt2_small",
        description=(
            "Magenta RealTime 2 MLX model size: mrt2_small (low quality/fast) "
            "or mrt2_base (high quality)."
        ),
    )
    style: StyleFamily = Field(
        default="rock",
        description="Initial style family for AI accompaniment (§75).",
    )
    follow_improvise: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Product control from 0.0 (Follow closely) to 1.0 (Improvise freely) (§74).",
    )
    ring_capacity_seconds: float = Field(
        default=2.0,
        gt=0.1,
        le=10.0,
        description="Capacity of the 48 kHz stereo AI output ring buffer in seconds.",
    )
    target_lead_seconds: float = Field(
        default=0.32,
        gt=0.04,
        le=2.0,
        description="Target lead buffer (s) maintained by the AI generation worker.",
    )
    chunk_frames: int = Field(
        default=4,
        ge=1,
        le=25,
        description="Number of 40 ms MRT2 frames (1920 samples each) per generate() call.",
    )
    drum_token: int = Field(
        default=0,
        ge=-1,
        le=1,
        description="DRUM_PIANOROLL token passed to MRT2 (0 = no drums, -1 = masked).",
    )
    entry_delay_bars: int = Field(
        default=1,
        ge=0,
        le=8,
        description="Bars after the drummer joins before AI accompaniment enters (§30).",
    )
    silence_gate_bars: float = Field(
        default=4.0,
        gt=0.0,
        description="Bars of player silence before AI accompaniment gates out.",
    )

    @model_validator(mode="after")
    def _check_lead_fits_capacity(self) -> Self:
        if self.target_lead_seconds >= self.ring_capacity_seconds:
            raise ValueError("target_lead_seconds must be less than ring_capacity_seconds")
        return self


class RecordingConfig(BaseModel):
    """Master stereo and multi-track stem WAV recording settings (Steps 4.5 & 6.2, §59, §150)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Start recording master stereo (or stem) output when the session starts.",
    )
    record_stems: bool = Field(
        default=False,
        description=(
            "Write 4 frame-aligned WAV stems (`bass.wav`, `drums.wav`, `ai_guitar.wav`, "
            "`master.wav`) inside a timestamped directory (Step 6.2, §59, §150)."
        ),
    )
    output_dir: Path = Field(
        default=DEFAULT_RECORDINGS_DIR,
        description="Directory where recorded `YYYY-MM-DD_HH-MM-jam.wav` files are written.",
    )
    ring_capacity_seconds: float = Field(
        default=4.0,
        gt=0.2,
        le=30.0,
        description="Capacity of the bounded stereo recording ring buffer in seconds.",
    )


class MidiConfig(BaseModel):
    """Optional MIDI keyboard input settings (Step 6.3, §55, §150)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_device_name: str | None = Field(
        default=None,
        description="MIDI input port name or unique substring (None = disabled).",
    )


class UserPreferences(BaseModel):
    """Persisted user settings saved in `~/Library/Application Support/JamMate/settings.json`."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    input_device_name: str | None = None
    output_device_name: str | None = None
    midi_input_name: str | None = None
    block_size_frames: int = Field(default=128, ge=32, le=4096)
    instrument: Instrument = "bass"
    style: StyleFamily = "rock"
    model_size: ModelSize = "mrt2_small"
    meter: MeterName = "4/4"
    drums_gain: float = Field(default=0.8, ge=0.0, le=4.0)
    ai_gain: float = Field(default=0.7, ge=0.0, le=4.0)

    @classmethod
    def from_config(cls, config: "JamMateConfig") -> "UserPreferences":
        return cls(
            input_device_name=config.audio.input_device_name,
            output_device_name=config.audio.output_device_name,
            midi_input_name=config.midi.input_device_name,
            block_size_frames=config.audio.block_size_frames,
            instrument=config.analysis.instrument,
            style=config.drums.style,
            model_size=config.ai.model_size,
            meter=config.beat_clock.meter,
            drums_gain=config.audio.drums_gain,
            ai_gain=config.audio.ai_gain,
        )

    def apply_to_config(self, config: "JamMateConfig") -> "JamMateConfig":
        """Apply persisted preferences onto `config`."""
        fallback = max(self.block_size_frames, config.audio.fallback_block_size_frames)
        audio = config.audio.model_copy(
            update={
                "input_device_name": self.input_device_name,
                "output_device_name": self.output_device_name,
                "block_size_frames": self.block_size_frames,
                "fallback_block_size_frames": fallback,
                "drums_gain": self.drums_gain,
                "ai_gain": self.ai_gain,
            }
        )
        midi = config.midi.model_copy(update={"input_device_name": self.midi_input_name})
        analysis = config.analysis.model_copy(update={"instrument": self.instrument})
        beat_clock = config.beat_clock.model_copy(
            update={"meter": self.meter, "beats_per_bar": METER_BEATS_PER_BAR[self.meter]}
        )
        drums = config.drums.model_copy(update={"style": self.style})
        ai = config.ai.model_copy(update={"style": self.style, "model_size": self.model_size})
        return config.model_copy(
            update={
                "audio": audio,
                "midi": midi,
                "analysis": analysis,
                "beat_clock": beat_clock,
                "drums": drums,
                "ai": ai,
            }
        )


def load_preferences(path: Path = DEFAULT_PREFERENCES_PATH) -> UserPreferences:
    """Load `UserPreferences` from `path`, returning defaults when missing or invalid (§85)."""
    if not path.is_file():
        return UserPreferences()
    try:
        return UserPreferences.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load preferences from %s (%s); using defaults.", path, exc)
        return UserPreferences()


def save_preferences(prefs: UserPreferences, path: Path = DEFAULT_PREFERENCES_PATH) -> None:
    """Write `prefs` atomically as JSON to `path` (§85)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(prefs.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to save preferences to %s: %s", path, exc)


class JamMateConfig(BaseModel):
    """Root configuration object passed to subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio: AudioConfig = Field(default_factory=AudioConfig)
    midi: MidiConfig = Field(default_factory=MidiConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    state: MusicalStateConfig = Field(default_factory=MusicalStateConfig)
    riff: RiffConfig = Field(default_factory=RiffConfig)
    beat_clock: BeatClockConfig = Field(default_factory=BeatClockConfig)
    drums: DrumConfig = Field(default_factory=DrumConfig)
    ai: AccompanimentConfig = Field(default_factory=AccompanimentConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)

    @model_validator(mode="after")
    def _check_pitch_range_fits_sample_rate(self) -> Self:
        if self.analysis.pitch_range_hz[1] >= self.audio.sample_rate_hz / 4:
            raise ValueError("pitch_max_hz must be below a quarter of the sample rate")
        return self

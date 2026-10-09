"""Semantic musical events: what the analysis layer reports about the bass (§24).

Analysis turns raw audio into these small immutable records. Everything downstream (Musical
State, drums, AI, GUI) works on events and state, never on raw audio.

Every event carries its moment twice:
- `frame_index`: the absolute sample frame on the audio stream's clock, the master timing
  reference (D5). Use it for any scheduling or timing math.
- `timestamp_seconds`: the same moment in seconds since the stream started, for logs and display.

Producers compute both from one frame index with `frames_to_seconds`, so they always agree.
This module imports nothing else from `jammate`: `analysis/notes.py` imports it.
"""

import math
from dataclasses import dataclass

# MIDI note number at sounding pitch, 0-127 (40 = E2 = 82.41 Hz). See `analysis/notes.py`.
type MidiNote = int
# Absolute sample-frame index on the audio stream's clock (D5).
type SampleFrame = int

LOWEST_MIDI_NOTE: MidiNote = 0
HIGHEST_MIDI_NOTE: MidiNote = 127


def frames_to_seconds(frames: int, sample_rate_hz: int) -> float:
    """Convert a frame index or frame count to seconds."""
    _check_sample_rate(sample_rate_hz)
    return frames / sample_rate_hz


def seconds_to_frames(seconds: float, sample_rate_hz: int) -> SampleFrame:
    """Convert seconds to the nearest whole frame (halves round up)."""
    _check_sample_rate(sample_rate_hz)
    return math.floor(seconds * sample_rate_hz + 0.5)


@dataclass(frozen=True, kw_only=True)
class PitchEvent:
    """A stable new note: the stabilizer emits one only when the note musically changes (§26)."""

    frame_index: SampleFrame
    timestamp_seconds: float
    midi_note: MidiNote
    frequency_hz: float
    confidence: float

    def __post_init__(self) -> None:
        _check_moment(self.frame_index, self.timestamp_seconds)
        if not LOWEST_MIDI_NOTE <= self.midi_note <= HIGHEST_MIDI_NOTE:
            raise ValueError(f"midi_note must be 0-127, got {self.midi_note}")
        if not (math.isfinite(self.frequency_hz) and self.frequency_hz > 0):
            raise ValueError(f"frequency_hz must be positive, got {self.frequency_hz}")
        _check_unit_interval("confidence", self.confidence)


@dataclass(frozen=True, kw_only=True)
class OnsetEvent:
    """A new attack on the bass (§27). `strength` is detector-relative: larger = harder attack."""

    frame_index: SampleFrame
    timestamp_seconds: float
    strength: float

    def __post_init__(self) -> None:
        _check_moment(self.frame_index, self.timestamp_seconds)
        if not (math.isfinite(self.strength) and self.strength >= 0):
            raise ValueError(f"strength must be >= 0, got {self.strength}")


@dataclass(frozen=True, kw_only=True)
class TempoUpdate:
    """A new tempo estimate (§28, §29). `bpm` is None while there is no estimate at all."""

    frame_index: SampleFrame
    timestamp_seconds: float
    bpm: float | None
    confidence: float
    locked: bool

    def __post_init__(self) -> None:
        _check_moment(self.frame_index, self.timestamp_seconds)
        if self.bpm is not None and not (math.isfinite(self.bpm) and self.bpm > 0):
            raise ValueError(f"bpm must be positive or None, got {self.bpm}")
        _check_unit_interval("confidence", self.confidence)
        if self.locked and self.bpm is None:
            raise ValueError("a locked tempo needs a bpm")


@dataclass(frozen=True, kw_only=True)
class IntensityUpdate:
    """How hard the player is playing, normalized to 0.0 (very quiet) … 1.0 (aggressive) (§47)."""

    frame_index: SampleFrame
    timestamp_seconds: float
    intensity: float

    def __post_init__(self) -> None:
        _check_moment(self.frame_index, self.timestamp_seconds)
        _check_unit_interval("intensity", self.intensity)


@dataclass(frozen=True, kw_only=True)
class SilenceEvent:
    """The sounding note ended and nothing replaced it: the current note is now None (§70)."""

    frame_index: SampleFrame
    timestamp_seconds: float

    def __post_init__(self) -> None:
        _check_moment(self.frame_index, self.timestamp_seconds)


type MusicalEvent = PitchEvent | OnsetEvent | TempoUpdate | IntensityUpdate | SilenceEvent


def _check_sample_rate(sample_rate_hz: int) -> None:
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")


def _check_moment(frame_index: SampleFrame, timestamp_seconds: float) -> None:
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    if not (math.isfinite(timestamp_seconds) and timestamp_seconds >= 0):
        raise ValueError(f"timestamp_seconds must be >= 0, got {timestamp_seconds}")


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"{name} must be in [0, 1], got {value}")

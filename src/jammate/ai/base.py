"""AccompanimentEngine Protocol, NoteTracker, and failure boundary (Step 4.1, §51, §69, §82)."""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from jammate.audio.ring_buffer import RingBuffer
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    MusicalEvent,
    PitchEvent,
    SilenceEvent,
)

logger = logging.getLogger(__name__)

PIANOROLL_PITCHES = 128
TOKEN_OFF = 0
TOKEN_HELD = 1
TOKEN_ONSET = 2


class AccompanimentStatus(StrEnum):
    """Lifecycle and health state of the AI accompaniment subsystem (§82, §139)."""

    DISABLED = "disabled"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AccompanimentStats:
    """Diagnostic snapshot of the AI accompaniment subsystem (§81, §138)."""

    status: AccompanimentStatus
    style: str
    active_note: MidiNote | None
    intensity: float
    follow_improvise: float
    buffer_seconds: float
    ms_per_frame: float
    frames_generated: int
    last_error: str | None = None


@runtime_checkable
class AccompanimentEngine(Protocol):
    """Model-independent interface for realtime generative accompaniment (§51, §52)."""

    @property
    def output_ring(self) -> RingBuffer | None: ...

    @property
    def status(self) -> AccompanimentStatus: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def set_style(self, style: str) -> None: ...

    def note_on(self, midi_note: MidiNote) -> None: ...

    def note_off(self, midi_note: MidiNote) -> None: ...

    def set_intensity(self, intensity: float) -> None: ...

    def set_follow_improvise(self, follow_improvise: float) -> None: ...

    def stats(self) -> AccompanimentStats: ...


class NoteTracker:
    """Monophonic note-lifecycle tracker turning analysis events into `note_on` / `note_off` (§69).

    Guarantees:
    - At most one note is active at any time.
    - When changing pitch (`old -> new`), `note_off(old)` is always called before `note_on(new)`.
    - Repeating the already-active `midi_note` is a no-op (no duplicate `note_on`).
    - `SilenceEvent` ends the active note with `note_off`.
    - `all_notes_off()` ends any active note on session stop or error cleanup (§84, §137).
    - Also maintains a 128-slot pianoroll helper (`consume_pianoroll`) where a newly triggered note
      reports `2` (`TOKEN_ONSET`) on its first chunk and `1` (`TOKEN_HELD`) while held.
    """

    def __init__(self, engine: AccompanimentEngine | None = None) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._active_note: MidiNote | None = None
        self._active_chord_notes: tuple[MidiNote, ...] = ()
        self._onset_pending: bool = False

    @property
    def active_note(self) -> MidiNote | None:
        return self._active_note

    @property
    def active_chord_notes(self) -> tuple[MidiNote, ...]:
        return self._active_chord_notes

    def set_engine(self, engine: AccompanimentEngine | None) -> None:
        self._engine = engine

    def handle_event(self, event: MusicalEvent) -> None:
        """Process a `MusicalEvent` (`PitchEvent`, `SilenceEvent`, `IntensityUpdate`)."""
        if isinstance(event, PitchEvent):
            self.on_pitch(event.midi_note, chord_notes=event.chord_notes)
        elif isinstance(event, SilenceEvent):
            self.on_silence()
        elif isinstance(event, IntensityUpdate) and self._engine is not None:
            self._engine.set_intensity(event.intensity)

    def on_pitch(
        self,
        midi_note: MidiNote,
        *,
        chord_notes: tuple[MidiNote, ...] = (),
    ) -> None:
        if not 0 <= midi_note < PIANOROLL_PITCHES:
            raise ValueError(f"midi_note must be in 0..127, got {midi_note}")
        valid_chord = tuple(n for n in chord_notes if 0 <= n < PIANOROLL_PITCHES)
        with self._lock:
            previous = self._active_note
            prev_chord = self._active_chord_notes
            if previous == midi_note and prev_chord == valid_chord:
                return
            self._active_note = midi_note
            self._active_chord_notes = valid_chord
            self._onset_pending = True

        engine = self._engine
        if engine is not None and previous != midi_note:
            if previous is not None:
                engine.note_off(previous)
            engine.note_on(midi_note)

    def on_silence(self) -> None:
        with self._lock:
            previous = self._active_note
            if previous is None:
                return
            self._active_note = None
            self._active_chord_notes = ()
            self._onset_pending = False

        if self._engine is not None:
            self._engine.note_off(previous)

    def all_notes_off(self) -> None:
        """Turn off any active note (called on stop/shutdown, §69, §84)."""
        self.on_silence()

    def consume_pianoroll(self) -> np.ndarray:
        """Return a 128-int32 pianoroll (`2`=onset on first read, `1`=held thereafter, `0`=off)."""
        roll = np.zeros(PIANOROLL_PITCHES, dtype=np.int32)
        with self._lock:
            note = self._active_note
            if note is not None:
                token = TOKEN_ONSET if self._onset_pending else TOKEN_HELD
                roll[note] = token
                for chord_note in self._active_chord_notes:
                    roll[chord_note] = token
                self._onset_pending = False
        return roll


class GuardedAccompaniment:
    """Failure boundary around an `AccompanimentEngine` (§82, §137).

    Catches any exception from the underlying adapter, logs it outside the audio callback, marks
    `status = AccompanimentStatus.FAILED`, and stops forwarding calls so bass monitoring and drums
    continue unaffected.
    """

    def __init__(self, inner: AccompanimentEngine) -> None:
        self._inner = inner
        self._failed = False
        self._last_error: str | None = None

    @property
    def inner(self) -> AccompanimentEngine:
        return self._inner

    @property
    def output_ring(self) -> RingBuffer | None:
        if self._failed or self._inner.status == AccompanimentStatus.FAILED:
            return None
        return self._inner.output_ring

    @property
    def status(self) -> AccompanimentStatus:
        if self._failed:
            return AccompanimentStatus.FAILED
        return self._inner.status

    @property
    def last_error(self) -> str | None:
        return self._last_error or self._inner.stats().last_error

    def start(self) -> None:
        if self._failed:
            return
        try:
            self._inner.start()
        except Exception as exc:
            self._trip("start", exc)

    def stop(self) -> None:
        try:
            self._inner.stop()
        except Exception as exc:
            self._trip("stop", exc)

    def set_style(self, style: str) -> None:
        if self._failed:
            return
        try:
            self._inner.set_style(style)
        except Exception as exc:
            self._trip("set_style", exc)

    def note_on(self, midi_note: MidiNote) -> None:
        if self._failed:
            return
        try:
            self._inner.note_on(midi_note)
        except Exception as exc:
            self._trip("note_on", exc)

    def note_off(self, midi_note: MidiNote) -> None:
        if self._failed:
            return
        try:
            self._inner.note_off(midi_note)
        except Exception as exc:
            self._trip("note_off", exc)

    def set_intensity(self, intensity: float) -> None:
        if self._failed:
            return
        try:
            self._inner.set_intensity(intensity)
        except Exception as exc:
            self._trip("set_intensity", exc)

    def set_follow_improvise(self, follow_improvise: float) -> None:
        if self._failed:
            return
        try:
            self._inner.set_follow_improvise(follow_improvise)
        except Exception as exc:
            self._trip("set_follow_improvise", exc)

    def stats(self) -> AccompanimentStats:
        try:
            inner_stats = self._inner.stats()
        except Exception as exc:
            self._trip("stats", exc)
            return AccompanimentStats(
                status=AccompanimentStatus.FAILED,
                style="rock",
                active_note=None,
                intensity=0.0,
                follow_improvise=0.5,
                buffer_seconds=0.0,
                ms_per_frame=0.0,
                frames_generated=0,
                last_error=self._last_error,
            )
        if self._failed:
            return AccompanimentStats(
                status=AccompanimentStatus.FAILED,
                style=inner_stats.style,
                active_note=inner_stats.active_note,
                intensity=inner_stats.intensity,
                follow_improvise=inner_stats.follow_improvise,
                buffer_seconds=0.0,
                ms_per_frame=inner_stats.ms_per_frame,
                frames_generated=inner_stats.frames_generated,
                last_error=self._last_error or inner_stats.last_error,
            )
        return inner_stats

    def _trip(self, operation: str, exc: Exception) -> None:
        self._failed = True
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.error("AI accompaniment failed during %s: %s", operation, self._last_error)
        with contextlib.suppress(Exception):
            self._inner.stop()

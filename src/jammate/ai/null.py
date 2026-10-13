"""NullAccompaniment: no-op adapter when AI accompaniment is disabled or unavailable (Step 4.1)."""

from __future__ import annotations

from jammate.ai.base import AccompanimentStats, AccompanimentStatus
from jammate.audio.ring_buffer import RingBuffer
from jammate.music.events import MidiNote


class NullAccompaniment:
    """No-op `AccompanimentEngine` used when AI is disabled or the model is unavailable (§82)."""

    def __init__(
        self,
        *,
        style: str = "rock",
        follow_improvise: float = 0.5,
        unavailable_reason: str | None = None,
    ) -> None:
        self._style = style
        self._follow_improvise = float(follow_improvise)
        self._intensity = 0.0
        self._active_note: MidiNote | None = None
        self._unavailable_reason = unavailable_reason
        self._status = (
            AccompanimentStatus.UNAVAILABLE
            if unavailable_reason is not None
            else AccompanimentStatus.DISABLED
        )

    @property
    def output_ring(self) -> RingBuffer | None:
        return None

    @property
    def status(self) -> AccompanimentStatus:
        return self._status

    @property
    def active_note(self) -> MidiNote | None:
        return self._active_note

    @property
    def style(self) -> str:
        return self._style

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def follow_improvise(self) -> float:
        return self._follow_improvise

    def start(self) -> None:
        if self._unavailable_reason is None:
            self._status = AccompanimentStatus.DISABLED

    def stop(self) -> None:
        self._active_note = None

    def set_style(self, style: str) -> None:
        self._style = style

    def note_on(self, midi_note: MidiNote) -> None:
        self._active_note = midi_note

    def note_off(self, midi_note: MidiNote) -> None:
        if self._active_note == midi_note:
            self._active_note = None

    def set_intensity(self, intensity: float) -> None:
        self._intensity = max(0.0, min(1.0, float(intensity)))

    def set_follow_improvise(self, follow_improvise: float) -> None:
        if not 0.0 <= follow_improvise <= 1.0:
            raise ValueError("follow_improvise must be between 0.0 and 1.0")
        self._follow_improvise = float(follow_improvise)

    def stats(self) -> AccompanimentStats:
        return AccompanimentStats(
            status=self._status,
            style=self._style,
            active_note=self._active_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
            buffer_seconds=0.0,
            ms_per_frame=0.0,
            frames_generated=0,
            last_error=self._unavailable_reason,
        )

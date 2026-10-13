"""Tests for AccompanimentEngine, NoteTracker, NullAccompaniment, and GuardedAccompaniment."""

from __future__ import annotations

import numpy as np
import pytest

from jammate.ai.base import (
    PIANOROLL_PITCHES,
    TOKEN_HELD,
    TOKEN_OFF,
    TOKEN_ONSET,
    AccompanimentEngine,
    AccompanimentStats,
    AccompanimentStatus,
    NoteTracker,
)
from jammate.ai.null import NullAccompaniment
from jammate.audio.engine import BlockProcessor
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import JamMateConfig
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    PitchEvent,
    SilenceEvent,
)
from jammate.session import JamSession


class RecordingEngine:
    """Test double recording all `AccompanimentEngine` calls and optionally failing."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fail_on = fail_on
        self._ring = RingBuffer(capacity_frames=4_096, channels=2)
        self._status = AccompanimentStatus.READY
        self._style = "rock"
        self._active_note: MidiNote | None = None
        self._intensity = 0.0
        self._follow_improvise = 0.5

    @property
    def output_ring(self) -> RingBuffer | None:
        return self._ring

    @property
    def status(self) -> AccompanimentStatus:
        return self._status

    def _check_fail(self, method: str) -> None:
        if self.fail_on == method:
            raise RuntimeError(f"boom in {method}")

    def start(self) -> None:
        self.calls.append(("start", None))
        self._check_fail("start")
        self._status = AccompanimentStatus.RUNNING

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self._check_fail("stop")
        self._status = AccompanimentStatus.STOPPED

    def set_style(self, style: str) -> None:
        self.calls.append(("set_style", style))
        self._check_fail("set_style")
        self._style = style

    def note_on(self, midi_note: MidiNote) -> None:
        self.calls.append(("note_on", midi_note))
        self._check_fail("note_on")
        self._active_note = midi_note

    def note_off(self, midi_note: MidiNote) -> None:
        self.calls.append(("note_off", midi_note))
        self._check_fail("note_off")
        if self._active_note == midi_note:
            self._active_note = None

    def set_intensity(self, intensity: float) -> None:
        self.calls.append(("set_intensity", intensity))
        self._check_fail("set_intensity")
        self._intensity = intensity

    def set_follow_improvise(self, follow_improvise: float) -> None:
        self.calls.append(("set_follow_improvise", follow_improvise))
        self._check_fail("set_follow_improvise")
        self._follow_improvise = follow_improvise

    def stats(self) -> AccompanimentStats:
        self._check_fail("stats")
        return AccompanimentStats(
            status=self._status,
            style=self._style,
            active_note=self._active_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
            buffer_seconds=self._ring.available_frames() / 48_000.0,
            ms_per_frame=12.5,
            frames_generated=25,
        )


def test_null_accompaniment_satisfies_protocol_and_tracks_state() -> None:
    null = NullAccompaniment(style="stoner_rock", follow_improvise=0.3)
    assert isinstance(null, AccompanimentEngine)
    assert null.status == AccompanimentStatus.DISABLED
    assert null.output_ring is None

    null.start()
    null.note_on(40)
    assert null.active_note == 40
    null.set_intensity(0.75)
    null.set_style("funk")
    null.set_follow_improvise(0.8)
    stats = null.stats()
    assert stats.style == "funk"
    assert stats.active_note == 40
    assert stats.intensity == pytest.approx(0.75)
    assert stats.follow_improvise == pytest.approx(0.8)

    null.note_off(40)
    assert null.active_note is None

    unavailable = NullAccompaniment(unavailable_reason="weights missing")
    assert unavailable.status == AccompanimentStatus.UNAVAILABLE
    assert unavailable.stats().last_error == "weights missing"


def test_note_tracker_enforces_mono_note_off_before_note_on_and_pianoroll_tokens() -> None:
    engine = RecordingEngine()
    tracker = NoteTracker(engine)

    # Initial state: no active note, pianoroll is all zeros.
    assert tracker.active_note is None
    assert np.all(tracker.consume_pianoroll() == TOKEN_OFF)

    # First pitch E2 (MIDI 40) -> note_on(40).
    tracker.handle_event(
        PitchEvent(
            frame_index=1_000,
            timestamp_seconds=0.02,
            frequency_hz=82.41,
            midi_note=40,
            confidence=0.9,
        )
    )
    assert tracker.active_note == 40
    assert engine.calls == [("note_on", 40)]

    # First pianoroll read after note_on reports TOKEN_ONSET (2); second reports TOKEN_HELD (1).
    roll_onset = tracker.consume_pianoroll()
    assert roll_onset.shape == (PIANOROLL_PITCHES,)
    assert roll_onset[40] == TOKEN_ONSET
    assert np.sum(roll_onset != TOKEN_OFF) == 1

    roll_held = tracker.consume_pianoroll()
    assert roll_held[40] == TOKEN_HELD

    # Repeating the same note is a no-op (no duplicate note_on or spurious onset).
    tracker.handle_event(
        PitchEvent(
            frame_index=2_000,
            timestamp_seconds=0.04,
            frequency_hz=82.41,
            midi_note=40,
            confidence=0.9,
        )
    )
    assert engine.calls == [("note_on", 40)]

    # Pitch change to G2 (MIDI 43): note_off(40) MUST precede note_on(43).
    tracker.handle_event(
        PitchEvent(
            frame_index=3_000,
            timestamp_seconds=0.06,
            frequency_hz=98.0,
            midi_note=43,
            confidence=0.9,
        )
    )
    assert tracker.active_note == 43
    assert engine.calls == [("note_on", 40), ("note_off", 40), ("note_on", 43)]
    roll_g2 = tracker.consume_pianoroll()
    assert roll_g2[40] == TOKEN_OFF
    assert roll_g2[43] == TOKEN_ONSET

    # SilenceEvent turns off G2.
    tracker.handle_event(SilenceEvent(frame_index=4_000, timestamp_seconds=0.08))
    assert tracker.active_note is None
    assert engine.calls == [("note_on", 40), ("note_off", 40), ("note_on", 43), ("note_off", 43)]
    assert np.all(tracker.consume_pianoroll() == TOKEN_OFF)

    # Play A2 (MIDI 45), then all_notes_off() on shutdown ensures no hanging note.
    tracker.on_pitch(45)
    assert tracker.active_note == 45
    tracker.all_notes_off()
    assert tracker.active_note is None
    assert engine.calls[-2:] == [("note_on", 45), ("note_off", 45)]


def test_ai_bus_underrun_outputs_silence_for_missing_frames_and_counts_underrun() -> None:
    input_ring = RingBuffer(capacity_frames=1_024, channels=1)
    ai_ring = RingBuffer(capacity_frames=1_024, channels=2)
    processor = BlockProcessor(
        input_ring,
        input_channel_index=0,
        monitor_gain=0.0,
        drums_gain=0.0,
        ai_gain=0.5,
        ai_ring=ai_ring,
    )

    indata = np.zeros((128, 1), dtype=np.float32)
    outdata = np.ones((128, 2), dtype=np.float32)

    # Empty AI ring -> full underrun, output is exact silence, ai_underruns increments to 1.
    processor.process(indata, outdata)
    assert processor.ai_underruns == 1
    assert np.all(outdata == 0.0)

    # Partial AI ring (64 frames of 0.4) -> first 64 frames mixed at 0.4 * 0.5 = 0.2,
    # remaining 64 frames zero-filled, ai_underruns increments to 2.
    partial = np.full((64, 2), 0.4, dtype=np.float32)
    ai_ring.write(partial)
    processor.process(indata, outdata)
    assert processor.ai_underruns == 2
    assert np.allclose(outdata[:64], 0.2, atol=1e-6)
    assert np.all(outdata[64:] == 0.0)


def test_guarded_accompaniment_and_session_survive_adapter_failure(fake_stream) -> None:
    del fake_stream
    failing = RecordingEngine(fail_on="note_on")
    session = JamSession(
        JamMateConfig(),
        accompaniment_factory=lambda _cfg, _sr: failing,
    )
    session.start()
    assert session.ai.status == AccompanimentStatus.RUNNING
    assert session.engine.processor.ai_ring is failing.output_ring

    # Feeding a PitchEvent triggers failing.note_on -> caught by GuardedAccompaniment!
    session._handle_event(
        PitchEvent(
            frame_index=512,
            timestamp_seconds=0.01,
            frequency_hz=82.41,
            midi_note=40,
            confidence=0.9,
        )
    )
    assert session.ai.status == AccompanimentStatus.FAILED
    assert session.ai.stats().status == AccompanimentStatus.FAILED
    assert "boom in note_on" in (session.ai.last_error or "")
    # AI ring is detached so the callback stops reading from the failed engine.
    assert session.engine.processor.ai_ring is None

    # Bass monitoring and drums remain running!
    assert session.is_running
    assert session.drums.is_running
    indata = np.full((128, 1), 0.25, dtype=np.float32)
    outdata = np.zeros((128, 2), dtype=np.float32)
    session.engine._callback(indata, outdata, 128, None, None)
    assert np.allclose(outdata, 0.25, atol=1e-6)
    assert session.engine.processor.errors == 0

    # Further events (intensity, silence) do not raise.
    session._handle_event(IntensityUpdate(frame_index=1_024, timestamp_seconds=0.02, intensity=0.8))
    session.stop()
    assert not session.is_running


def test_session_stop_sends_all_notes_off_before_stopping_ai(fake_stream) -> None:
    del fake_stream
    rec = RecordingEngine()
    session = JamSession(
        JamMateConfig(),
        accompaniment_factory=lambda _cfg, _sr: rec,
    )
    session.start()
    session._handle_event(
        PitchEvent(
            frame_index=512,
            timestamp_seconds=0.01,
            frequency_hz=110.0,
            midi_note=45,
            confidence=0.95,
        )
    )
    assert session.note_tracker.active_note == 45

    session.stop()
    assert session.note_tracker.active_note is None
    assert rec.calls == [
        ("start", None),
        ("note_on", 45),
        ("note_off", 45),
        ("stop", None),
    ]

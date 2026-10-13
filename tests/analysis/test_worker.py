import threading

import numpy as np
import pytest

from jammate.analysis.notes import midi_to_name
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisWorker, analyze_offline
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AnalysisConfig
from jammate.music.events import (
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SilenceEvent,
)
from jammate.music.state import replay
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, noise, plucked_note, riff, silence

SR = 48_000
GUITAR = AnalysisConfig(instrument="guitar")
BASS = AnalysisConfig(instrument="bass")
HOP = GUITAR.hop_frames


def _write(ring: RingBuffer, samples: np.ndarray, block: int = 128) -> None:
    for start in range(0, len(samples), block):
        ring.write(samples[start : start + block].reshape(-1, 1).astype(np.float32))


def _notes(events: list) -> list[str]:
    """Note names and "--" for silences, ignoring every other kind of event."""
    return [
        midi_to_name(e.event.midi_note) if isinstance(e.event, PitchEvent) else "--"
        for e in events
        if isinstance(e.event, PitchEvent | SilenceEvent)
    ]


# --- process_available ---------------------------------------------------------------------------


def test_processes_whole_hops_and_keeps_the_remainder() -> None:
    ring = RingBuffer(48_000)
    frames: list[PitchFrame] = []
    worker = AnalysisWorker(ring, GUITAR, SR, on_pitch_frame=frames.append)

    _write(ring, np.zeros(1000))
    assert worker.process_available() == 2  # 960 frames; 40 wait for the next hop
    _write(ring, np.zeros(500))
    assert worker.process_available() == 1

    assert [frame.frame_index for frame in frames] == [480, 960, 1440]
    assert worker.stats().hops_processed == 3
    assert worker.stats().dropped_frames == 0


def test_emits_events_on_the_processing_thread_with_stream_positions() -> None:
    ring = RingBuffer(48_000)
    events: list[MusicalEvent] = []
    worker = AnalysisWorker(ring, GUITAR, SR, on_event=events.append)
    tone = plucked_note("A2", 1.5, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    samples = np.concatenate([silence(0.2), tone, silence(0.5)])  # 2.2 s

    for start in range(0, len(samples), 4800):
        _write(ring, samples[start : start + 4800])
        worker.process_available()

    notes = [event for event in events if isinstance(event, PitchEvent | SilenceEvent)]
    assert [type(event) for event in notes] == [PitchEvent, SilenceEvent]
    assert notes[0].midi_note == 45
    assert 0.2 < notes[0].timestamp_seconds < 0.3
    (onset,) = [event for event in events if isinstance(event, OnsetEvent)]
    assert abs(onset.frame_index - 9600) < 100  # the pluck at 0.2 s
    states = [event.state for event in events if isinstance(event, PlayingStateUpdate)]
    assert states == [PlayingState.PLAYING, PlayingState.SUSTAINING, PlayingState.STOPPED]
    assert worker.stats().onsets == 1
    state = replay(events)  # what the Musical State makes of them
    assert state.current_note is None
    assert state.recent_notes == (45,)
    assert state.playing_state == PlayingState.STOPPED


def test_skips_to_the_latest_audio_when_far_behind() -> None:
    ring = RingBuffer(48_000)
    worker = AnalysisWorker(ring, GUITAR, SR)  # max backlog 0.1 s = 4800 frames

    _write(ring, np.zeros(10_000))
    assert worker.process_available() == 0  # skipped everything unread
    _write(ring, np.zeros(HOP))
    assert worker.process_available() == 1

    stats = worker.stats()
    assert stats.skips == 1
    assert stats.dropped_frames == 10_000
    assert stats.latest is not None and stats.latest.frame_index == 10_000 + HOP


def test_restarts_the_hop_after_a_ring_overrun() -> None:
    ring = RingBuffer(1024)
    frames: list[PitchFrame] = []
    config = GUITAR.model_copy(update={"max_backlog_seconds": 1.0})
    worker = AnalysisWorker(ring, config, SR, on_pitch_frame=frames.append)

    _write(ring, np.zeros(300))
    worker.process_available()  # 300 of 480 frames buffered
    _write(ring, np.zeros(2000))  # overruns the 1024-frame ring
    worker.process_available()

    # After the gap the hop restarts at the oldest valid frame (2300 - 1024 = 1276).
    assert [frame.frame_index for frame in frames] == [1276 + HOP, 1276 + 2 * HOP]
    assert worker.stats().dropped_frames == 1276 - 300


def test_tracks_peaks_and_clipping() -> None:
    ring = RingBuffer(48_000)
    worker = AnalysisWorker(ring, GUITAR, SR)

    _write(ring, np.full(HOP * 3, 0.5))
    _write(ring, np.full(HOP, 1.0))
    worker.process_available()

    stats = worker.stats()
    assert stats.max_peak_dbfs == 0.0
    assert stats.recent_peak_dbfs == 0.0
    assert stats.clipped_hops == 1
    assert stats.hop_mean_seconds > 0
    assert stats.hop_max_seconds >= stats.hop_mean_seconds


# --- thread lifecycle ----------------------------------------------------------------------------


def test_thread_analyzes_until_stopped_and_joins() -> None:
    ring = RingBuffer(48_000)
    analyzed = threading.Event()
    worker = AnalysisWorker(ring, GUITAR, SR, on_pitch_frame=lambda frame: analyzed.set())

    worker.start()
    assert worker.is_running
    _write(ring, np.zeros(HOP))
    assert analyzed.wait(timeout=5.0)
    worker.stop()

    assert not worker.is_running
    with pytest.raises(RuntimeError):
        worker.start()


def test_thread_survives_errors_in_the_event_consumer() -> None:
    ring = RingBuffer(48_000)
    failed = threading.Event()

    def broken_consumer(frame: PitchFrame) -> None:
        failed.set()
        raise RuntimeError("consumer bug")

    worker = AnalysisWorker(ring, GUITAR, SR, on_pitch_frame=broken_consumer)
    worker.start()
    _write(ring, np.zeros(HOP))
    assert failed.wait(timeout=5.0)
    worker.stop()

    assert worker.stats().errors >= 1
    assert isinstance(worker.last_error, RuntimeError)


def test_stop_without_start_is_harmless() -> None:
    AnalysisWorker(RingBuffer(4800), GUITAR, SR).stop()


# --- end to end on synthetic signals -------------------------------------------------------------


@pytest.mark.parametrize("note", ["E1", "A1", "D2", "G2", "E2", "A2", "D3", "G3"])
def test_synthetic_bass_notes(note: str) -> None:
    samples = np.concatenate([silence(0.2), plucked_note(note, 1.5), silence(0.5)])

    events, _ = analyze_offline(samples, BASS, SR)

    assert _notes(events) == [note, "--"]


@pytest.mark.parametrize(
    "note", ["E2", "A2", "D3", "G3", "B3", "E4", "E3", "A3", "D4", "G4", "B4", "E5"]
)
def test_synthetic_guitar_notes(note: str) -> None:
    tone = plucked_note(note, 1.5, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    samples = np.concatenate([silence(0.2), tone, silence(0.5)])

    events, _ = analyze_offline(samples, GUITAR, SR)

    assert _notes(events) == [note, "--"]
    (pitch,) = [e for e in events if isinstance(e.event, PitchEvent)]
    assert 0.2 <= pitch.event.timestamp_seconds < 0.3
    assert pitch.emitted_frame - pitch.event.frame_index < 0.1 * SR


def test_synthetic_riff_reports_each_note_change_once() -> None:
    rendered = riff(
        ["E2", "G2", "A2", "E2"],
        bpm=120,
        step_beats=0.5,
        repeats=2,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    events, _ = analyze_offline(rendered.samples, GUITAR, SR)

    # The repeated E2 across the loop boundary is one note, not two.
    assert _notes(events) == ["E2", "G2", "A2", "E2", "G2", "A2", "E2"]
    onsets_seconds = [frame / SR for frame in rendered.onset_frames]
    pitch_times = [e.event.timestamp_seconds for e in events if isinstance(e.event, PitchEvent)]
    expected_onsets = [onsets_seconds[i] for i in (0, 1, 2, 3, 5, 6, 7)]
    for detected, onset in zip(pitch_times, expected_onsets, strict=True):
        assert 0.0 <= detected - onset < 0.06


def test_silence_and_quiet_noise_produce_no_events() -> None:
    for samples in (silence(2.0), noise(2.0)):
        for config in (BASS, GUITAR):
            events, _ = analyze_offline(samples, config, SR)
            assert events == []


def test_loud_steady_noise_is_noise_not_playing() -> None:
    # Like a hum or a broken cable: above the level gate, but no pitch and no attacks.
    for config in (BASS, GUITAR):
        events, _ = analyze_offline(noise(6.0, rms_dbfs=-30.0), config, SR)
        state = replay(e.event for e in events)

        assert not [e for e in events if isinstance(e.event, PitchEvent)]
        # Only the noise switching on (out of digital silence) is an attack.
        onsets = [e.event for e in events if isinstance(e.event, OnsetEvent)]
        assert [onset.frame_index for onset in onsets] in ([], [0])
        assert state.playing_state == PlayingState.NOISE
        assert state.intensity < 0.1  # falling since the state became NOISE

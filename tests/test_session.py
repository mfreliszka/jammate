import threading
from collections.abc import Callable

import numpy as np
import pytest

from jammate.analysis.worker import AnalysisWorker
from jammate.audio.devices import DeviceNotFoundError
from jammate.audio.engine import AudioEngine, AudioEngineError, StreamInfo
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AnalysisConfig, AudioConfig, JamMateConfig
from jammate.music.events import MusicalEvent, PitchEvent, PlayingState, SilenceEvent
from jammate.music.state import MusicalState
from jammate.session import JamSession
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, FakeStream
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, plucked_note, silence

BLOCK = 128
CONFIG = JamMateConfig(analysis=AnalysisConfig(instrument="guitar"))


class RecordingEngine(AudioEngine):
    def __init__(self, audio: AudioConfig, calls: list[str]) -> None:
        super().__init__(audio, INPUT_DEVICE, OUTPUT_DEVICE)
        self.calls = calls

    def start(self) -> StreamInfo:
        self.calls.append("engine.start")
        return super().start()

    def close(self) -> None:
        self.calls.append("engine.close")
        super().close()


class ManualWorker(AnalysisWorker):
    """An AnalysisWorker without its thread: the test drives `process_available`."""

    calls: list[str]

    def start(self) -> None:
        self.calls.append("worker.start")

    def stop(self) -> None:
        self.calls.append("worker.stop")


class Parts:
    """Factories that build a fake-stream engine and a thread-less worker, recording the
    lifecycle calls in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.worker: ManualWorker | None = None

    def engine(self, audio: AudioConfig) -> AudioEngine:
        return RecordingEngine(audio, self.calls)

    def manual_worker(
        self,
        ring: RingBuffer,
        config: AnalysisConfig,
        sample_rate_hz: int,
        on_event: Callable[[MusicalEvent], None],
    ) -> AnalysisWorker:
        self.worker = ManualWorker(ring, config, sample_rate_hz, on_event=on_event)
        self.worker.calls = self.calls
        return self.worker

    def session(self, **kwargs: object) -> JamSession:
        return JamSession(
            CONFIG,
            engine_factory=self.engine,
            worker_factory=self.manual_worker,
            **kwargs,  # type: ignore[arg-type]
        )


def _play(session: JamSession, samples: np.ndarray, worker: ManualWorker | None = None) -> None:
    """Act as the audio thread: deliver `samples` through the engine's callback. With a manual
    `worker`, run its pass after each block, as if its thread kept up."""
    outdata = np.zeros((BLOCK, 2), dtype=np.float32)
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        block = samples[start : start + BLOCK].astype(np.float32).reshape(-1, 1)
        session.engine._callback(block, outdata, BLOCK, None, None)
        if worker is not None:
            worker.process_available()


def _a2_note() -> np.ndarray:
    tone = plucked_note("A2", 1.0, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    return np.concatenate([silence(0.2), tone, silence(0.5)])


@pytest.mark.usefixtures("fake_stream")
def test_startup_runs_analysis_before_the_stream_and_shutdown_reverses_it() -> None:
    parts = Parts()
    session = parts.session()
    assert parts.calls == []  # built, but nothing runs yet (§83)

    info = session.start()
    assert session.is_running
    assert info.block_size_frames == BLOCK
    session.stop()

    assert parts.calls == ["worker.start", "engine.start", "worker.stop", "engine.close"]  # §84
    assert not session.is_running
    assert FakeStream.instances[0].closed


@pytest.mark.usefixtures("fake_stream")
def test_analysis_events_update_the_state_then_reach_the_listener() -> None:
    parts = Parts()
    heard: list[tuple[MusicalEvent, MusicalState]] = []
    session: JamSession = parts.session(on_event=lambda event: heard.append((event, session.state)))
    session.start()

    _play(session, _a2_note(), parts.worker)

    pitch = next(state for event, state in heard if isinstance(event, PitchEvent))
    assert pitch.current_note == 45  # the state already held the note when the listener ran
    assert any(isinstance(event, SilenceEvent) for event, _ in heard)
    state = session.state
    assert state.current_note is None
    assert state.recent_notes == (45,)
    assert state.playing_state == PlayingState.STOPPED
    session.stop()


def test_a_stream_that_fails_to_start_leaves_nothing_running(
    fake_stream: type[FakeStream],
) -> None:
    fake_stream.reject_block_sizes = frozenset({128, 256})
    parts = Parts()
    session = parts.session()

    with pytest.raises(AudioEngineError):
        session.start()

    assert parts.calls == ["worker.start", "engine.start", "worker.stop", "engine.close"]
    assert not session.is_running


@pytest.mark.usefixtures("fake_stream")
def test_stop_is_safe_twice_and_without_start() -> None:
    parts = Parts()
    session = parts.session()

    session.stop()
    session.stop()

    assert parts.calls == ["worker.stop", "engine.close"]


@pytest.mark.usefixtures("fake_stream")
def test_a_session_runs_once() -> None:
    session = Parts().session()
    session.start()
    session.stop()

    with pytest.raises(RuntimeError, match="runs once"):
        session.start()


@pytest.mark.usefixtures("fake_stream")
def test_leaving_a_with_block_shuts_down() -> None:
    parts = Parts()

    with parts.session() as session:
        session.start()

    assert parts.calls[-2:] == ["worker.stop", "engine.close"]


def test_unusable_devices_fail_before_anything_is_built() -> None:
    def no_device(audio: AudioConfig) -> AudioEngine:
        raise DeviceNotFoundError("No input device matches 'Behringer'.")

    with pytest.raises(DeviceNotFoundError):
        JamSession(CONFIG, engine_factory=no_device)


@pytest.mark.usefixtures("fake_stream")
def test_the_analysis_thread_is_the_only_state_writer() -> None:
    # The real worker thread; the test plays the audio thread and waits for the note. It writes
    # faster than real time, so the worker may not skip ahead (`max_backlog_seconds`).
    noted = threading.Event()
    patient = AnalysisConfig(instrument="guitar", max_backlog_seconds=1.5)
    session = JamSession(
        JamMateConfig(analysis=patient),
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        on_event=lambda event: noted.set() if isinstance(event, PitchEvent) else None,
    )
    session.start()
    try:
        _play(session, _a2_note()[:48_000])  # 1 s
        assert noted.wait(timeout=5.0)
        assert session.state.current_note == 45
        with pytest.raises(RuntimeError, match="single writer"):
            session._state.apply(SilenceEvent(frame_index=96_000, timestamp_seconds=2.0))
    finally:
        session.stop()
    assert not session.worker.is_running

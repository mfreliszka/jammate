import io
import logging
import signal
from collections.abc import Callable

import numpy as np
import pytest

from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisStats, AnalysisWorker
from jammate.audio.devices import DeviceNotFoundError
from jammate.audio.engine import AudioEngine, CallbackTiming, EngineStats
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AnalysisConfig, AudioConfig, JamMateConfig
from jammate.headless import (
    StatusSnapshot,
    format_elapsed,
    format_event,
    format_status,
    format_stream_time,
    run_headless,
)
from jammate.music.events import (
    IntensityUpdate,
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SilenceEvent,
)
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, FakeClock, FakeStream
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, plucked_note

BLOCK = 128
GUITAR_CONFIG = JamMateConfig(analysis=AnalysisConfig(instrument="guitar"))


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def _engine_stats(**overrides: object) -> EngineStats:
    values: dict[str, object] = {
        "frames_processed": 0,
        "blocks_processed": 0,
        "input_overflows": 0,
        "output_underflows": 0,
        "startup_xruns": 1,
        "callback_errors": 0,
        "callback_timing": CallbackTiming(100, 10e-6, 40e-6),
    }
    values.update(overrides)
    return EngineStats(**values)  # type: ignore[arg-type]


def _analysis_stats(**overrides: object) -> AnalysisStats:
    values: dict[str, object] = {
        "hops_processed": 100,
        "dropped_frames": 0,
        "skips": 0,
        "errors": 0,
        "hop_mean_seconds": 100e-6,
        "hop_max_seconds": 400e-6,
        "recent_peak_dbfs": -6.5,
        "max_peak_dbfs": -3.0,
        "clipped_hops": 0,
        "latest": PitchFrame(4800, 82.4, 0.99, -24.0, -7.0, -20.0),
        "current_note": 40,
        "onsets": 12,
        "intensity": 0.62,
        "playing_state": PlayingState.PLAYING,
    }
    values.update(overrides)
    return AnalysisStats(**values)  # type: ignore[arg-type]


def _snapshot(clipped: bool = False, **analysis_overrides: object) -> StatusSnapshot:
    return StatusSnapshot(
        elapsed_seconds=3725.0,
        ring_fill_ratio=0.25,
        block_period_seconds=BLOCK / 48_000,
        clipped=clipped,
        stats=_engine_stats(input_overflows=2),
        analysis=_analysis_stats(**analysis_overrides),
    )


# --- formatting ----------------------------------------------------------------------------------


def test_format_elapsed_and_stream_time() -> None:
    assert format_elapsed(3725.9) == "01:02:05"
    assert format_stream_time(75.5) == "01:15.500"


def test_format_status_shows_note_level_and_diagnostics() -> None:
    line = format_status(_snapshot())

    assert line.startswith("01:02:05  note E2   playing    int 0.62  ")
    assert "level rms  -24.0 peak   -6.5 dBFS" in line
    assert "xruns in 2 out 0 (start-up 1)" in line
    assert "ring  25%" in line
    assert "callback  10/  40 us (2%)" in line
    assert "analysis 100/ 400 us" in line
    for flag in ("CLIP", "ERRORS", "DROPPED"):
        assert flag not in line


def test_format_status_flags_problems() -> None:
    line = format_status(
        _snapshot(
            clipped=True,
            current_note=None,
            errors=2,
            dropped_frames=4800,
            playing_state=PlayingState.NOISE,
            intensity=0.0,
        )
    )

    assert "note --   noise      int 0.00" in line
    assert "dBFS CLIP" in line
    assert "DROPPED 4800" in line
    assert "ANALYSIS ERRORS 2" in line


def test_format_event_lines() -> None:
    pitch = PitchEvent(
        frame_index=48_000, timestamp_seconds=1.0, midi_note=40, frequency_hz=82.6, confidence=0.97
    )
    silence = SilenceEvent(frame_index=96_000, timestamp_seconds=2.0)

    assert format_event(pitch) == "  00:01.000  E2     82.60 Hz   +4 c  conf 0.97"
    assert format_event(silence) == "  00:02.000  --   silence"
    playing = PlayingStateUpdate(
        frame_index=144_000, timestamp_seconds=3.0, state=PlayingState.TOO_QUIET
    )
    assert format_event(playing) == "  00:03.000  [too_quiet]"
    assert format_event(OnsetEvent(frame_index=0, timestamp_seconds=0.0, strength=1.0)) is None
    assert (
        format_event(IntensityUpdate(frame_index=0, timestamp_seconds=0.0, intensity=0.5)) is None
    )


# --- run_headless (real engine and worker, fake stream, no threads) -------------------------------


class ManualWorker(AnalysisWorker):
    """An AnalysisWorker without its thread: the test drives `process_available`."""

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class Session:
    """Drives `run_headless`: each fake sleep advances the clock, feeds that much audio and runs
    the analysis; Ctrl+C is simulated after `polls_before_ctrl_c` sleeps."""

    def __init__(self, polls_before_ctrl_c: int, audio: np.ndarray | None = None) -> None:
        self.engine: AudioEngine | None = None
        self.worker: ManualWorker | None = None
        self.polls = 0
        self.polls_before_ctrl_c = polls_before_ctrl_c
        self.audio = (
            0.5 * np.sin(np.arange(48_000) * 2 * np.pi * 220 / 48_000) if audio is None else audio
        )
        self.position = 0
        self.clock = FakeClock()

    def engine_factory(self, audio: AudioConfig) -> AudioEngine:
        self.engine = AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE)
        return self.engine

    def worker_factory(
        self,
        ring: RingBuffer,
        config: AnalysisConfig,
        sample_rate_hz: int,
        on_event: Callable[[MusicalEvent], None],
    ) -> AnalysisWorker:
        self.worker = ManualWorker(ring, config, sample_rate_hz, on_event=on_event)
        return self.worker

    def sleep(self, seconds: float) -> None:
        self.polls += 1
        if self.polls > self.polls_before_ctrl_c:
            raise KeyboardInterrupt
        assert self.engine is not None and self.worker is not None
        self.clock.advance(seconds)
        outdata = np.zeros((BLOCK, 2), dtype=np.float32)
        while self.position + BLOCK <= self.clock.frame_index:
            block = self.audio[self.position : self.position + BLOCK].astype(np.float32)
            if len(block) < BLOCK:
                block = np.zeros(BLOCK, dtype=np.float32)
            self.position += BLOCK
            self.engine._callback(block.reshape(-1, 1), outdata, BLOCK, None, None)
        self.worker.process_available()

    def run(
        self,
        output: io.StringIO,
        sleep: Callable[[float], None] | None = None,
        config: JamMateConfig = GUITAR_CONFIG,
    ) -> int:
        return run_headless(
            config,
            engine_factory=self.engine_factory,
            worker_factory=self.worker_factory,
            sleep=sleep or self.sleep,
            clock=self.clock,
            output=output,
        )


@pytest.mark.usefixtures("fake_stream")
def test_run_prints_status_until_ctrl_c_then_shuts_down(caplog: pytest.LogCaptureFixture) -> None:
    previous_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    session = Session(polls_before_ctrl_c=20)  # 1 s
    output = io.StringIO()

    with caplog.at_level(logging.INFO):
        exit_code = session.run(output)

    lines = output.getvalue().splitlines()
    assert exit_code == 0
    assert lines[0] == "Listening. Press Ctrl+C to stop."
    # The timeline, as it happened: playing starts, the note; then two status lines (0.5 s, 1 s).
    assert lines[1].endswith("[playing]")
    assert lines[2].startswith("  00:00.") and " A3 " in lines[2]
    assert len(lines) == 5
    assert all("note A3" in line and "peak   -6.0 dBFS" in line for line in lines[3:])
    assert session.worker is not None and session.worker.started and session.worker.stopped
    assert FakeStream.instances[0].closed
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == previous_handlers
    assert "Analysis: guitar, 60-1400 Hz, window 1602 frames" in caplog.text
    assert "Session:" in caplog.text
    assert "375 blocks of 128 frames at 48000 Hz" in caplog.text  # 1 s
    assert "Analysis: 100 hops" in caplog.text
    assert "1 notes" in caplog.text  # the 220 Hz test tone


@pytest.mark.usefixtures("fake_stream")
def test_run_prints_each_note_change_as_it_happens() -> None:
    tone = plucked_note("A2", 0.6, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    audio = np.concatenate([np.zeros(9600, dtype=np.float32), tone, np.zeros(24_000)])
    output = io.StringIO()

    Session(polls_before_ctrl_c=40, audio=audio.astype(np.float32)).run(output)

    lines = output.getvalue().splitlines()
    events = [line for line in lines if line.startswith("  00:")]
    assert [line.split(maxsplit=1)[1][:9] for line in events] == [
        "[playing]",
        "A2    110",
        "--   sile",
        "[stopped]",
    ]
    assert events[1].startswith("  00:00.2")
    status = [line for line in lines if line.startswith("00:")]
    assert any("note A2" in line for line in status)


@pytest.mark.usefixtures("fake_stream")
def test_run_rewrites_one_line_on_a_terminal() -> None:
    output = TtyStringIO()

    Session(polls_before_ctrl_c=20).run(output)

    text = output.getvalue()
    assert text.count("\r00:00:0") == 2  # the status line, rewritten in place
    assert text.count("\r  00:00.") == 2  # [playing] and the note, printed above it
    assert text.endswith("\n")


@pytest.mark.usefixtures("fake_stream")
def test_run_stops_with_error_when_stream_dies() -> None:
    session = Session(polls_before_ctrl_c=10)

    def device_unplugged(seconds: float) -> None:
        FakeStream.instances[0].active = False

    exit_code = session.run(io.StringIO(), sleep=device_unplugged)

    assert exit_code == 1
    assert FakeStream.instances[0].closed
    assert session.worker is not None and session.worker.stopped


def test_run_reports_engine_start_failure(fake_stream: type[FakeStream]) -> None:
    fake_stream.reject_block_sizes = frozenset({128, 256})
    session = Session(polls_before_ctrl_c=1)

    exit_code = session.run(io.StringIO())

    assert exit_code == 1
    assert session.worker is not None and session.worker.stopped


def test_run_reports_device_resolution_failure() -> None:
    def factory(audio: AudioConfig) -> AudioEngine:
        raise DeviceNotFoundError("No input device matches 'Behringer'.")

    exit_code = run_headless(JamMateConfig(), engine_factory=factory, output=io.StringIO())

    assert exit_code == 1

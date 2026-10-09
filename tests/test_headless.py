import io
import itertools
import logging
import math
import signal
from collections.abc import Callable

import numpy as np
import pytest

from jammate.audio.devices import DeviceNotFoundError
from jammate.audio.engine import AudioEngine, CallbackTiming, EngineStats
from jammate.config import AudioConfig, JamMateConfig
from jammate.headless import (
    SILENCE_FLOOR_DBFS,
    InputLevel,
    StatusSnapshot,
    amplitude_to_dbfs,
    format_elapsed,
    format_status,
    measure_level,
    run_headless,
)
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, FakeStream

BLOCK = 128


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def _stats(**overrides: object) -> EngineStats:
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


def _snapshot(level: InputLevel, **stat_overrides: object) -> StatusSnapshot:
    return StatusSnapshot(
        elapsed_seconds=3725.0,
        level=level,
        ring_fill_ratio=0.25,
        block_period_seconds=BLOCK / 48_000,
        stats=_stats(**stat_overrides),
    )


# --- pure helpers --------------------------------------------------------------------------------


def test_amplitude_to_dbfs() -> None:
    assert amplitude_to_dbfs(1.0) == 0.0
    assert amplitude_to_dbfs(0.5) == pytest.approx(-6.02, abs=0.01)
    assert amplitude_to_dbfs(0.0) == SILENCE_FLOOR_DBFS
    assert amplitude_to_dbfs(1e-9) == SILENCE_FLOOR_DBFS


def test_measure_level_of_sine() -> None:
    samples = 0.5 * np.sin(2 * math.pi * 82.41 * np.arange(48_000) / 48_000).astype(np.float32)

    level = measure_level(samples)

    assert level.peak_dbfs == pytest.approx(-6.02, abs=0.05)
    assert level.rms_dbfs == pytest.approx(-9.03, abs=0.05)


def test_measure_level_of_nothing_is_silence() -> None:
    level = measure_level(np.zeros(0, dtype=np.float32))

    assert level == InputLevel(SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS)


def test_format_elapsed() -> None:
    assert format_elapsed(3725.9) == "01:02:05"


def test_format_status_shows_all_diagnostics() -> None:
    line = format_status(_snapshot(InputLevel(-24.0, -6.5), input_overflows=2))

    assert line.startswith("01:02:05")
    assert "rms  -24.0 peak   -6.5 dBFS" in line
    assert "xruns in 2 out 0 (start-up 1)" in line
    assert "ring  25%" in line
    assert "callback  10/  40 us (2% load)" in line
    assert "CLIP" not in line
    assert "ERRORS" not in line


def test_format_status_flags_clipping_and_errors() -> None:
    line = format_status(_snapshot(InputLevel(-3.0, 0.0), callback_errors=3))

    assert "dBFS CLIP" in line
    assert "CALLBACK ERRORS 3" in line


# --- run_headless (real engine on a fake stream) -------------------------------------------------


class Session:
    """Drives `run_headless`: each fake sleep feeds audio blocks, then Ctrl+C is simulated."""

    def __init__(self, polls_before_ctrl_c: int, amplitude: float = 0.5) -> None:
        self.engine: AudioEngine | None = None
        self.polls = 0
        self.polls_before_ctrl_c = polls_before_ctrl_c
        self.amplitude = amplitude
        self.ticks = itertools.count(step=0.5)

    def engine_factory(self, audio: AudioConfig) -> AudioEngine:
        self.engine = AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE)
        return self.engine

    def sleep(self, seconds: float) -> None:
        self.polls += 1
        if self.polls > self.polls_before_ctrl_c:
            raise KeyboardInterrupt
        assert self.engine is not None
        indata = np.full((BLOCK, 1), self.amplitude, dtype=np.float32)
        outdata = np.zeros((BLOCK, 2), dtype=np.float32)
        for _ in range(4):
            self.engine._callback(indata, outdata, BLOCK, None, None)

    def clock(self) -> float:
        return float(next(self.ticks))

    def run(self, output: io.StringIO, sleep: Callable[[float], None] | None = None) -> int:
        return run_headless(
            JamMateConfig(),
            engine_factory=self.engine_factory,
            sleep=sleep or self.sleep,
            clock=self.clock,
            output=output,
        )


@pytest.mark.usefixtures("fake_stream")
def test_run_prints_status_until_ctrl_c_then_shuts_down(caplog: pytest.LogCaptureFixture) -> None:
    previous_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    session = Session(polls_before_ctrl_c=2)
    output = io.StringIO()

    with caplog.at_level(logging.INFO):
        exit_code = session.run(output)

    lines = output.getvalue().splitlines()
    assert exit_code == 0
    assert lines[0] == "Monitoring. Press Ctrl+C to stop."
    assert len(lines) == 3
    assert all("peak   -6.0 dBFS" in line for line in lines[1:])
    assert FakeStream.instances[0].closed
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == previous_handlers
    assert "Session:" in caplog.text
    assert "8 blocks of 128 frames at 48000 Hz" in caplog.text


@pytest.mark.usefixtures("fake_stream")
def test_run_rewrites_one_line_on_a_terminal() -> None:
    output = TtyStringIO()

    Session(polls_before_ctrl_c=2).run(output)

    text = output.getvalue()
    assert text.count("\r") == 2
    assert text.endswith("\n")


@pytest.mark.usefixtures("fake_stream")
def test_run_stops_with_error_when_stream_dies() -> None:
    session = Session(polls_before_ctrl_c=10)

    def device_unplugged(seconds: float) -> None:
        FakeStream.instances[0].active = False

    exit_code = session.run(io.StringIO(), sleep=device_unplugged)

    assert exit_code == 1
    assert FakeStream.instances[0].closed


def test_run_reports_engine_start_failure(fake_stream: type[FakeStream]) -> None:
    fake_stream.reject_block_sizes = frozenset({128, 256})

    exit_code = Session(polls_before_ctrl_c=1).run(io.StringIO())

    assert exit_code == 1


def test_run_reports_device_resolution_failure() -> None:
    def factory(audio: AudioConfig) -> AudioEngine:
        raise DeviceNotFoundError("No input device matches 'Behringer'.")

    exit_code = run_headless(JamMateConfig(), engine_factory=factory, output=io.StringIO())

    assert exit_code == 1

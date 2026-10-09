"""Headless runner: runs the audio engine from the terminal with a live status line.

Until the analysis worker exists (Step 2.2), the main thread is the input ring's only consumer
and uses it just for the level meter. Everything here runs on the main thread.
"""

import logging
import math
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import TextIO

import numpy as np

from jammate.audio.devices import DeviceResolutionError
from jammate.audio.engine import AudioEngine, AudioEngineError, EngineStats
from jammate.config import AudioConfig, JamMateConfig

logger = logging.getLogger(__name__)

STATUS_INTERVAL_SECONDS = 0.5
SILENCE_FLOOR_DBFS = -120.0
CLIP_THRESHOLD_DBFS = -0.1


@dataclass(frozen=True)
class InputLevel:
    rms_dbfs: float
    peak_dbfs: float


@dataclass(frozen=True)
class StatusSnapshot:
    elapsed_seconds: float
    level: InputLevel
    ring_fill_ratio: float
    block_period_seconds: float
    stats: EngineStats


def amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return SILENCE_FLOOR_DBFS
    return max(SILENCE_FLOOR_DBFS, 20.0 * math.log10(amplitude))


def measure_level(samples: np.ndarray) -> InputLevel:
    if samples.size == 0:
        return InputLevel(SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    return InputLevel(amplitude_to_dbfs(rms), amplitude_to_dbfs(peak))


def format_elapsed(elapsed_seconds: float) -> str:
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_status(snapshot: StatusSnapshot) -> str:
    stats = snapshot.stats
    timing = stats.callback_timing
    load_percent = 100.0 * timing.max_seconds / snapshot.block_period_seconds
    clip = " CLIP" if snapshot.level.peak_dbfs >= CLIP_THRESHOLD_DBFS else ""
    line = (
        f"{format_elapsed(snapshot.elapsed_seconds)}"
        f"  level rms {snapshot.level.rms_dbfs:6.1f} peak {snapshot.level.peak_dbfs:6.1f} dBFS"
        f"{clip}"
        f"  xruns in {stats.input_overflows} out {stats.output_underflows}"
        f" (start-up {stats.startup_xruns})"
        f"  ring {100.0 * snapshot.ring_fill_ratio:3.0f}%"
        f"  callback {timing.mean_seconds * 1e6:3.0f}/{timing.max_seconds * 1e6:4.0f} us"
        f" ({load_percent:.0f}% load)"
    )
    if stats.callback_errors:
        line += f"  CALLBACK ERRORS {stats.callback_errors}"
    return line


class _RunSummary:
    """Whole-run maxima. The timing window (~5 s) is longer than the status interval, so
    polling every interval sees every block's duration."""

    def __init__(self) -> None:
        self.peak_dbfs = SILENCE_FLOOR_DBFS
        self.max_callback_seconds = 0.0

    def update(self, level: InputLevel, stats: EngineStats) -> None:
        self.peak_dbfs = max(self.peak_dbfs, level.peak_dbfs)
        self.max_callback_seconds = max(
            self.max_callback_seconds, stats.callback_timing.max_seconds
        )


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def run_headless(
    config: JamMateConfig,
    *,
    engine_factory: Callable[[AudioConfig], AudioEngine] = AudioEngine.from_config,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    output: TextIO = sys.stdout,
    status_interval_seconds: float = STATUS_INTERVAL_SECONDS,
) -> int:
    """Monitor the bass until Ctrl+C or SIGTERM, then shut down cleanly. Returns an exit code."""
    try:
        engine = engine_factory(config.audio)
    except (DeviceResolutionError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    live_line = output.isatty()
    summary = _RunSummary()
    exit_code = 0
    started_seconds = clock()
    reported_errors = 0
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        info = engine.start()
        ring = engine.input_ring
        buffer = np.zeros((ring.capacity_frames, ring.channels), dtype=np.float32)
        started_seconds = clock()
        output.write("Monitoring. Press Ctrl+C to stop.\n")
        output.flush()

        while True:
            sleep(status_interval_seconds)
            if not engine.is_running:
                logger.error("The audio stream stopped unexpectedly (interface disconnected?).")
                exit_code = 1
                break

            fill_ratio = ring.available_frames() / ring.capacity_frames
            read = ring.read(buffer)
            level = measure_level(buffer[: read.frame_count, 0])
            stats = engine.stats()
            summary.update(level, stats)
            if stats.callback_errors > reported_errors:
                reported_errors = stats.callback_errors
                logger.error("Audio callback error: %r", engine.processor.last_error)

            line = format_status(
                StatusSnapshot(
                    elapsed_seconds=clock() - started_seconds,
                    level=level,
                    ring_fill_ratio=fill_ratio,
                    block_period_seconds=info.block_period_seconds,
                    stats=stats,
                )
            )
            output.write(f"\r{line}\x1b[K" if live_line else f"{line}\n")
            output.flush()
    except KeyboardInterrupt:
        pass
    except AudioEngineError as exc:
        logger.error("%s", exc)
        exit_code = 1
    finally:
        # A second Ctrl+C must not interrupt the shutdown.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if live_line:
                output.write("\n")
            engine.close()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)

    _log_summary(engine, clock() - started_seconds, summary)
    return exit_code


def _log_summary(engine: AudioEngine, elapsed_seconds: float, summary: _RunSummary) -> None:
    info = engine.stream_info
    if info is None:
        return
    stats = engine.stats()
    logger.info(
        "Session: %s, %d blocks of %d frames at %d Hz. Xruns: %d input overflows, "
        "%d output underflows (%d at start-up). Callback max %.0f us (%.0f%% of the block "
        "period). Input peak %.1f dBFS. Callback errors: %d.",
        format_elapsed(elapsed_seconds),
        stats.blocks_processed,
        info.block_size_frames,
        info.sample_rate_hz,
        stats.input_overflows,
        stats.output_underflows,
        stats.startup_xruns,
        summary.max_callback_seconds * 1e6,
        100.0 * summary.max_callback_seconds / info.block_period_seconds,
        summary.peak_dbfs,
        stats.callback_errors,
    )

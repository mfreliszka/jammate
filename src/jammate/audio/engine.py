"""Realtime audio engine: owns the PortAudio duplex stream and the per-block callback work.

`BlockProcessor` holds everything that runs on the PortAudio callback thread. It is plain NumPy so
tests can drive it without hardware. `AudioEngine` owns the stream lifecycle and runs on the
main thread.
"""

import logging
import math
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import numpy as np
import sounddevice as sd

from jammate.audio.devices import AudioDevice, query_inventory, resolve_device
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AudioConfig

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHANNELS = 2
CALLBACK_TIMING_WINDOW_BLOCKS = 2048
# CoreAudio reports an input overflow on the first block(s) of every stream start. Xruns within
# this window are counted as `startup_xruns`, so they don't mask real dropouts.
STARTUP_SETTLE_SECONDS = 0.05


class AudioEngineError(RuntimeError):
    """The audio stream could not be opened or does not run with the requested settings."""


@dataclass(frozen=True)
class CallbackTiming:
    """Duration of the per-block work over the most recent blocks."""

    blocks_measured: int
    mean_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class StreamInfo:
    """What the opened stream actually runs with (may differ from what was requested)."""

    sample_rate_hz: int
    block_size_frames: int
    input_channels: int
    output_channels: int
    input_latency_seconds: float
    output_latency_seconds: float

    @property
    def block_period_seconds(self) -> float:
        return self.block_size_frames / self.sample_rate_hz


@dataclass(frozen=True)
class EngineStats:
    frames_processed: int
    blocks_processed: int
    input_overflows: int
    output_underflows: int
    startup_xruns: int
    callback_errors: int
    callback_timing: CallbackTiming


class BlockProcessor:
    """Work done for every audio block, on the PortAudio callback thread.

    Must stay non-blocking and allocation-free (root AGENTS.md realtime rules). Counters are
    written only here and read by other threads for diagnostics.
    """

    def __init__(
        self,
        input_ring: RingBuffer,
        input_channel_index: int,
        monitor_gain: float,
        timing_window_blocks: int = CALLBACK_TIMING_WINDOW_BLOCKS,
    ) -> None:
        self.input_ring = input_ring
        self.input_channel_index = input_channel_index
        # Set from other threads by plain assignment, which is atomic under the GIL.
        self.monitor_gain = monitor_gain
        self.frames_processed = 0
        self.blocks_processed = 0
        self.input_overflows = 0
        self.output_underflows = 0
        self.startup_xruns = 0
        self.errors = 0
        self._settle_until_block = 0
        self.last_error: Exception | None = None
        self._timing_window_blocks = timing_window_blocks
        self._durations_ns = np.zeros(timing_window_blocks, dtype=np.int64)

    def process(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        input_overflow: bool = False,
        output_underflow: bool = False,
    ) -> None:
        """Handle one block: `indata` is `(frames, input_channels)`, `outdata` is written fully."""
        start_ns = time.perf_counter_ns()
        try:
            channel = self.input_channel_index
            bass = indata[:, channel : channel + 1]
            self.input_ring.write(bass)
            # The monitor is the first output bus: it writes every output sample, which also
            # replaces the zero-fill. Later buses are added on top.
            np.multiply(bass, self.monitor_gain, out=outdata)
        except Exception as exc:
            # Never let an exception reach PortAudio; the main thread reports `errors`.
            outdata.fill(0.0)
            self.errors += 1
            self.last_error = exc

        self.frames_processed += outdata.shape[0]
        if input_overflow or output_underflow:
            if self.blocks_processed < self._settle_until_block:
                self.startup_xruns += 1
            else:
                self.input_overflows += input_overflow
                self.output_underflows += output_underflow
        slot = self.blocks_processed % self._timing_window_blocks
        self._durations_ns[slot] = time.perf_counter_ns() - start_ns
        self.blocks_processed += 1

    def begin_settling(self, block_count: int) -> None:
        """Treat xruns in the next `block_count` blocks as start-up transients.

        Call from the main thread while the stream is stopped.
        """
        self._settle_until_block = self.blocks_processed + block_count

    def timing(self) -> CallbackTiming:
        """Read the timing window (not on the callback thread). Values may be one block stale."""
        measured = min(self.blocks_processed, self._timing_window_blocks)
        if measured == 0:
            return CallbackTiming(0, 0.0, 0.0)
        durations_ns = self._durations_ns[:measured].copy()
        return CallbackTiming(
            blocks_measured=measured,
            mean_seconds=float(durations_ns.mean()) / 1e9,
            max_seconds=float(durations_ns.max()) / 1e9,
        )


class AudioEngine:
    """Owns the duplex stream: mono bass in, stereo out. Use from the main thread only.

    Lifecycle: `start()` opens the stream if needed and starts it, `stop()` pauses it, and
    `close()` stops it and releases the devices. Also usable as a context manager that closes
    on exit.
    """

    def __init__(
        self, config: AudioConfig, input_device: AudioDevice, output_device: AudioDevice
    ) -> None:
        if config.input_channel_index >= input_device.max_input_channels:
            raise ValueError(
                f"Input channel {config.input_channel_index + 1} requested, but "
                f"{input_device.name!r} has {input_device.max_input_channels} input channel(s)."
            )
        if not output_device.is_output:
            raise ValueError(f"{output_device.name!r} has no output channels.")

        self._config = config
        self._input_device = input_device
        self._output_device = output_device
        self.input_ring = RingBuffer(config.input_ring_capacity_frames, channels=1)
        self.processor = BlockProcessor(
            self.input_ring, config.input_channel_index, config.monitor_gain
        )
        self._stream: sd.Stream | None = None
        self.stream_info: StreamInfo | None = None

    @classmethod
    def from_config(cls, config: AudioConfig) -> Self:
        """Resolve the configured devices by name (None = system default) and build an engine."""
        inventory = query_inventory()
        input_device = resolve_device(inventory, "input", config.input_device_name)
        output_device = resolve_device(inventory, "output", config.output_device_name)
        return cls(config, input_device, output_device)

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    def set_monitor_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= 4.0:
            raise ValueError("monitor gain must be between 0.0 and 4.0")
        self.processor.monitor_gain = float(gain)

    def start(self) -> StreamInfo:
        if self._stream is None or self.stream_info is None:
            self._stream, self.stream_info = self._open_stream()
        if not self._stream.active:
            settle_blocks = math.ceil(
                STARTUP_SETTLE_SECONDS / self.stream_info.block_period_seconds
            )
            self.processor.begin_settling(settle_blocks)
            self._stream.start()
            logger.info("Audio stream started.")
        return self.stream_info

    def stop(self) -> None:
        if self._stream is not None and self._stream.active:
            self._stream.stop()
            logger.info("Audio stream stopped. %s", self._format_counters())

    def close(self) -> None:
        if self._stream is None:
            return
        self.stop()
        self._stream.close()
        self._stream = None
        logger.info("Audio stream closed; devices released.")

    def stats(self) -> EngineStats:
        processor = self.processor
        return EngineStats(
            frames_processed=processor.frames_processed,
            blocks_processed=processor.blocks_processed,
            input_overflows=processor.input_overflows,
            output_underflows=processor.output_underflows,
            startup_xruns=processor.startup_xruns,
            callback_errors=processor.errors,
            callback_timing=processor.timing(),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        if status:
            self.processor.process(indata, outdata, status.input_overflow, status.output_underflow)
        else:
            self.processor.process(indata, outdata)

    def _open_stream(self) -> tuple[sd.Stream, StreamInfo]:
        config = self._config
        input_channels = config.input_channel_index + 1
        output_channels = min(MAX_OUTPUT_CHANNELS, self._output_device.max_output_channels)
        block_sizes = dict.fromkeys((config.block_size_frames, config.fallback_block_size_frames))

        last_error: Exception | None = None
        for block_size in block_sizes:
            try:
                stream = sd.Stream(
                    device=(self._input_device.index, self._output_device.index),
                    channels=(input_channels, output_channels),
                    samplerate=config.sample_rate_hz,
                    blocksize=block_size,
                    dtype="float32",
                    latency="low",
                    callback=self._callback,
                )
            except sd.PortAudioError as exc:
                logger.warning(
                    "Could not open the stream with %d-frame blocks: %s", block_size, exc
                )
                last_error = exc
                continue

            input_latency, output_latency = stream.latency
            info = StreamInfo(
                sample_rate_hz=round(stream.samplerate),
                block_size_frames=stream.blocksize,
                input_channels=input_channels,
                output_channels=output_channels,
                input_latency_seconds=float(input_latency),
                output_latency_seconds=float(output_latency),
            )
            if info.sample_rate_hz != config.sample_rate_hz:
                # Never change the sample rate silently (§141).
                stream.close()
                raise AudioEngineError(
                    f"Stream runs at {info.sample_rate_hz} Hz, expected {config.sample_rate_hz} Hz."
                )
            logger.info(
                "Audio stream opened: in %r channel %d, out %r (%d ch), %d Hz, %d-frame blocks, "
                "latency in %.1f ms / out %.1f ms.",
                self._input_device.name,
                config.input_channel_index + 1,
                self._output_device.name,
                output_channels,
                info.sample_rate_hz,
                info.block_size_frames,
                info.input_latency_seconds * 1000,
                info.output_latency_seconds * 1000,
            )
            return stream, info

        raise AudioEngineError(
            f"Could not open the audio stream with block sizes {list(block_sizes)}."
        ) from last_error

    def _format_counters(self) -> str:
        stats = self.stats()
        return (
            f"{stats.blocks_processed} blocks, {stats.input_overflows} input overflows, "
            f"{stats.output_underflows} output underflows, {stats.startup_xruns} start-up xruns, "
            f"{stats.callback_errors} callback errors"
        )

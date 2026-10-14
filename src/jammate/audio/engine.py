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
from jammate.audio.recorder import STEM_RECORDER_CHANNELS
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AudioConfig
from jammate.drums.sampler import MAX_RENDER_BLOCK_FRAMES, DrumSampler

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHANNELS = 2
CALLBACK_TIMING_WINDOW_BLOCKS = 2048
# CoreAudio reports an input overflow on the first block(s) of every stream start. Xruns within
# this window are counted as `startup_xruns`, so they don't mask real dropouts.
STARTUP_SETTLE_SECONDS = 0.05


def soft_clip_inplace(samples: np.ndarray, threshold: float, scratch: np.ndarray) -> None:
    """Apply a C1-continuous soft-clip in place with zero allocations.

    Linear (exact identity) for `|x| <= threshold`; above `threshold`, excess is compressed via a
    smooth `tanh` knee approaching `±1.0`. `scratch` must be preallocated with the same shape as
    `samples`.
    """
    headroom = 1.0 - threshold
    np.clip(samples, -threshold, threshold, out=scratch)
    np.subtract(samples, scratch, out=samples)
    np.divide(samples, headroom, out=samples)
    np.tanh(samples, out=samples)
    np.multiply(samples, headroom, out=samples)
    np.add(scratch, samples, out=samples)


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
    ai_underruns: int = 0


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
        *,
        monitor_muted: bool = False,
        drums_gain: float = 0.8,
        drums_muted: bool = False,
        drum_sampler: DrumSampler | None = None,
        ai_gain: float = 0.7,
        ai_muted: bool = False,
        ai_ring: RingBuffer | None = None,
        record_ring: RingBuffer | None = None,
        stem_record_ring: RingBuffer | None = None,
        soft_clip_threshold: float = 0.85,
        max_block_frames: int = MAX_RENDER_BLOCK_FRAMES,
    ) -> None:
        self.input_ring = input_ring
        self.input_channel_index = input_channel_index
        # Bus controls are set from other threads by plain assignment (atomic under the GIL).
        self.monitor_gain = monitor_gain
        self.monitor_muted = monitor_muted
        self.drums_gain = drums_gain
        self.drums_muted = drums_muted
        self.drum_sampler = drum_sampler
        self.ai_gain = ai_gain
        self.ai_muted = ai_muted
        self.ai_ring = ai_ring
        self.record_ring = record_ring
        self.stem_record_ring = stem_record_ring
        self.soft_clip_threshold = soft_clip_threshold

        self.frames_processed = 0
        self.blocks_processed = 0
        self.input_overflows = 0
        self.output_underflows = 0
        self.startup_xruns = 0
        self.ai_underruns = 0
        self.errors = 0
        self._settle_until_block = 0
        self.last_error: Exception | None = None
        self._timing_window_blocks = timing_window_blocks
        self._durations_ns = np.zeros(timing_window_blocks, dtype=np.int64)
        self._bus_scratch = np.zeros((max_block_frames, MAX_OUTPUT_CHANNELS), dtype=np.float32)
        self._clip_scratch = np.zeros((max_block_frames, MAX_OUTPUT_CHANNELS), dtype=np.float32)
        self._stem_scratch = np.zeros((max_block_frames, STEM_RECORDER_CHANNELS), dtype=np.float32)

    def process(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        input_overflow: bool = False,
        output_underflow: bool = False,
    ) -> None:
        """Handle one block: `indata` is `(frames, input_channels)`, `outdata` is written fully."""
        start_ns = time.perf_counter_ns()
        block_frames = outdata.shape[0]
        out_channels = outdata.shape[1]
        block_start_frame = self.frames_processed
        try:
            channel = self.input_channel_index
            bass = indata[:, channel : channel + 1]
            self.input_ring.write(bass)

            stem_ring = self.stem_record_ring
            stem_buf = self._stem_scratch[:block_frames] if stem_ring is not None else None

            # 1. Monitor bus: writes every output sample, replacing an explicit zero-fill.
            monitor_gain = 0.0 if self.monitor_muted else self.monitor_gain
            if monitor_gain == 0.0:
                outdata.fill(0.0)
                if stem_buf is not None:
                    stem_buf[:, 0:1].fill(0.0)
            else:
                np.multiply(bass, monitor_gain, out=outdata)
                if stem_buf is not None:
                    np.copyto(stem_buf[:, 0:1], outdata[:, 0:1])

            # 2. Drums bus: always advance the sampler so muted events never pile up in the ring.
            sampler = self.drum_sampler
            if sampler is not None:
                bus = self._bus_scratch[:block_frames, :out_channels]
                sampler.render_block(bus, block_start_frame)
                drums_gain = 0.0 if self.drums_muted else self.drums_gain
                if drums_gain > 0.0:
                    np.multiply(bus, drums_gain, out=bus)
                    np.add(outdata, bus, out=outdata)
                    if stem_buf is not None:
                        np.copyto(
                            stem_buf[:, 1:3],
                            bus[:, :2] if out_channels >= 2 else bus[:, :1],
                        )
                elif stem_buf is not None:
                    stem_buf[:, 1:3].fill(0.0)
            elif stem_buf is not None:
                stem_buf[:, 1:3].fill(0.0)

            # 3. AI accompaniment bus: consume from the AI ring when present.
            ai_ring = self.ai_ring
            if ai_ring is not None:
                ring_channels = ai_ring.channels
                ai_buf = self._bus_scratch[:block_frames, :ring_channels]
                read = ai_ring.read(ai_buf)
                if read.frame_count < block_frames:
                    ai_buf[read.frame_count :].fill(0.0)
                    self.ai_underruns += 1
                ai_gain = 0.0 if self.ai_muted else self.ai_gain
                if ai_gain > 0.0 and read.frame_count > 0:
                    bus = ai_buf[:, :out_channels]
                    np.multiply(bus, ai_gain, out=bus)
                    np.add(outdata, bus, out=outdata)
                    if stem_buf is not None:
                        np.copyto(
                            stem_buf[:, 3:5],
                            bus[:, :2] if out_channels >= 2 else bus[:, :1],
                        )
                elif stem_buf is not None:
                    stem_buf[:, 3:5].fill(0.0)
            elif stem_buf is not None:
                stem_buf[:, 3:5].fill(0.0)

            # 4. Master safety soft-clip.
            clip_scratch = self._clip_scratch[:block_frames, :out_channels]
            soft_clip_inplace(outdata, self.soft_clip_threshold, clip_scratch)

            # 5. Master stereo recording tap (Step 4.5, §59) & 7-channel stem tap (Step 6.2, §150).
            record_ring = self.record_ring
            if record_ring is not None:
                if out_channels == record_ring.channels:
                    record_ring.write(outdata)
                else:
                    rec_buf = self._bus_scratch[:block_frames, : record_ring.channels]
                    np.copyto(rec_buf, outdata[:, :1])
                    record_ring.write(rec_buf)

            if stem_ring is not None and stem_buf is not None:
                np.copyto(
                    stem_buf[:, 5:7],
                    outdata[:, :2] if out_channels >= 2 else outdata[:, :1],
                )
                stem_ring.write(stem_buf)
        except Exception as exc:
            # Never let an exception reach PortAudio; the main thread reports `errors`.
            outdata.fill(0.0)
            self.errors += 1
            self.last_error = exc

        self.frames_processed = block_start_frame + block_frames
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
        self,
        config: AudioConfig,
        input_device: AudioDevice,
        output_device: AudioDevice,
        *,
        drum_sampler: DrumSampler | None = None,
        ai_ring: RingBuffer | None = None,
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
            self.input_ring,
            config.input_channel_index,
            config.monitor_gain,
            monitor_muted=config.monitor_muted,
            drums_gain=config.drums_gain,
            drums_muted=config.drums_muted,
            drum_sampler=drum_sampler,
            ai_gain=config.ai_gain,
            ai_muted=config.ai_muted,
            ai_ring=ai_ring,
            soft_clip_threshold=config.soft_clip_threshold,
        )
        self._stream: sd.Stream | None = None
        self.stream_info: StreamInfo | None = None

    @classmethod
    def from_config(
        cls,
        config: AudioConfig,
        *,
        drum_sampler: DrumSampler | None = None,
        ai_ring: RingBuffer | None = None,
    ) -> Self:
        """Resolve the configured devices by name (None = system default) and build an engine."""
        inventory = query_inventory()
        input_device = resolve_device(inventory, "input", config.input_device_name)
        output_device = resolve_device(inventory, "output", config.output_device_name)
        return cls(
            config,
            input_device,
            output_device,
            drum_sampler=drum_sampler,
            ai_ring=ai_ring,
        )

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    def set_monitor_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= 4.0:
            raise ValueError("monitor gain must be between 0.0 and 4.0")
        self.processor.monitor_gain = float(gain)

    def set_monitor_muted(self, muted: bool) -> None:
        self.processor.monitor_muted = bool(muted)

    def set_drums_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= 4.0:
            raise ValueError("drums gain must be between 0.0 and 4.0")
        self.processor.drums_gain = float(gain)

    def set_drums_muted(self, muted: bool) -> None:
        self.processor.drums_muted = bool(muted)

    def set_ai_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= 4.0:
            raise ValueError("ai gain must be between 0.0 and 4.0")
        self.processor.ai_gain = float(gain)

    def set_ai_muted(self, muted: bool) -> None:
        self.processor.ai_muted = bool(muted)

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
            callback_timing=processor.callback_timing
            if hasattr(processor, "callback_timing")
            else processor.timing(),
            ai_underruns=processor.ai_underruns,
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

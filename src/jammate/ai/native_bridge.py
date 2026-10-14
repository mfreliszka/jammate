"""Native/shared-memory AccompanimentEngine ring bridge (Step 7.1, §19, §51, §87, §89, §151).

Allows an external Python/C++ MRT2 runner process to stream 48 kHz stereo `float32` accompaniment
frames into JamMate's `AudioEngine` over POSIX shared memory (`multiprocessing.shared_memory`)
without holding the main process GIL during model inference.

Architecture:
- `SharedMemoryAudioRing`: lock-free single-producer/single-consumer (SPSC) stereo `float32` ring
  backed by a `SharedMemory` segment with a 64-byte `int64` control header (`write_pos`,
  `read_pos`, `capacity_frames`, `channels`, `dropped_frames`, `active_note`, `intensity_milli`,
  `follow_improvise_milli`) followed by `capacity_frames * channels` contiguous `float32` samples.
  Uses the C++20 `ring_write_f32` and `ring_copy_range_f32` kernels (with automatic NumPy fallback).
- `NativeBridgeAccompaniment`: implements `AccompanimentEngine` (`ai/base.py`), exposing a standard
  `RingBuffer` as `output_ring` for the realtime audio callback while draining newly produced frames
  from `SharedMemoryAudioRing` on a lightweight bridge worker thread (or deterministically via
  `drain_once()` in tests).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from jammate.ai.base import AccompanimentStats, AccompanimentStatus
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AccompanimentConfig
from jammate.music.events import MidiNote
from jammate.native.kernels import ring_copy_range_f32, ring_write_f32

logger = logging.getLogger(__name__)

_HEADER_INT64_COUNT = 8
_HEADER_BYTES = _HEADER_INT64_COUNT * 8  # 64-byte cache-line-aligned header

_IDX_WRITE_POS = 0
_IDX_READ_POS = 1
_IDX_CAPACITY = 2
_IDX_CHANNELS = 3
_IDX_DROPPED = 4
_IDX_ACTIVE_NOTE = 5
_IDX_INTENSITY_MILLI = 6
_IDX_FOLLOW_IMPROVISE_MILLI = 7

_MRT_FRAME_SAMPLES = 1920  # 40 ms @ 48 kHz


class SharedMemoryAudioRing:
    """Lock-free SPSC `float32` audio ring buffer in POSIX shared memory (§19, §89, §151).

    Can either create a new shared memory segment (`create=True`) or attach to an existing one by
    `name` (`create=False`).
    """

    def __init__(
        self,
        capacity_frames: int = 48_000 * 6,
        channels: int = 2,
        *,
        name: str | None = None,
        create: bool = True,
        use_native: bool = True,
    ) -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._use_native = use_native
        self._owns_shm = create
        self._closed = False
        audio_bytes = capacity_frames * channels * np.dtype(np.float32).itemsize
        total_bytes = _HEADER_BYTES + audio_bytes

        if create:
            self._shm = SharedMemory(name=name, create=True, size=total_bytes)
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._shm = SharedMemory(name=name, create=False)

        buf = self._shm.buf
        assert buf is not None
        self._header = np.ndarray(
            (_HEADER_INT64_COUNT,),
            dtype=np.int64,
            buffer=buf[:_HEADER_BYTES],
        )
        if create:
            self._header[_IDX_WRITE_POS] = 0
            self._header[_IDX_READ_POS] = 0
            self._header[_IDX_CAPACITY] = capacity_frames
            self._header[_IDX_CHANNELS] = channels
            self._header[_IDX_DROPPED] = 0
            self._header[_IDX_ACTIVE_NOTE] = -1
            self._header[_IDX_INTENSITY_MILLI] = 500
            self._header[_IDX_FOLLOW_IMPROVISE_MILLI] = 500
        else:
            capacity_frames = int(self._header[_IDX_CAPACITY])
            channels = int(self._header[_IDX_CHANNELS])
            audio_bytes = capacity_frames * channels * np.dtype(np.float32).itemsize

        self._capacity = capacity_frames
        self._channels = channels
        self._audio = np.ndarray(
            (capacity_frames, channels),
            dtype=np.float32,
            buffer=buf[_HEADER_BYTES : _HEADER_BYTES + audio_bytes],
        )
        if create:
            self._audio.fill(0.0)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def write_pos(self) -> int:
        return int(self._header[_IDX_WRITE_POS])

    @property
    def read_pos(self) -> int:
        return int(self._header[_IDX_READ_POS])

    @property
    def dropped_frames(self) -> int:
        return int(self._header[_IDX_DROPPED])

    @property
    def active_note(self) -> MidiNote | None:
        raw = int(self._header[_IDX_ACTIVE_NOTE])
        return raw if 0 <= raw < 128 else None

    @property
    def intensity(self) -> float:
        return float(self._header[_IDX_INTENSITY_MILLI]) / 1000.0

    @property
    def follow_improvise(self) -> float:
        return float(self._header[_IDX_FOLLOW_IMPROVISE_MILLI]) / 1000.0

    def set_conditioning(
        self,
        *,
        active_note: MidiNote | None = None,
        intensity: float | None = None,
        follow_improvise: float | None = None,
    ) -> None:
        """Update control fields in the shared-memory header for an external runner process."""
        self._header[_IDX_ACTIVE_NOTE] = -1 if active_note is None else int(active_note)
        if intensity is not None:
            clamped_i = max(0.0, min(1.0, float(intensity)))
            self._header[_IDX_INTENSITY_MILLI] = round(clamped_i * 1000.0)
        if follow_improvise is not None:
            clamped_f = max(0.0, min(1.0, float(follow_improvise)))
            self._header[_IDX_FOLLOW_IMPROVISE_MILLI] = round(clamped_f * 1000.0)

    def available_read(self) -> int:
        """Number of unread frames currently available (`0..capacity`)."""
        avail = int(self._header[_IDX_WRITE_POS]) - int(self._header[_IDX_READ_POS])
        if avail <= 0:
            return 0
        return min(avail, self._capacity)

    def available_frames(self) -> int:
        """Alias for `available_read()` matching `RingBuffer.available_frames()`."""
        return self.available_read()

    def write(self, frames: np.ndarray) -> int:
        """Producer: write `(N, channels)` (or 1D mono) `float32` frames into shared memory."""
        if frames.ndim == 1:
            frames_2d = (
                frames[:, np.newaxis] if self._channels == 1 else np.column_stack((frames, frames))
            )
        else:
            frames_2d = frames
        if frames_2d.shape[1] != self._channels:
            raise ValueError(f"Expected {self._channels} channels, got shape {frames_2d.shape}")
        frame_count = int(frames_2d.shape[0])
        if frame_count <= 0:
            return 0

        src = np.ascontiguousarray(frames_2d, dtype=np.float32)
        if frame_count > self._capacity:
            src = src[-self._capacity :]
            frame_count = self._capacity

        write_pos = int(self._header[_IDX_WRITE_POS])
        ring_write_f32(
            self._audio,
            write_pos,
            src,
            use_native=self._use_native,
        )
        self._header[_IDX_WRITE_POS] = write_pos + frame_count
        return frame_count

    def read_into(self, out: np.ndarray) -> int:
        """Consumer: copy up to `len(out)` unread frames into `out` and advance `read_pos`."""
        write_pos = int(self._header[_IDX_WRITE_POS])
        read_pos = int(self._header[_IDX_READ_POS])
        unread = write_pos - read_pos
        if unread <= 0:
            return 0

        if unread > self._capacity:
            dropped = unread - self._capacity
            self._header[_IDX_DROPPED] = int(self._header[_IDX_DROPPED]) + dropped
            read_pos = write_pos - self._capacity

        to_read = min(int(out.shape[0]), write_pos - read_pos)
        if to_read <= 0:
            return 0

        dst = out if out.ndim == 2 else out[:, np.newaxis]
        ring_copy_range_f32(
            self._audio,
            read_pos,
            to_read,
            dst,
            use_native=self._use_native,
        )
        self._header[_IDX_READ_POS] = read_pos + to_read
        return to_read

    def close(self) -> None:
        """Close the shared memory mapping in this process (idempotent)."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._shm.close()

    def unlink(self) -> None:
        """Close and unlink the shared memory segment if created by this instance."""
        self.close()
        if self._owns_shm:
            self._owns_shm = False
            with contextlib.suppress(FileNotFoundError, Exception):
                self._shm.unlink()


class NativeBridgeAccompaniment:
    """`AccompanimentEngine` bridging `SharedMemoryAudioRing` into `AudioEngine` (§89, §151).

    An external MRT2 process (or an injected `producer_fn` callback) writes stereo `float32`
    frames into `shm_ring`. A lightweight drain worker copies available frames into `output_ring`
    so the realtime PortAudio callback reads from standard `RingBuffer` memory with zero locks
    and zero GIL contention during model inference.
    """

    def __init__(
        self,
        config: AccompanimentConfig | None = None,
        *,
        sample_rate_hz: int = 48_000,
        shm_name: str | None = None,
        create_shm: bool = True,
        use_native: bool = True,
        producer_fn: Callable[[SharedMemoryAudioRing], float | None] | None = None,
        drain_chunk_frames: int = 1920,
        poll_interval_seconds: float = 0.002,
    ) -> None:
        self._config = config or AccompanimentConfig()
        self._sample_rate_hz = sample_rate_hz
        self._use_native = use_native
        self._producer_fn = producer_fn
        self._drain_chunk_frames = max(128, int(drain_chunk_frames))
        self._poll_interval_seconds = max(0.0005, float(poll_interval_seconds))

        capacity_frames = max(
            self._drain_chunk_frames * 2,
            round(self._config.ring_capacity_seconds * sample_rate_hz),
        )
        self._output_ring = RingBuffer(capacity_frames=capacity_frames, channels=2)
        self._shm_ring = SharedMemoryAudioRing(
            capacity_frames=capacity_frames,
            channels=2,
            name=shm_name,
            create=create_shm,
            use_native=use_native,
        )
        self._drain_scratch = np.zeros((self._drain_chunk_frames, 2), dtype=np.float32)

        self._status = (
            AccompanimentStatus.READY if self._config.enabled else AccompanimentStatus.DISABLED
        )
        self._style = self._config.style
        self._active_note: MidiNote | None = None
        self._intensity: float = 0.5
        self._follow_improvise: float = self._config.follow_improvise
        self._frames_generated: int = 0
        self._samples_drained: int = 0
        self._ms_per_frame: float = 0.0
        self._last_error: str | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def output_ring(self) -> RingBuffer | None:
        if self._status in (
            AccompanimentStatus.DISABLED,
            AccompanimentStatus.UNAVAILABLE,
            AccompanimentStatus.FAILED,
        ):
            return None
        return self._output_ring

    @property
    def shm_ring(self) -> SharedMemoryAudioRing:
        return self._shm_ring

    @property
    def shm_name(self) -> str:
        return self._shm_ring.name

    @property
    def status(self) -> AccompanimentStatus:
        return self._status

    def start(self) -> None:
        if self._status == AccompanimentStatus.DISABLED:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._status = AccompanimentStatus.RUNNING
        self._thread = threading.Thread(
            target=self._run_loop,
            name="jammate-native-ai-bridge",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        self._active_note = None
        with contextlib.suppress(Exception):
            self._shm_ring.set_conditioning(active_note=None)
        if self._status == AccompanimentStatus.RUNNING:
            self._status = AccompanimentStatus.STOPPED

    def close(self) -> None:
        """Stop the bridge thread and unlink/close the underlying shared memory segment."""
        self.stop()
        self._shm_ring.unlink()

    def set_style(self, style: str) -> None:
        self._style = style

    def note_on(self, midi_note: MidiNote) -> None:
        self._active_note = midi_note
        self._shm_ring.set_conditioning(
            active_note=midi_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
        )

    def note_off(self, midi_note: MidiNote) -> None:
        if self._active_note == midi_note:
            self._active_note = None
            self._shm_ring.set_conditioning(
                active_note=None,
                intensity=self._intensity,
                follow_improvise=self._follow_improvise,
            )

    def set_intensity(self, intensity: float) -> None:
        self._intensity = max(0.0, min(1.0, float(intensity)))
        self._shm_ring.set_conditioning(
            active_note=self._active_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
        )

    def set_follow_improvise(self, follow_improvise: float) -> None:
        self._follow_improvise = max(0.0, min(1.0, float(follow_improvise)))
        self._shm_ring.set_conditioning(
            active_note=self._active_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
        )

    def push_external_frames(self, frames: np.ndarray, *, ms_per_frame: float = 0.0) -> int:
        """Write stereo `float32` frames into `shm_ring` (for external runner or tests)."""
        written = self._shm_ring.write(frames)
        if ms_per_frame > 0.0:
            self._ms_per_frame = float(ms_per_frame)
        return written

    def drain_once(self) -> int:
        """Drain all currently available frames from `shm_ring` into `output_ring`."""
        total_drained = 0
        while True:
            avail = self._shm_ring.available_read()
            if avail <= 0:
                break
            chunk = min(avail, self._drain_chunk_frames)
            scratchView = self._drain_scratch[:chunk]
            read_count = self._shm_ring.read_into(scratchView)
            if read_count <= 0:
                break
            self._output_ring.write(scratchView[:read_count])
            total_drained += read_count
            self._samples_drained += read_count
            self._frames_generated = self._samples_drained // _MRT_FRAME_SAMPLES
        return total_drained

    def stats(self) -> AccompanimentStats:
        buffered_frames = self._output_ring.available_frames() + self._shm_ring.available_frames()
        buffer_seconds = buffered_frames / float(self._sample_rate_hz)
        return AccompanimentStats(
            status=self._status,
            style=self._style,
            active_note=self._active_note,
            intensity=self._intensity,
            follow_improvise=self._follow_improvise,
            buffer_seconds=buffer_seconds,
            ms_per_frame=self._ms_per_frame,
            frames_generated=self._frames_generated,
            last_error=self._last_error,
        )

    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._producer_fn is not None:
                    t0 = time.perf_counter()
                    reported_ms = self._producer_fn(self._shm_ring)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if reported_ms is not None and reported_ms > 0.0:
                        self._ms_per_frame = float(reported_ms)
                    elif elapsed_ms > 0.0:
                        self._ms_per_frame = elapsed_ms

                drained = self.drain_once()
                if drained == 0 or self._producer_fn is not None:
                    self._stop_event.wait(self._poll_interval_seconds)
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            self._status = AccompanimentStatus.FAILED
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.error("NativeBridgeAccompaniment worker failed: %s", self._last_error)

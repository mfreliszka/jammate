"""Realtime-safe master stereo WAV recorder (Step 4.5, §§59, 84, 129).

The audio callback writes the post-clip stereo `outdata` block into a preallocated `RingBuffer`
(`channels=2`). A background writer thread (`jammate-recorder`) drains the ring and streams the
audio to `recordings/YYYY-MM-DD_HH-MM-jam.wav` with `soundfile`. Overflow never blocks the audio
callback; dropped frames are counted via the ring's overrun counter. On `stop()`, the remaining
buffered frames are drained and flushed before closing the WAV file (§84).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from jammate.audio.ring_buffer import RingBuffer
from jammate.config import RecordingConfig
from jammate.music.events import seconds_to_frames

logger = logging.getLogger(__name__)

RECORDER_CHANNELS = 2
WRITER_CHUNK_FRAMES = 4_096
WRITER_POLL_SECONDS = 0.010


@dataclass(frozen=True, slots=True)
class RecorderStats:
    """Diagnostics snapshot for `MasterRecorder`."""

    is_recording: bool
    file_path: Path | None
    frames_written: int
    dropped_frames: int
    overrun_events: int


def format_recording_filename(now: datetime | None = None) -> str:
    """Return a wall-clock filename `YYYY-MM-DD_HH-MM-jam.wav` (§59, §129)."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    return f"{stamp}-jam.wav"


def resolve_recording_path(output_dir: Path, now: datetime | None = None) -> Path:
    """Resolve a non-colliding `YYYY-MM-DD_HH-MM-jam.wav` path inside `output_dir`."""
    base_name = format_recording_filename(now)
    candidate = output_dir / base_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        numbered = output_dir / f"{stem}-{counter}{suffix}"
        if not numbered.exists():
            return numbered
        counter += 1


class MasterRecorder:
    """Streams master stereo output from a bounded `RingBuffer` to a WAV file on disk (§59)."""

    def __init__(
        self,
        config: RecordingConfig,
        sample_rate_hz: int = 48_000,
    ) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        capacity_frames = max(
            WRITER_CHUNK_FRAMES * 2,
            seconds_to_frames(config.ring_capacity_seconds, sample_rate_hz),
        )
        self._ring = RingBuffer(capacity_frames=capacity_frames, channels=RECORDER_CHANNELS)
        self._read_scratch = np.zeros((WRITER_CHUNK_FRAMES, RECORDER_CHANNELS), dtype=np.float32)
        self._sound_file: sf.SoundFile | None = None
        self._file_path: Path | None = None
        self._frames_written = 0
        self._is_recording = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._io_lock = threading.Lock()

    @property
    def ring(self) -> RingBuffer:
        """The underlying stereo `RingBuffer` (preallocated at construction)."""
        return self._ring

    @property
    def active_ring(self) -> RingBuffer | None:
        """Return `ring` while recording is active, else `None` for the audio callback."""
        return self._ring if self._is_recording else None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def dropped_frames(self) -> int:
        return self._ring.dropped_frames

    def stats(self) -> RecorderStats:
        return RecorderStats(
            is_recording=self._is_recording,
            file_path=self._file_path,
            frames_written=self._frames_written,
            dropped_frames=self._ring.dropped_frames,
            overrun_events=self._ring.overrun_count,
        )

    def open_file(self, now: datetime | None = None) -> Path:
        """Open the target WAV file and reset counters without starting the worker thread."""
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        path = resolve_recording_path(self._config.output_dir, now=now)
        # Drain any stale samples from a prior run before opening a new file.
        while self._ring.available_frames() > 0:
            self._ring.read(self._read_scratch)
        self._ring.dropped_frames = 0
        self._ring.overrun_count = 0
        self._sound_file = sf.SoundFile(
            path,
            mode="w",
            samplerate=self._sample_rate_hz,
            channels=RECORDER_CHANNELS,
            subtype="FLOAT",
        )
        self._file_path = path
        self._frames_written = 0
        self._is_recording = True
        return path

    def start(self, now: datetime | None = None) -> Path:
        """Open a new WAV file and start the background writer thread (`jammate-recorder`)."""
        if self._is_recording:
            assert self._file_path is not None
            return self._file_path
        path = self.open_file(now=now)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="jammate-recorder",
            daemon=False,
        )
        self._thread.start()
        logger.info("Recording master output to %s", path)
        return path

    def drain_available(self) -> int:
        """Drain all currently available frames from `ring` into the open WAV file."""
        with self._io_lock:
            if self._sound_file is None:
                return 0
            written_now = 0
            while True:
                avail = self._ring.available_frames()
                if avail <= 0:
                    break
                to_read = min(avail, WRITER_CHUNK_FRAMES)
                buf = self._read_scratch[:to_read]
                result = self._ring.read(buf)
                if result.frame_count <= 0:
                    break
                self._sound_file.write(buf[: result.frame_count])
                self._frames_written += result.frame_count
                written_now += result.frame_count
            return written_now

    def stop(self) -> Path | None:
        """Stop the writer thread, flush remaining ring frames, and close the WAV file (§84)."""
        if not self._is_recording and self._sound_file is None:
            return self._file_path
        self._is_recording = False
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join()

        self.drain_available()
        with self._io_lock:
            sf_obj = self._sound_file
            self._sound_file = None
            if sf_obj is not None:
                sf_obj.flush()
                sf_obj.close()
        if self._file_path is not None:
            logger.info(
                "Recording stopped: %s (%d frames written, %d dropped).",
                self._file_path,
                self._frames_written,
                self._ring.dropped_frames,
            )
        return self._file_path

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._ring.available_frames() > 0:
                    self.drain_available()
                else:
                    self._stop_event.wait(WRITER_POLL_SECONDS)
        except Exception as exc:
            logger.error("MasterRecorder writer thread failed: %s", exc)
            self._is_recording = False

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
STEM_RECORDER_CHANNELS = 7
WRITER_CHUNK_FRAMES = 4_096
WRITER_POLL_SECONDS = 0.010

STEM_FILENAMES: dict[str, str] = {
    "bass": "bass.wav",
    "drums": "drums.wav",
    "ai_guitar": "ai_guitar.wav",
    "master": "master.wav",
}
STEM_CHANNELS: dict[str, int] = {
    "bass": 1,
    "drums": 2,
    "ai_guitar": 2,
    "master": 2,
}


@dataclass(frozen=True, slots=True)
class RecorderStats:
    """Diagnostics snapshot for `MasterRecorder` and `StemRecorder`."""

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


def format_stem_dirname(now: datetime | None = None) -> str:
    """Return a wall-clock stem directory name `YYYY-MM-DD_HH-MM-SS-jam` (§59, §150)."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}-jam"


def resolve_stem_dir(output_dir: Path, now: datetime | None = None) -> Path:
    """Resolve a non-colliding `YYYY-MM-DD_HH-MM-SS-jam` directory inside `output_dir`."""
    base_name = format_stem_dirname(now)
    candidate = output_dir / base_name
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        numbered = output_dir / f"{base_name}-{counter}"
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


class StemRecorder:
    """Streams 4 frame-aligned WAV stems from a 7-channel `RingBuffer` (Step 6.2, §59, §150).

    Channel layout in the 7-channel ring buffer:
    - `[0:1]`: `bass.wav` (mono instrument monitor bus)
    - `[1:3]`: `drums.wav` (stereo drum bus)
    - `[3:5]`: `ai_guitar.wav` (stereo AI accompaniment bus)
    - `[5:7]`: `master.wav` (post-soft-clip stereo master output)
    """

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
        self._ring = RingBuffer(capacity_frames=capacity_frames, channels=STEM_RECORDER_CHANNELS)
        self._read_scratch = np.zeros(
            (WRITER_CHUNK_FRAMES, STEM_RECORDER_CHANNELS), dtype=np.float32
        )
        self._sound_files: dict[str, sf.SoundFile] = {}
        self._stem_paths: dict[str, Path] = {}
        self._dir_path: Path | None = None
        self._frames_written = 0
        self._is_recording = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._io_lock = threading.Lock()

    @property
    def ring(self) -> RingBuffer:
        """The underlying 7-channel `RingBuffer` (preallocated at construction)."""
        return self._ring

    @property
    def active_ring(self) -> RingBuffer | None:
        """Return `ring` while stem recording is active, else `None` for the audio callback."""
        return self._ring if self._is_recording else None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def dir_path(self) -> Path | None:
        return self._dir_path

    @property
    def file_path(self) -> Path | None:
        return self._dir_path

    @property
    def stem_paths(self) -> dict[str, Path]:
        return dict(self._stem_paths)

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def dropped_frames(self) -> int:
        return self._ring.dropped_frames

    def stats(self) -> RecorderStats:
        return RecorderStats(
            is_recording=self._is_recording,
            file_path=self._dir_path,
            frames_written=self._frames_written,
            dropped_frames=self._ring.dropped_frames,
            overrun_events=self._ring.overrun_count,
        )

    def open_files(self, now: datetime | None = None) -> Path:
        """Create the stem directory and open all 4 WAV files without starting the worker thread."""
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        stem_dir = resolve_stem_dir(self._config.output_dir, now=now)
        stem_dir.mkdir(parents=True, exist_ok=True)
        while self._ring.available_frames() > 0:
            self._ring.read(self._read_scratch)
        self._ring.dropped_frames = 0
        self._ring.overrun_count = 0

        opened_files: dict[str, sf.SoundFile] = {}
        paths: dict[str, Path] = {}
        try:
            for key, filename in STEM_FILENAMES.items():
                path = stem_dir / filename
                paths[key] = path
                opened_files[key] = sf.SoundFile(
                    path,
                    mode="w",
                    samplerate=self._sample_rate_hz,
                    channels=STEM_CHANNELS[key],
                    subtype="FLOAT",
                )
        except Exception:
            for handle in opened_files.values():
                handle.close()
            raise

        self._sound_files = opened_files
        self._stem_paths = paths
        self._dir_path = stem_dir
        self._frames_written = 0
        self._is_recording = True
        return stem_dir

    def start(self, now: datetime | None = None) -> Path:
        """Open the 4 stem WAV files and start the writer thread (`jammate-stem-recorder`)."""
        if self._is_recording:
            assert self._dir_path is not None
            return self._dir_path
        stem_dir = self.open_files(now=now)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="jammate-stem-recorder",
            daemon=False,
        )
        self._thread.start()
        logger.info("Recording stems to %s", stem_dir)
        return stem_dir

    def drain_available(self) -> int:
        """Drain all currently available 7-channel frames into the 4 open stem WAV files."""
        with self._io_lock:
            if not self._sound_files:
                return 0
            bass_sf = self._sound_files["bass"]
            drums_sf = self._sound_files["drums"]
            ai_sf = self._sound_files["ai_guitar"]
            master_sf = self._sound_files["master"]
            written_now = 0
            while True:
                avail = self._ring.available_frames()
                if avail <= 0:
                    break
                to_read = min(avail, WRITER_CHUNK_FRAMES)
                buf = self._read_scratch[:to_read]
                result = self._ring.read(buf)
                n = result.frame_count
                if n <= 0:
                    break
                chunk = buf[:n]
                bass_sf.write(chunk[:, 0:1])
                drums_sf.write(chunk[:, 1:3])
                ai_sf.write(chunk[:, 3:5])
                master_sf.write(chunk[:, 5:7])
                self._frames_written += n
                written_now += n
            return written_now

    def stop(self) -> Path | None:
        """Stop the writer thread, flush remaining ring frames, and close all 4 stem WAV files."""
        if not self._is_recording and not self._sound_files:
            return self._dir_path
        self._is_recording = False
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join()

        self.drain_available()
        with self._io_lock:
            handles = list(self._sound_files.values())
            self._sound_files.clear()
            for handle in handles:
                handle.flush()
                handle.close()
        if self._dir_path is not None:
            logger.info(
                "Stem recording stopped: %s (%d frames written, %d dropped).",
                self._dir_path,
                self._frames_written,
                self._ring.dropped_frames,
            )
        return self._dir_path

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._ring.available_frames() > 0:
                    self.drain_available()
                else:
                    self._stop_event.wait(WRITER_POLL_SECONDS)
        except Exception as exc:
            logger.error("StemRecorder writer thread failed: %s", exc)
            self._is_recording = False

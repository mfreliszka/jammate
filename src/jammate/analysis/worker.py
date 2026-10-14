"""The analysis worker: the thread between the input ring and the rest of JamMate (§22, §23).

`AnalysisWorker` is the input ring's only consumer. It cuts the input into hops; on each hop it
runs pitch detection, note stabilization, onset detection, the playing state, intensity and
tempo, and hands the resulting events to `on_event` on its own thread. It never blocks the audio
callback: it only reads the ring.

`process_available()` does one pass over the unread input without any thread, which is how
tests and `analyze_offline` drive it.

`AnalysisStats` holds diagnostics only. What the analysis currently believes (note, tempo,
intensity, playing state) reaches the rest of JamMate as events, folded into the Musical State
(`music/state.py`, §38).
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jammate.analysis.dynamics import IntensityTracker, PlayingStateTracker
from jammate.analysis.onset import OnsetDetector
from jammate.analysis.pitch import PitchDetector, PitchFrame
from jammate.analysis.stabilizer import NoteStabilizer
from jammate.analysis.tempo import TempoTracker
from jammate.audio.levels import CLIP_THRESHOLD_DBFS, SILENCE_FLOOR_DBFS
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AnalysisConfig, Instrument
from jammate.music.events import (
    MusicalEvent,
    SampleFrame,
    seconds_to_frames,
)

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 2.0
# Wake often. On Apple Silicon a thread that sleeps ~5 ms between short bursts runs 5-10x slower
# after each wake, and while slow it delays the audio callback, which needs the GIL. Measured on
# 2026-10-09 (Step 2.2): waking every 5 ms gave callback gaps of p99.9 6.1 ms; every 1 ms, 3.9 ms
# (nominal 2.67 ms at 128 frames).
POLL_SECONDS = 0.001
PEAK_METER_SECONDS = 0.5  # `AnalysisStats.recent_peak_dbfs` covers this much input


@dataclass(frozen=True)
class AnalysisStats:
    hops_processed: int
    dropped_frames: int  # input frames never analyzed (ring overruns and deliberate skips)
    skips: int  # times the worker fell behind and jumped to the latest audio
    errors: int
    hop_mean_seconds: float
    hop_max_seconds: float
    recent_peak_dbfs: float  # loudest input peak in the last PEAK_METER_SECONDS
    max_peak_dbfs: float  # loudest input peak so far
    clipped_hops: int  # hops whose peak reached CLIP_THRESHOLD_DBFS
    latest: PitchFrame | None  # the most recent hop's estimate
    onsets: int


@dataclass(frozen=True)
class EmittedEvent:
    """An event from `analyze_offline`, with the stream position at which it was emitted.
    `emitted_frame - event.frame_index` is the decision latency."""

    event: MusicalEvent
    emitted_frame: SampleFrame


class AnalysisWorker:
    def __init__(
        self,
        ring: RingBuffer,
        config: AnalysisConfig,
        sample_rate_hz: int,
        *,
        on_event: Callable[[MusicalEvent], None] | None = None,
        on_pitch_frame: Callable[[PitchFrame], None] | None = None,
    ) -> None:
        """`on_event` and `on_pitch_frame` (a debug hook, every hop) run on the worker thread."""
        self._ring = ring
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._hop_frames = config.hop_frames
        self._detector = PitchDetector(config, sample_rate_hz)
        self._stabilizer = NoteStabilizer(config, sample_rate_hz)
        self._onsets = OnsetDetector(config, sample_rate_hz)
        self._intensity = IntensityTracker(config, sample_rate_hz)
        self._playing = PlayingStateTracker(config, sample_rate_hz)
        self._tempo = TempoTracker(config, sample_rate_hz)
        self._on_event = on_event or (lambda event: None)
        self._on_pitch_frame = on_pitch_frame
        self._max_backlog_frames = max(
            seconds_to_frames(config.max_backlog_seconds, sample_rate_hz), self._hop_frames
        )
        self._hop_buffer = np.zeros((self._hop_frames, ring.channels), dtype=np.float32)
        self._hop_fill = 0
        self._next_frame = ring.read_position
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        # Written only by the thread that processes hops; read by others.
        self._hops_processed = 0
        self._skips = 0
        self._errors = 0
        self._hop_total_ns = 0
        self._hop_max_ns = 0
        self._max_peak_dbfs = SILENCE_FLOOR_DBFS
        # Bounded history of hop peaks; other threads take its max without coordination.
        self._recent_peaks_dbfs = np.full(
            math.ceil(PEAK_METER_SECONDS * sample_rate_hz / self._hop_frames), SILENCE_FLOOR_DBFS
        )
        self._clipped_hops = 0
        self._onset_count = 0
        self._latest: PitchFrame | None = None
        self.last_error: BaseException | None = None

    @property
    def instrument(self) -> Instrument:
        return self._config.instrument

    @property
    def window_frames(self) -> int:
        return self._detector.window_frames

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_instrument(self, instrument: Instrument) -> None:
        """Switch the active instrument (`guitar` or `bass`) and rebuild the pitch detector."""
        if instrument == self._config.instrument:
            return
        new_config = self._config.model_copy(update={"instrument": instrument})
        new_detector = PitchDetector(new_config, self._sample_rate_hz)
        self._config = new_config
        self._detector = new_detector
        self._stabilizer.reset()

    def set_hop_listener(self, listener: Callable[[PitchFrame], None] | None) -> None:
        """Set or replace the per-hop callback (called on the worker thread after hop events)."""
        self._on_pitch_frame = listener

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("AnalysisWorker can only be started once")
        self._thread = threading.Thread(target=self._run, name="jammate-analysis")
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish its current pass and wait for it."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                logger.error(
                    "The analysis worker did not stop within %.1f s.", STOP_TIMEOUT_SECONDS
                )

    def stats(self) -> AnalysisStats:
        hops = self._hops_processed
        return AnalysisStats(
            hops_processed=hops,
            dropped_frames=self._ring.dropped_frames,
            skips=self._skips,
            errors=self._errors,
            hop_mean_seconds=self._hop_total_ns / hops / 1e9 if hops else 0.0,
            hop_max_seconds=self._hop_max_ns / 1e9,
            recent_peak_dbfs=float(np.max(self._recent_peaks_dbfs)),
            max_peak_dbfs=self._max_peak_dbfs,
            clipped_hops=self._clipped_hops,
            latest=self._latest,
            onsets=self._onset_count,
        )

    def process_available(self) -> int:
        """Analyze every complete hop of unread input. Returns the number of hops analyzed."""
        if self._ring.available_frames() > self._max_backlog_frames:
            # Fell behind: drop the backlog and continue from now (§23). The ring counts the
            # skipped frames as dropped.
            skipped_to = self._ring.read_latest(self._hop_buffer[:0])
            self._restart_at(skipped_to.start_frame)
            self._skips += 1

        hops = 0
        while True:
            read = self._ring.read(self._hop_buffer[self._hop_fill :])
            if read.frame_count == 0:
                return hops
            if read.start_frame != self._next_frame:
                # The ring overran: the partial hop before the gap is useless.
                fill = self._hop_fill
                self._hop_buffer[: read.frame_count] = self._hop_buffer[
                    fill : fill + read.frame_count
                ]
                self._restart_at(read.start_frame)
            self._hop_fill += read.frame_count
            self._next_frame = read.start_frame + read.frame_count
            if self._hop_fill == self._hop_frames:
                self._analyze_hop(self._next_frame)
                self._hop_fill = 0
                hops += 1

    def _restart_at(self, frame_index: SampleFrame) -> None:
        self._hop_fill = 0
        self._next_frame = frame_index
        self._stabilizer.reset()

    def _analyze_hop(self, end_frame: SampleFrame) -> None:
        started_ns = time.perf_counter_ns()
        hop = self._hop_buffer[:, 0]
        frame = self._detector.process_hop(hop, end_frame)
        onset = self._onsets.process_hop(hop, frame)
        self._latest = frame
        self._max_peak_dbfs = max(self._max_peak_dbfs, frame.peak_dbfs)
        self._recent_peaks_dbfs[self._hops_processed % len(self._recent_peaks_dbfs)] = (
            frame.peak_dbfs
        )
        if frame.peak_dbfs >= CLIP_THRESHOLD_DBFS:
            self._clipped_hops += 1
        note_event = self._stabilizer.update(frame)
        playing_event = self._playing.update(frame, onset, self._stabilizer.current_note)
        intensity_event = self._intensity.update(frame, onset, self._playing.state.user_is_playing)
        tempo_event = self._tempo.update(onset, frame.frame_index)
        if onset is not None:
            self._onset_count += 1
        for event in (onset, note_event, playing_event, intensity_event, tempo_event):
            if event is not None:
                self._on_event(event)
        if self._on_pitch_frame is not None:
            self._on_pitch_frame(frame)
        elapsed_ns = time.perf_counter_ns() - started_ns
        self._hops_processed += 1
        self._hop_total_ns += elapsed_ns
        self._hop_max_ns = max(self._hop_max_ns, elapsed_ns)

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            try:
                self.process_available()
            except Exception as exc:
                self._errors += 1
                self.last_error = exc
                if self._errors == 1:
                    logger.exception("Analysis worker error (later errors are only counted).")
            self._stop_requested.wait(POLL_SECONDS)


def analyze_offline(
    samples: np.ndarray,
    config: AnalysisConfig,
    sample_rate_hz: int,
    *,
    block_frames: int = 128,
    on_pitch_frame: Callable[[PitchFrame], None] | None = None,
) -> tuple[list[EmittedEvent], AnalysisStats]:
    """Run the live analysis path over mono `samples`: each block goes through a ring buffer to
    an `AnalysisWorker`, exactly as the audio callback would deliver it, without threads."""
    ring = RingBuffer(max(block_frames, config.hop_frames) * 4)
    emitted: list[EmittedEvent] = []
    worker = AnalysisWorker(
        ring,
        config,
        sample_rate_hz,
        on_event=lambda event: emitted.append(EmittedEvent(event, ring.frames_written)),
        on_pitch_frame=on_pitch_frame,
    )
    block = np.zeros((block_frames, 1), dtype=np.float32)
    for start in range(0, len(samples), block_frames):
        chunk = samples[start : start + block_frames]
        block[: len(chunk), 0] = chunk
        ring.write(block[: len(chunk)])
        worker.process_available()
    return emitted, worker.stats()

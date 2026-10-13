"""JamSession: the owner of every component's lifecycle (D4, §83, §84).

It wires the parts together:

    audio callback ─input ring─► AnalysisWorker ─events─► MusicalStateManager ─► snapshots
                                                      └─► on_event (the headless timeline)

Startup (§83): the constructor resolves the devices and builds every component, so nothing
expensive happens once audio flows. `start()` starts the analysis worker, then opens and starts
the stream, so the worker is waiting for the first block.

Shutdown (§84): `stop()` stops the analysis worker, then closes the stream, which releases the
devices. Later steps slot in around them. Drums (Phase 3) and the AI (Phase 4) start after
analysis and stop before it, and recording flushes after the stream.

A session runs once; build a new one for the next jam. Call `start()` and `stop()` from one
thread (the main or GUI thread). `state` and the stats can be read from any thread.
"""

import logging
from collections.abc import Callable
from types import TracebackType
from typing import Self

from jammate.analysis.worker import AnalysisWorker
from jammate.audio.engine import AudioEngine, StreamInfo
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AnalysisConfig, AudioConfig, JamMateConfig
from jammate.music.events import MusicalEvent
from jammate.music.state import MusicalState, MusicalStateManager

logger = logging.getLogger(__name__)

type EngineFactory = Callable[[AudioConfig], AudioEngine]
type WorkerFactory = Callable[
    [RingBuffer, AnalysisConfig, int, Callable[[MusicalEvent], None]], AnalysisWorker
]


def default_worker_factory(
    ring: RingBuffer,
    config: AnalysisConfig,
    sample_rate_hz: int,
    on_event: Callable[[MusicalEvent], None],
) -> AnalysisWorker:
    return AnalysisWorker(ring, config, sample_rate_hz, on_event=on_event)


class JamSession:
    def __init__(
        self,
        config: JamMateConfig,
        *,
        on_event: Callable[[MusicalEvent], None] | None = None,
        engine_factory: EngineFactory = AudioEngine.from_config,
        worker_factory: WorkerFactory = default_worker_factory,
    ) -> None:
        """Resolve the devices and build every component; nothing runs yet. `on_event` sees each
        analysis event after the Musical State has taken it in. It runs on the analysis thread,
        so hand events on through something bounded and lock-free. Raises
        `DeviceResolutionError` or `ValueError` when the configured devices can't be used."""
        self.config = config
        self._on_event = on_event
        self._state = MusicalStateManager(config.state)
        self.engine = engine_factory(config.audio)
        sample_rate_hz = config.audio.sample_rate_hz
        self.worker = worker_factory(
            self.engine.input_ring, config.analysis, sample_rate_hz, self._handle_event
        )
        self._started = False
        self._stopped = False
        _log_analysis_setup(config.analysis, self.worker, sample_rate_hz)

    @property
    def state(self) -> MusicalState:
        """The current Musical State snapshot. Never blocks."""
        return self._state.snapshot

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped and self.engine.is_running

    def start(self) -> StreamInfo:
        """Start analysis, then the audio stream. If the stream fails, everything is shut down
        again before the error propagates."""
        if self._started:
            raise RuntimeError("A JamSession runs once; build a new one for the next jam.")
        self._started = True
        self.worker.start()
        try:
            return self.engine.start()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Shut down in §84 order. Safe to call more than once, and without `start()`."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self.worker.stop()
        finally:
            self.engine.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def _handle_event(self, event: MusicalEvent) -> None:
        """Runs on the analysis worker thread, the Musical State's only writer."""
        self._state.apply(event)
        if self._on_event is not None:
            self._on_event(event)


def _log_analysis_setup(
    config: AnalysisConfig, worker: AnalysisWorker, sample_rate_hz: int
) -> None:
    min_hz, max_hz = config.pitch_range_hz
    logger.info(
        "Analysis: %s, %.0f-%.0f Hz, window %d frames (%.1f ms), hop %d frames (%.1f ms).",
        config.instrument,
        min_hz,
        max_hz,
        worker.window_frames,
        1000 * worker.window_frames / sample_rate_hz,
        config.hop_frames,
        1000 * config.hop_frames / sample_rate_hz,
    )

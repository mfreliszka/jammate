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
from pathlib import Path
from types import TracebackType
from typing import Self

from jammate.ai.base import AccompanimentEngine, GuardedAccompaniment, NoteTracker
from jammate.ai.mrt import create_accompaniment_engine
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisWorker
from jammate.audio.engine import AudioEngine, StreamInfo
from jammate.audio.recorder import MasterRecorder
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import (
    AccompanimentConfig,
    AnalysisConfig,
    AudioConfig,
    Instrument,
    JamMateConfig,
    StyleFamily,
)
from jammate.drums.engine import DrumEngine
from jammate.drums.sampler import DrumSampler
from jammate.music.beat_clock import BeatClock
from jammate.music.events import MusicalEvent
from jammate.music.state import MusicalState, MusicalStateManager
from jammate.music.styles import AiEntryController, get_style_preset, map_follow_improvise

logger = logging.getLogger(__name__)

type EngineFactory = Callable[[AudioConfig], AudioEngine]
type WorkerFactory = Callable[
    [RingBuffer, AnalysisConfig, int, Callable[[MusicalEvent], None]], AnalysisWorker
]
type AccompanimentFactory = Callable[[AccompanimentConfig, int], AccompanimentEngine]


def default_worker_factory(
    ring: RingBuffer,
    config: AnalysisConfig,
    sample_rate_hz: int,
    on_event: Callable[[MusicalEvent], None],
) -> AnalysisWorker:
    return AnalysisWorker(ring, config, sample_rate_hz, on_event=on_event)


def default_accompaniment_factory(
    config: AccompanimentConfig,
    sample_rate_hz: int,
) -> AccompanimentEngine:
    return create_accompaniment_engine(config, sample_rate_hz)


class JamSession:
    def __init__(
        self,
        config: JamMateConfig,
        *,
        on_event: Callable[[MusicalEvent], None] | None = None,
        engine_factory: EngineFactory = AudioEngine.from_config,
        worker_factory: WorkerFactory = default_worker_factory,
        accompaniment_factory: AccompanimentFactory = default_accompaniment_factory,
    ) -> None:
        """Resolve the devices and build every component; nothing runs yet. `on_event` sees each
        analysis event after the Musical State has taken it in. It runs on the analysis thread,
        so hand events on through something bounded and lock-free. Raises
        `DeviceResolutionError` or `ValueError` when the configured devices can't be used."""
        self.config = config
        self._on_event = on_event
        manual_bpm = config.analysis.tempo.manual_bpm
        self._state = MusicalStateManager(config.state, manual_bpm=manual_bpm)
        sample_rate_hz = config.audio.sample_rate_hz
        self.drum_sampler = DrumSampler.from_config(config.drums, sample_rate_hz)
        raw_ai = accompaniment_factory(config.ai, sample_rate_hz)
        self.ai = GuardedAccompaniment(raw_ai)
        self.note_tracker = NoteTracker(self.ai)
        self.ai_entry = AiEntryController(
            config.ai,
            beats_per_bar=config.beat_clock.beats_per_bar,
        )
        self.recorder = MasterRecorder(config.recording, sample_rate_hz)
        self.engine = engine_factory(config.audio)
        self.engine.processor.drum_sampler = self.drum_sampler
        self.engine.processor.ai_ring = self.ai.output_ring
        self.engine.processor.record_ring = None
        self.beat_clock = BeatClock(config.beat_clock, sample_rate_hz, manual_bpm=manual_bpm)
        self.drums = DrumEngine(
            config.drums,
            sample_rate_hz,
            self.drum_sampler.event_ring,
            self.beat_clock,
            lambda: self.state,
            lambda: self.engine.processor.frames_processed,
            manual_bpm_active=manual_bpm is not None,
        )
        self._follow_improvise: float = config.ai.follow_improvise
        self.worker = worker_factory(
            self.engine.input_ring, config.analysis, sample_rate_hz, self._handle_event
        )
        self.worker.set_hop_listener(self._handle_hop)
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

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    @property
    def style(self) -> StyleFamily:
        return self.drums.style

    @property
    def instrument(self) -> Instrument:
        return self.worker.instrument

    @property
    def follow_improvise(self) -> float:
        return self._follow_improvise

    def set_instrument(self, instrument: Instrument) -> None:
        """Switch the active input instrument (`guitar` or `bass`) (Step 5.1, D12)."""
        self.worker.set_instrument(instrument)
        self.config = self.config.model_copy(
            update={"analysis": self.config.analysis.model_copy(update={"instrument": instrument})}
        )

    def set_style(self, style: StyleFamily) -> None:
        """Switch the active style preset for both drums and AI accompaniment (Step 4.3, §75)."""
        preset = get_style_preset(style)
        self.drums.set_style(preset.drum_family)
        mapping = map_follow_improvise(preset, self._follow_improvise, self.state.intensity)
        self.drums.controller.set_fill_every_bars(mapping.fill_every_bars)
        self.ai.set_style(preset.name)

    def set_follow_improvise(self, follow_improvise: float) -> None:
        """Update Follow-Improvise (`0.0..1.0`) for both AI and drum fill variation (§74)."""
        preset = get_style_preset(self.drums.style)
        mapping = map_follow_improvise(preset, follow_improvise, self.state.intensity)
        self._follow_improvise = mapping.follow_improvise
        self.drums.controller.set_fill_every_bars(mapping.fill_every_bars)
        self.ai.set_follow_improvise(mapping.follow_improvise)

    def start_recording(self) -> Path:
        """Start streaming the master stereo output to a new WAV file (§59)."""
        path = self.recorder.start()
        self.engine.processor.record_ring = self.recorder.active_ring
        return path

    def stop_recording(self) -> Path | None:
        """Detach the recording ring from the callback, flush buffered samples, and close."""
        self.engine.processor.record_ring = None
        return self.recorder.stop()

    def start(self) -> StreamInfo:
        """Start analysis, drums, AI and optional recorder, then the audio stream (§83)."""
        if self._started:
            raise RuntimeError("A JamSession runs once; build a new one for the next jam.")
        self._started = True
        self.worker.start()
        self.drums.start()
        self.ai.start()
        self.engine.processor.ai_ring = self.ai.output_ring
        if self.config.recording.enabled:
            self.start_recording()
        try:
            return self.engine.start()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Shut down in §84 order (all notes off, AI, drums, analysis, stream, flush recording)."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self.note_tracker.all_notes_off()
        finally:
            try:
                self.ai.stop()
            finally:
                try:
                    self.drums.stop()
                finally:
                    try:
                        self.worker.stop()
                    finally:
                        try:
                            self.engine.close()
                        finally:
                            self.stop_recording()

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
        self.beat_clock.handle_event(event)
        self._state.apply(event)
        if self.beat_clock.is_running or self._state.snapshot.beat is not None:
            self._state.update_beat(
                event.frame_index, self.beat_clock.position_at(event.frame_index)
            )
        self.note_tracker.handle_event(event)
        self._sync_ai_entry()
        self.engine.processor.ai_ring = self.ai.output_ring
        if self._on_event is not None:
            self._on_event(event)

    def _handle_hop(self, frame: PitchFrame) -> None:
        """Runs on the analysis worker thread after each hop to keep beat/bar/phase fresh."""
        if self.beat_clock.is_running or self._state.snapshot.beat is not None:
            self._state.update_beat(
                frame.frame_index, self.beat_clock.position_at(frame.frame_index)
            )
        self._sync_ai_entry()

    def _sync_ai_entry(self) -> None:
        self.ai_entry.update(
            self._state.snapshot,
            self.drums.drummer_state,
            self.drums.controller.entry_bar,
        )
        set_gate = getattr(self.ai.inner, "set_gate_gain", None)
        if callable(set_gate):
            set_gate(self.ai_entry.gate_gain)


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

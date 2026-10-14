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
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Self

from jammate.ai.base import AccompanimentEngine, GuardedAccompaniment, NoteTracker
from jammate.ai.mrt import create_accompaniment_engine
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisWorker
from jammate.audio.engine import AudioEngine, StreamInfo
from jammate.audio.recorder import MasterRecorder, StemRecorder
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import (
    AccompanimentConfig,
    AnalysisConfig,
    AudioConfig,
    Instrument,
    JamMateConfig,
    MeterName,
    StyleFamily,
)
from jammate.drums.engine import DrumEngine
from jammate.drums.sampler import DrumSampler
from jammate.midi.input import MidiInputWorker, MidiPort
from jammate.music.beat_clock import BeatClock
from jammate.music.director import BandDirective, BandDirector
from jammate.music.events import MusicalEvent, SampleFrame
from jammate.music.overrides import (
    ManualOverrides,
    MeterTuple,
    TapTempoTracker,
    format_meter,
    meter_beats_per_bar,
    meter_steps_per_bar,
    parse_meter,
    validate_manual_key,
)
from jammate.music.riff import RiffDetector
from jammate.music.state import MusicalState, MusicalStateManager, SectionType
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
        midi_port: MidiPort | None = None,
    ) -> None:
        """Resolve the devices and build every component; nothing runs yet. `on_event` sees each
        analysis event after the Musical State has taken it in. It runs on the analysis thread,
        so hand events on through something bounded and lock-free. Raises
        `DeviceResolutionError` or `ValueError` when the configured devices can't be used."""
        self.config = config
        self._on_event = on_event
        self._event_lock = threading.Lock()
        manual_bpm = config.analysis.tempo.manual_bpm
        manual_key = validate_manual_key(config.analysis.key.manual_key)
        meter_tuple = parse_meter(config.beat_clock.meter)
        beats_per_bar = meter_beats_per_bar(meter_tuple)
        self.tap_tracker = TapTempoTracker(
            min_bpm=config.analysis.tempo.min_bpm,
            max_bpm=config.analysis.tempo.max_bpm,
        )
        self._overrides = ManualOverrides(
            bpm=float(manual_bpm) if manual_bpm is not None else None,
            meter=meter_tuple,
            key=manual_key,
        )
        self._state = MusicalStateManager(
            config.state,
            manual_bpm=manual_bpm,
            manual_key=manual_key,
        )
        self._state.set_instrument(config.analysis.instrument)
        sample_rate_hz = config.audio.sample_rate_hz
        self.drum_sampler = DrumSampler.from_config(config.drums, sample_rate_hz)
        raw_ai = accompaniment_factory(config.ai, sample_rate_hz)
        self.ai = GuardedAccompaniment(raw_ai)
        self.note_tracker = NoteTracker(self.ai)
        self.ai_entry = AiEntryController(
            config.ai,
            beats_per_bar=beats_per_bar,
        )
        self.recorder = MasterRecorder(config.recording, sample_rate_hz)
        self.stem_recorder = StemRecorder(config.recording, sample_rate_hz)
        self._record_stems: bool = config.recording.record_stems
        self.engine = engine_factory(config.audio)
        self.engine.processor.drum_sampler = self.drum_sampler
        self.engine.processor.ai_ring = self.ai.output_ring
        self.engine.processor.record_ring = None
        self.engine.processor.stem_record_ring = None
        self.beat_clock = BeatClock(config.beat_clock, sample_rate_hz, manual_bpm=manual_bpm)
        self.riff_detector = RiffDetector(config.riff)
        self.riff_detector.set_steps_per_bar(meter_steps_per_bar(meter_tuple))
        self.director = BandDirector(fill_every_bars=config.drums.fill_every_bars)
        self.drums = DrumEngine(
            config.drums,
            sample_rate_hz,
            self.drum_sampler.event_ring,
            self.beat_clock,
            lambda: self.state,
            lambda: self.engine.processor.frames_processed,
            beats_per_bar=beats_per_bar,
            manual_bpm_active=manual_bpm is not None,
        )
        self._follow_improvise: float = config.ai.follow_improvise
        self.worker = worker_factory(
            self.engine.input_ring, config.analysis, sample_rate_hz, self._handle_event
        )
        self.worker.set_hop_listener(self._handle_hop)
        self.midi_worker = MidiInputWorker(
            config.analysis,
            sample_rate_hz,
            port_name=config.midi.input_device_name,
            port=midi_port,
            frame_source=lambda: self.engine.processor.frames_processed,
            on_event=self._handle_event,
        )
        if manual_key is not None:
            set_worker_key = getattr(self.worker, "set_manual_key", None)
            if callable(set_worker_key):
                set_worker_key(manual_key)
            self.midi_worker.set_manual_key(manual_key)
            est = getattr(self.worker, "key_estimate", None)
            if est is not None and est.root_pc is not None:
                self.note_tracker.set_harmonic_anchor(est.root_pc, user_is_playing=False)
        self._started = False
        self._stopped = False
        _log_analysis_setup(config.analysis, self.worker, sample_rate_hz)

    @property
    def state(self) -> MusicalState:
        """The current Musical State snapshot. Never blocks."""
        return self._state.snapshot

    @property
    def directive(self) -> BandDirective:
        """The current BandDirector decision snapshot."""
        return self.director.directive

    @property
    def overrides(self) -> ManualOverrides:
        """Current manual overrides for BPM, meter, and key (Step 6.1, §31, §149)."""
        return self._overrides

    @property
    def meter(self) -> MeterName:
        """Current active time-signature meter (`'4/4'`, `'3/4'`, or `'6/8'`)."""
        return format_meter(self._overrides.meter)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped and self.engine.is_running

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording or self.stem_recorder.is_recording

    @property
    def record_stems(self) -> bool:
        return self._record_stems

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
        self.midi_worker.set_instrument(instrument)
        self._state.set_instrument(instrument)
        self.config = self.config.model_copy(
            update={"analysis": self.config.analysis.model_copy(update={"instrument": instrument})}
        )

    def set_manual_bpm(self, bpm: float | None) -> float | None:
        """Set or clear a manual BPM override (Step 6.1, §31, §149)."""
        target_bpm = float(bpm) if bpm is not None else None
        set_worker_bpm = getattr(self.worker, "set_manual_bpm", None)
        if callable(set_worker_bpm):
            set_worker_bpm(target_bpm)
        self.midi_worker.set_manual_bpm(target_bpm)
        frame = self.engine.processor.frames_processed
        self.beat_clock.set_manual_bpm(target_bpm, anchor_frame=frame)
        self.drums.set_manual_bpm_active(target_bpm is not None)
        self._state.set_manual_bpm(target_bpm, position=self.beat_clock.position_at(frame))
        self._overrides = replace(self._overrides, bpm=target_bpm)
        return self._overrides.bpm

    def tap_tempo(self, timestamp_seconds: float | None = None) -> float | None:
        """Record a Tap Tempo press and apply the resulting manual BPM when >= 2 taps align."""
        bpm = self.tap_tracker.tap(timestamp_seconds)
        if bpm is not None:
            self.set_manual_bpm(bpm)
        return bpm

    def set_meter(self, meter: MeterName | MeterTuple) -> MeterName:
        """Switch the active meter (`'4/4'`, `'3/4'`, or `'6/8'`) (Step 6.1, §31, §149)."""
        meter_tuple = parse_meter(meter)
        meter_name = format_meter(meter_tuple)
        beats = meter_beats_per_bar(meter_tuple)
        steps = meter_steps_per_bar(meter_tuple)
        frame = self.engine.processor.frames_processed
        self.beat_clock.set_beats_per_bar(beats, anchor_frame=frame)
        self.drums.set_beats_per_bar(beats)
        self.ai_entry.set_beats_per_bar(beats)
        self.riff_detector.set_steps_per_bar(steps)
        if self.beat_clock.is_running:
            self._state.set_manual_bpm(
                self._overrides.bpm,
                position=self.beat_clock.position_at(frame),
            )
            if self._overrides.bpm is None and self.beat_clock.bpm is not None:
                # Preserve auto-locked BPM when switching meter mid-jam
                pos = self.beat_clock.position_at(frame)
                snap = self._state.snapshot
                self._state._snapshot = replace(
                    snap,
                    bpm=self.beat_clock.bpm,
                    tempo_locked=True,
                    tempo_confidence=max(snap.tempo_confidence, 0.85),
                    bar=pos.bar if pos is not None else snap.bar,
                    beat=pos.beat if pos is not None else snap.beat,
                    beat_phase=pos.beat_phase if pos is not None else snap.beat_phase,
                )
        self._overrides = replace(self._overrides, meter=meter_tuple)
        self.config = self.config.model_copy(
            update={
                "beat_clock": self.config.beat_clock.model_copy(
                    update={"meter": meter_name, "beats_per_bar": beats}
                )
            }
        )
        return meter_name

    def set_manual_key(self, key: str | None) -> str | None:
        """Set or clear a manual key override (e.g. `'E minor'`, `'C major'`, or `None`)."""
        normalized = validate_manual_key(key)
        set_worker_key = getattr(self.worker, "set_manual_key", None)
        if callable(set_worker_key):
            set_worker_key(normalized)
        self.midi_worker.set_manual_key(normalized)
        self._state.set_manual_key(normalized)
        self._overrides = replace(self._overrides, key=normalized)
        est = getattr(self.worker, "key_estimate", None)
        if est is not None:
            self.note_tracker.set_harmonic_anchor(
                est.root_pc,
                user_is_playing=self.state.user_is_playing,
            )
        return normalized

    def reset_musical_context(self) -> None:
        """Reset tempo search, BeatClock, key histograms, RiffDetector, SessionMemory, and
        BandDirector without stopping the audio stream (Step 6.1, §31, §149)."""
        self.tap_tracker.reset()
        reset_worker = getattr(self.worker, "reset_context", None)
        if callable(reset_worker):
            reset_worker()
        self.midi_worker.reset_context()
        frame = self.engine.processor.frames_processed
        self.beat_clock.reset_context(anchor_frame=frame)
        self.riff_detector.reset()
        self.director.reset()
        self.drums.reset_context()
        self.ai_entry.reset()
        self.note_tracker.all_notes_off()
        self._state.reset(position=self.beat_clock.position_at(frame))

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

    def set_record_stems(self, record_stems: bool) -> None:
        """Select whether `start_recording()` writes 4 multi-track stems or a single master WAV."""
        self._record_stems = bool(record_stems)
        self.config = self.config.model_copy(
            update={
                "recording": self.config.recording.model_copy(
                    update={"record_stems": self._record_stems}
                )
            }
        )

    def start_recording(self, *, stems: bool | None = None) -> Path:
        """Start streaming the master stereo output or 4 multi-track stems to disk (§59, §150)."""
        use_stems = self._record_stems if stems is None else bool(stems)
        if use_stems:
            self.engine.processor.record_ring = None
            path = self.stem_recorder.start()
            self.engine.processor.stem_record_ring = self.stem_recorder.active_ring
            return path
        self.engine.processor.stem_record_ring = None
        path = self.recorder.start()
        self.engine.processor.record_ring = self.recorder.active_ring
        return path

    def stop_recording(self) -> Path | None:
        """Detach recording rings from the callback, flush buffered samples, and close."""
        self.engine.processor.record_ring = None
        self.engine.processor.stem_record_ring = None
        stem_path = self.stem_recorder.stop() if self.stem_recorder.is_recording else None
        master_path = self.recorder.stop()
        return stem_path if stem_path is not None else master_path

    def start(self) -> StreamInfo:
        """Start analysis, MIDI, drums, AI and optional recorder, then the audio stream (§83)."""
        if self._started:
            raise RuntimeError("A JamSession runs once; build a new one for the next jam.")
        self._started = True
        self.worker.start()
        self.midi_worker.start()
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
        """Shut down in §84 order (MIDI, all notes off, AI, drums, analysis, stream, recording)."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self.midi_worker.stop()
        finally:
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
        """Runs on the analysis or MIDI worker thread, serialized by `_event_lock`."""
        with self._event_lock:
            self.beat_clock.handle_event(event)
            self.drums.handle_event(event)
            self.riff_detector.handle_event(event)
            self._state.apply(event)
            if self.beat_clock.is_running or self._state.snapshot.beat is not None:
                self._state.update_beat(
                    event.frame_index, self.beat_clock.position_at(event.frame_index)
                )
            self._sync_riff(event.frame_index)
            self.note_tracker.handle_event(event)
            self._sync_ai_entry()
            self.engine.processor.ai_ring = self.ai.output_ring
        if self._on_event is not None:
            self._on_event(event)

    def _handle_hop(self, frame: PitchFrame) -> None:
        """Runs on the analysis worker thread after each hop to keep beat/bar/phase fresh."""
        with self._event_lock:
            if self.beat_clock.is_running or self._state.snapshot.beat is not None:
                self._state.update_beat(
                    frame.frame_index, self.beat_clock.position_at(frame.frame_index)
                )
            self._sync_riff(frame.frame_index)
            self._sync_ai_entry()

    def _sync_riff(self, frame_index: SampleFrame) -> None:
        snap = self._state.snapshot
        pat_id: str | None = None
        if snap.bar is not None and snap.bar > 1:
            pat = self.drums.controller.pattern_for_bar(snap.bar - 1)
            if pat is not None:
                pat_id = pat.name
        result = self.riff_detector.advance_to(
            frame_index,
            self.beat_clock.snapshot,
            preferred_drum_pattern_id=pat_id,
            preferred_intensity_tier=self.drums.tier,
            estimated_key=snap.key_label,
            bass_register=snap.bass_register,
        )
        if result is not None:
            section: SectionType
            if not snap.tempo_locked or not snap.user_is_playing:
                section = "listening"
            elif result.riff_id is None:
                section = "intro"
            else:
                section = "groove"
            self._state.update_riff(
                frame_index,
                riff_id=result.riff_id,
                riff_iteration=result.riff_iteration,
                riff_changed=result.riff_changed,
                section=section,
            )
        directive = self.director.evaluate(
            self._state.snapshot,
            self.riff_detector.memory,
            preset=get_style_preset(self.drums.style),
            follow_improvise=self._follow_improvise,
        )
        if directive.section != self._state.snapshot.section:
            self._state.update_section(frame_index, directive.section)
        self.drums.apply_directive(directive, self._state.snapshot.bar)
        self.note_tracker.set_harmonic_anchor(
            directive.target_key_pc,
            user_is_playing=self._state.snapshot.user_is_playing,
        )
        apply_ai_directive = getattr(self.ai.inner, "apply_directive", None)
        if callable(apply_ai_directive):
            apply_ai_directive(directive)

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

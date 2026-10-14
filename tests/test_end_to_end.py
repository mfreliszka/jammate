"""End-to-end full-band validation tests (Step 4.6, §§68, 108, 114, 115)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from jammate.ai.mrt import (
    MRT2_CHANNELS,
    MRT2_SAMPLES_PER_FRAME,
    PIANOROLL_WITH_ONSETS,
    Mrt2Accompaniment,
)
from jammate.audio.engine import BlockProcessor
from jammate.audio.recorder import MasterRecorder
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import (
    AccompanimentConfig,
    AnalysisConfig,
    AudioConfig,
    DrumConfig,
    JamMateConfig,
    RecordingConfig,
)
from jammate.drums.engine import DrumEngine, DrummerState
from jammate.drums.sampler import DrumSampler
from jammate.music.beat_clock import BeatClock
from jammate.music.events import (
    IntensityUpdate,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SilenceEvent,
    TempoUpdate,
    frames_to_seconds,
)
from jammate.music.state import MusicalStateManager
from jammate.music.styles import AiEntryController
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, riff


@dataclass(frozen=True, slots=True)
class _FakeWaveform:
    samples: np.ndarray


class _FastFakeMrtSystem:
    """Deterministic fast MRT2 system stub that encodes conditioned pitch into generated stereo."""

    def embed_style(self, text: str) -> np.ndarray:
        return np.array([float(len(text))], dtype=np.float32)

    def generate(
        self,
        *,
        conditioning: dict[str, Any],
        cfg_scales: dict[str, float],
        temperature: float,
        top_k: int,
        frames: int,
        state: Any = None,
    ) -> tuple[_FakeWaveform, int]:
        del cfg_scales, temperature, top_k
        step = 0 if state is None else int(state)
        pianoroll = np.asarray(conditioning[PIANOROLL_WITH_ONSETS.key])
        active = np.flatnonzero(pianoroll > 0)
        amp = 0.15 + (0.002 * float(active[0]) if len(active) > 0 else 0.0)
        samples = np.full(
            (frames * MRT2_SAMPLES_PER_FRAME, MRT2_CHANNELS),
            amp,
            dtype=np.float32,
        )
        return _FakeWaveform(samples=samples), step + frames


def test_end_to_end_all_manual_scenarios_on_synthetic_guitar() -> None:
    """Validate §68 scenarios 1-4 (notes, tempo lock, drum entry, stop/hold/stop/resume)."""
    import importlib.util
    import sys

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_jam.py"
    spec = importlib.util.spec_from_file_location("render_jam", script_path)
    assert spec is not None and spec.loader is not None
    render_jam = importlib.util.module_from_spec(spec)
    sys.modules["render_jam"] = render_jam
    spec.loader.exec_module(render_jam)
    render_jam_offline = render_jam.render_jam_offline

    # Scenarios 1-3: repeating guitar riff locks tempo in ~2-4 s, detects notes, drums + AI enter
    test_riff = riff(
        ["E2", "G2", "A2", "B2"],
        bpm=100.0,
        step_beats=0.5,
        repeats=12,
        lead_in_seconds=0.25,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    ai_cfg = AccompanimentConfig(enabled=True, style="rock", entry_delay_bars=1)
    fake_mrt = Mrt2Accompaniment(
        ai_cfg,
        sample_rate_hz=48_000,
        system_factory=lambda _size: _FastFakeMrtSystem(),
    )
    cfg = JamMateConfig(
        audio=AudioConfig(ai_muted=False),
        analysis=AnalysisConfig(instrument="guitar"),
        drums=DrumConfig(style="rock"),
        ai=ai_cfg,
    )
    stereo, summary = render_jam_offline(
        test_riff.samples,
        cfg,
        accompaniment_engine=fake_mrt,
    )
    assert stereo.shape == (len(test_riff.samples), 2)
    assert summary.notes_detected >= 8
    assert summary.tempo_locked is True
    assert summary.locked_bpm == pytest.approx(100.0, abs=1.5)
    assert summary.first_lock_seconds is not None and 1.5 <= summary.first_lock_seconds <= 4.5
    assert summary.drummer_entered is True
    assert summary.ai_entered is True
    assert summary.ai_entry_bar is not None and summary.entry_bar is not None
    assert summary.ai_entry_bar == summary.entry_bar + 1
    assert summary.ai_frames_generated > 0
    assert summary.ai_underruns == 0
    assert summary.dropped_events == 0


def test_ten_minute_continuous_full_band_jam_with_recording(tmp_path: Path) -> None:
    """Step 4.6 Done when (§68, §114): 10-minute full-band jam (guitar + drums + AI + recorder)."""
    sample_rate_hz = 48_000
    block_frames = 128
    bpm = 120.0  # 1 beat = 0.5 s = 24,000 frames; 1 bar = 2.0 s = 96,000 frames; 10 min = 300 bars
    duration_seconds = 600
    total_frames = sample_rate_hz * duration_seconds
    frames_per_beat = int(sample_rate_hz * 60.0 / bpm)
    frames_per_bar = frames_per_beat * 4

    ai_cfg = AccompanimentConfig(
        enabled=True,
        style="stoner_rock",
        follow_improvise=0.5,
        chunk_frames=4,
        entry_delay_bars=1,
        silence_gate_bars=4.0,
    )
    mrt = Mrt2Accompaniment(
        ai_cfg,
        sample_rate_hz=sample_rate_hz,
        system_factory=lambda _size: _FastFakeMrtSystem(),
    )
    ai_entry = AiEntryController(ai_cfg, beats_per_bar=4)

    drum_cfg = DrumConfig(style="stoner_rock")
    drum_sampler = DrumSampler.from_config(drum_cfg, sample_rate_hz=sample_rate_hz)
    input_ring = RingBuffer(capacity_frames=sample_rate_hz * 2, channels=1)
    recorder = MasterRecorder(
        RecordingConfig(output_dir=tmp_path, ring_capacity_seconds=10.0),
        sample_rate_hz=sample_rate_hz,
    )
    rec_path = recorder.open_file()

    processor = BlockProcessor(
        input_ring=input_ring,
        input_channel_index=0,
        monitor_gain=0.8,
        drums_gain=0.8,
        drum_sampler=drum_sampler,
        ai_gain=0.7,
        ai_muted=False,
        ai_ring=mrt.output_ring,
        record_ring=recorder.active_ring,
    )

    state_manager = MusicalStateManager(JamMateConfig().state, manual_bpm=bpm)
    beat_clock = BeatClock(
        JamMateConfig().beat_clock,
        sample_rate_hz=sample_rate_hz,
        manual_bpm=bpm,
    )
    drums = DrumEngine(
        drum_cfg,
        sample_rate_hz=sample_rate_hz,
        event_ring=drum_sampler.event_ring,
        beat_clock=beat_clock,
        state_provider=lambda: state_manager.snapshot,
        frame_provider=lambda: processor.frames_processed,
        manual_bpm_active=False,
    )

    # Start playing at frame 0 (bar 1, beat 1)
    state_manager.apply(
        PlayingStateUpdate(
            frame_index=0,
            timestamp_seconds=0.0,
            state=PlayingState.PLAYING,
        )
    )
    state_manager.apply(
        TempoUpdate(
            frame_index=0,
            timestamp_seconds=0.0,
            bpm=bpm,
            confidence=0.95,
            locked=True,
        )
    )
    state_manager.update_beat(0, beat_clock.position_at(0))
    drums.schedule_available(current_frame=0)

    riff_notes = (40, 43, 45, 47)  # E2, G2, A2, B2
    in_block = np.full((block_frames, 1), 0.2, dtype=np.float32)
    out_block = np.zeros((block_frames, 2), dtype=np.float32)

    pause_start_bar = 150
    pause_end_bar = 160  # 10 bars of silence mid-jam to exercise holding -> stop -> re-entry
    saw_holding = False
    saw_stopped = False
    saw_ai_active = False
    saw_ai_gated = False
    reentered_after_pause = False

    # Process in 128-frame blocks across the full 10-minute session
    for frame in range(0, total_frames, block_frames):
        # Update beat/bar at every beat boundary
        if frame % frames_per_beat == 0:
            beat_idx = frame // frames_per_beat
            bar_num = (beat_idx // 4) + 1
            ts = frames_to_seconds(frame, sample_rate_hz)
            in_pause = pause_start_bar <= bar_num < pause_end_bar

            if frame == (pause_start_bar - 1) * frames_per_bar:
                state_manager.apply(SilenceEvent(frame_index=frame, timestamp_seconds=ts))
                state_manager.apply(
                    PlayingStateUpdate(
                        frame_index=frame,
                        timestamp_seconds=ts,
                        state=PlayingState.STOPPED,
                    )
                )
                mrt.note_off(riff_notes[(beat_idx - 1) % len(riff_notes)])
            elif frame == (pause_end_bar - 1) * frames_per_bar:
                state_manager.apply(
                    PlayingStateUpdate(
                        frame_index=frame,
                        timestamp_seconds=ts,
                        state=PlayingState.PLAYING,
                    )
                )

            if not in_pause:
                note = riff_notes[beat_idx % len(riff_notes)]
                state_manager.apply(
                    OnsetEvent(frame_index=frame, timestamp_seconds=ts, strength=0.8)
                )
                state_manager.apply(
                    PitchEvent(
                        frame_index=frame,
                        timestamp_seconds=ts,
                        midi_note=note,
                        frequency_hz=82.41,
                        confidence=0.95,
                    )
                )
                mrt.note_on(note)
                # Cycle intensity between low, mid, and high tiers across the 10-minute jam
                intensity = 0.25 if (bar_num % 30) < 10 else (0.55 if (bar_num % 30) < 20 else 0.85)
                state_manager.apply(
                    IntensityUpdate(
                        frame_index=frame,
                        timestamp_seconds=ts,
                        intensity=intensity,
                    )
                )
                mrt.set_intensity(intensity)

            state_manager.update_beat(frame, beat_clock.position_at(frame))
            drums.schedule_available(current_frame=frame)
            ai_entry.update(
                state_manager.snapshot,
                drums.drummer_state,
                drums.controller.entry_bar,
            )
            mrt.set_gate_gain(ai_entry.gate_gain)

            if drums.drummer_state == DrummerState.HOLDING:
                saw_holding = True
            elif drums.drummer_state == DrummerState.STOPPED:
                saw_stopped = True
            elif saw_stopped and drums.drummer_state == DrummerState.PLAYING:
                reentered_after_pause = True
            if ai_entry.is_active:
                saw_ai_active = True
            elif saw_ai_active and in_pause:
                saw_ai_gated = True

        # Keep AI output ring topped up
        if mrt.output_ring is not None:
            while mrt.output_ring.available_frames() < block_frames:
                mrt.generate_chunk()

        processor.process(
            in_block,
            out_block,
            input_overflow=False,
            output_underflow=False,
        )

        # Drain recorder every 2 seconds (750 blocks = 96,000 frames)
        if ((frame // block_frames) + 1) % 750 == 0:
            recorder.drain_available()

    recorder.stop()

    assert processor.frames_processed == total_frames
    assert processor.input_overflows == 0
    assert processor.output_underflows == 0
    assert processor.ai_underruns == 0
    assert drum_sampler.event_ring.dropped_events == 0
    assert drum_sampler.stolen_voices == 0
    assert saw_holding is True
    assert saw_stopped is True
    assert saw_ai_gated is True
    assert reentered_after_pause is True
    assert recorder.dropped_frames == 0
    assert recorder.frames_written == total_frames

    info = sf.info(rec_path)
    assert info.frames == total_frames
    assert info.channels == 2
    assert info.samplerate == sample_rate_hz


def test_post_mvp_milestones_5_6_7_full_system_validation(
    tmp_path: Path,
    fake_stream,
) -> None:
    """Step 7.3 (§§152-154): Full-system validation of Milestones 5, 6, and 7."""
    del fake_stream
    import importlib.util
    import sys

    import mido

    from jammate.ai.model_manager import inspect_model_assets, populate_placeholder_model_assets
    from jammate.ai.native_bridge import NativeBridgeAccompaniment
    from jammate.audio.engine import AudioEngine
    from jammate.midi.input import FakeMidiPort, MidiInputWorker
    from jammate.native import native_available, yin_cmndf_f32
    from jammate.session import JamSession
    from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE

    sample_rate_hz = 48_000
    bpm = 120.0
    frames_per_beat = int(sample_rate_hz * 60.0 / bpm)  # 24,000 frames

    # 1. Milestone 5 + 6: JamSession with guitar, stem recording, MIDI input, and AI
    stems_root = tmp_path / "recordings"
    fake_midi_port = FakeMidiPort()
    cfg = JamMateConfig(
        audio=AudioConfig(monitor_gain=0.8, drums_gain=0.8, ai_gain=0.7, ai_muted=False),
        analysis=AnalysisConfig(instrument="guitar"),
        drums=DrumConfig(
            style="rock",
            humanize_timing_ms=5.0,
            humanize_velocity_variation=0.06,
        ),
        ai=AccompanimentConfig(enabled=True, style="rock", entry_delay_bars=1),
        recording=RecordingConfig(enabled=True, record_stems=True, output_dir=stems_root),
    )

    fake_mrt = Mrt2Accompaniment(
        cfg.ai,
        sample_rate_hz=sample_rate_hz,
        system_factory=lambda _size: _FastFakeMrtSystem(),
    )
    session = JamSession(
        cfg,
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        accompaniment_factory=lambda _ai_cfg, _sr: fake_mrt,
    )
    session.start()
    try:
        # Tap tempo to lock 120 BPM (Milestone 6)
        assert session.tap_tempo(1.0) is None
        assert session.tap_tempo(1.5) == pytest.approx(120.0)
        assert session.tap_tempo(2.0) == pytest.approx(120.0)
        assert session.state.bpm == pytest.approx(120.0)
        assert session.state.tempo_locked is True

        # Helper to simulate playing 1 bar (4 beats) of a riff at `bar_index` (0-based)
        def _play_bar(
            bar_idx: int,
            notes: tuple[int, int, int, int],
            *,
            chord: tuple[int, ...] = (),
        ) -> None:
            for b_in_bar, midi_note in enumerate(notes):
                beat_idx = bar_idx * 4 + b_in_bar
                frame = beat_idx * frames_per_beat
                ts = frames_to_seconds(frame, sample_rate_hz)
                if bar_idx == 0 and b_in_bar == 0:
                    session._handle_event(
                        PlayingStateUpdate(
                            frame_index=frame,
                            timestamp_seconds=ts,
                            state=PlayingState.PLAYING,
                        )
                    )
                session._handle_event(
                    OnsetEvent(frame_index=frame, timestamp_seconds=ts, strength=0.85)
                )
                session._handle_event(
                    PitchEvent(
                        frame_index=frame,
                        timestamp_seconds=ts,
                        midi_note=midi_note,
                        frequency_hz=440.0 * (2.0 ** ((midi_note - 69) / 12.0)),
                        confidence=0.95,
                        chord_notes=chord,
                        chord_type="power" if len(chord) >= 2 else None,
                    )
                )
                session._handle_event(
                    IntensityUpdate(frame_index=frame, timestamp_seconds=ts, intensity=0.65)
                )

        # Section A (bars 0..3): Riff A in E minor (E2, G2, A2, B2) with E5 power chord on beat 1
        riff_a = (40, 43, 45, 47)
        for bar_idx in range(4):
            _play_bar(bar_idx, riff_a, chord=(40, 47, 52))
        state_a = session.state
        assert state_a.riff_id == "riff_A"
        assert state_a.riff_iteration >= 3
        assert state_a.chord_type == "power"

        # Section B (bars 4..6): Contrasting Riff B (C3, D3, B2, G2) -> triggers riff transition
        riff_b = (48, 50, 47, 43)
        saw_transition_fill = False
        for bar_idx in range(4, 7):
            _play_bar(bar_idx, riff_b)
            if session.directive.trigger_fill_now or session.directive.trigger_crash_now:
                saw_transition_fill = True
        state_b = session.state
        assert state_b.riff_id == "riff_B"
        assert saw_transition_fill is True

        # Solo section (bars 7..9): High-register guitar lead (E4, G4, A4, B4 >= MIDI 64)
        solo_notes = (64, 67, 69, 71)
        for bar_idx in range(7, 10):
            _play_bar(bar_idx, solo_notes)
        state_solo = session.state
        assert state_solo.bass_register == "high"
        assert state_solo.section in ("groove", "variation", "fill")
        assert session.directive.ai_role == "open_space"
        assert session.directive.ai_gain_multiplier <= 0.75

        # Return to Section A (bars 10..12): SessionMemory recalls riff_A
        for bar_idx in range(10, 13):
            _play_bar(bar_idx, riff_a)
        state_return = session.state
        assert state_return.riff_id == "riff_A"
        assert len(session.riff_detector.memory) >= 2

        # Multi-meter switch (4/4 -> 3/4 -> 6/8), manual key override, and live context reset
        session.set_meter("3/4")
        assert session.beat_clock.beats_per_bar == 3
        session.set_meter("6/8")
        assert session.beat_clock.beats_per_bar == 6
        session.set_meter("4/4")
        session.set_manual_key("E minor")
        assert session.state.estimated_key == "E"
        assert session.state.estimated_mode == "minor"

        # MIDI keyboard input (Milestone 6): inject E5 power chord via FakeMidiPort
        midi_worker = MidiInputWorker(
            cfg.analysis,
            sample_rate_hz=sample_rate_hz,
            port=fake_midi_port,
            frame_source=lambda: session.engine.processor.frames_processed,
            on_event=session._handle_event,
        )
        fake_midi_port.send(mido.Message("note_on", note=40, velocity=100))
        fake_midi_port.send(mido.Message("note_on", note=47, velocity=96))
        assert midi_worker.process_available() == 2
        assert session.state.current_note == 40
        assert session.state.chord_notes == (40, 47)
        midi_worker.stop()
        assert midi_worker.stats().active_notes == ()

        # Process 10 audio callback blocks so 4-stem recorder captures aligned audio
        if fake_mrt.output_ring is not None:
            fake_mrt.generate_chunk()
        in_block = np.full((128, 1), 0.2, dtype=np.float32)
        out_block = np.zeros((128, 2), dtype=np.float32)
        for _ in range(10):
            session.engine.processor.process(in_block, out_block)

        # Live context reset while running
        session.set_manual_bpm(None)
        session.set_manual_key(None)
        session.reset_musical_context()
        assert session.state.tempo_locked is False
        assert len(session.riff_detector.memory) == 0

        stem_dir = session.stem_recorder.dir_path
        assert stem_dir is not None
    finally:
        session.stop()

    # Verify 4-stem WAV files written with identical frame counts (10 * 128 = 1280 frames)
    for stem_file in ("bass.wav", "drums.wav", "ai_guitar.wav", "master.wav"):
        path = stem_dir / stem_file
        assert path.is_file()
        assert sf.info(path).frames == 1280

    # 2. Milestone 7: Native C++ DSP, shared-memory AI bridge, model manager, and .app bundle
    assert native_available() is True
    win = np.sin(np.linspace(0.0, 20.0 * np.pi, 2048, dtype=np.float32))
    np.testing.assert_allclose(
        yin_cmndf_f32(win, 800, use_native=True),
        yin_cmndf_f32(win, 800, use_native=False),
        rtol=1e-6,
        atol=1e-6,
    )

    bridge = NativeBridgeAccompaniment(AccompanimentConfig(enabled=True))
    try:
        bridge.push_external_frames(np.full((1920, 2), 0.25, dtype=np.float32))
        assert bridge.drain_once() == 1920
        assert bridge.output_ring is not None
        assert bridge.output_ring.available_frames() == 1920
    finally:
        bridge.close()

    magenta_dir = tmp_path / "magenta-rt-v2"
    assert not inspect_model_assets(magenta_dir).is_ready
    assert populate_placeholder_model_assets(magenta_dir).is_ready

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_macos_app.py"
    spec = importlib.util.spec_from_file_location("build_macos_app_e2e", script_path)
    assert spec is not None and spec.loader is not None
    build_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build_mod
    spec.loader.exec_module(build_mod)
    assert build_mod.validate_packaging_config().ok is True
    app_path = build_mod.build_app_bundle(tmp_path / "dist", sign=False)
    assert build_mod.verify_built_bundle(app_path) == ()

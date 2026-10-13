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

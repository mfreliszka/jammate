"""Tests for MasterRecorder and UserPreferences (Step 4.5, §§59, 84, 85, 129)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.audio.engine import BlockProcessor
from jammate.audio.recorder import (
    MasterRecorder,
    format_recording_filename,
    resolve_recording_path,
)
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import (
    AudioConfig,
    JamMateConfig,
    RecordingConfig,
    UserPreferences,
    load_preferences,
    save_preferences,
)


def test_format_and_resolve_recording_path(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 10, 14, 19, 45)
    assert format_recording_filename(fixed_now) == "2026-10-14_19-45-jam.wav"

    first = resolve_recording_path(tmp_path, now=fixed_now)
    assert first == tmp_path / "2026-10-14_19-45-jam.wav"
    first.write_bytes(b"placeholder")

    second = resolve_recording_path(tmp_path, now=fixed_now)
    assert second == tmp_path / "2026-10-14_19-45-jam-2.wav"
    second.write_bytes(b"placeholder")

    third = resolve_recording_path(tmp_path, now=fixed_now)
    assert third == tmp_path / "2026-10-14_19-45-jam-3.wav"


def test_master_recorder_manual_drain_and_stop_flushes_exact_samples(tmp_path: Path) -> None:
    rec_cfg = RecordingConfig(output_dir=tmp_path, ring_capacity_seconds=1.0)
    recorder = MasterRecorder(rec_cfg, sample_rate_hz=48_000)
    path = recorder.open_file(now=datetime(2026, 10, 14, 20, 0))
    assert recorder.is_recording
    assert recorder.active_ring is recorder.ring

    t = np.arange(1_024, dtype=np.float32) / 48_000.0
    left = 0.4 * np.sin(2.0 * np.pi * 110.0 * t).astype(np.float32)
    right = 0.25 * np.cos(2.0 * np.pi * 220.0 * t).astype(np.float32)
    stereo = np.column_stack((left, right))

    recorder.ring.write(stereo[:512])
    drained = recorder.drain_available()
    assert drained == 512

    # Leave the second half buffered in the ring and verify stop() flushes it (§84).
    recorder.ring.write(stereo[512:])
    closed_path = recorder.stop()
    assert closed_path == path
    assert not recorder.is_recording
    assert recorder.active_ring is None
    assert recorder.frames_written == 1_024
    assert recorder.dropped_frames == 0

    read_back, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 48_000
    assert read_back.shape == (1_024, 2)
    np.testing.assert_allclose(read_back, stereo, atol=1e-6)


def test_master_recorder_background_thread_streams_and_flushes(tmp_path: Path) -> None:
    rec_cfg = RecordingConfig(output_dir=tmp_path, ring_capacity_seconds=1.0)
    recorder = MasterRecorder(rec_cfg, sample_rate_hz=48_000)
    path = recorder.start(now=datetime(2026, 10, 14, 20, 15))
    try:
        block = np.full((128, 2), 0.125, dtype=np.float32)
        for _ in range(20):
            recorder.ring.write(block)
    finally:
        recorder.stop()

    assert path.is_file()
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 48_000
    assert data.shape == (20 * 128, 2)
    np.testing.assert_allclose(data, 0.125, atol=1e-6)


def test_master_recorder_overflow_counts_drops_and_never_blocks(tmp_path: Path) -> None:
    rec_cfg = RecordingConfig(output_dir=tmp_path, ring_capacity_seconds=0.25)
    recorder = MasterRecorder(rec_cfg, sample_rate_hz=48_000)
    path = recorder.open_file(now=datetime(2026, 10, 14, 20, 30))

    capacity = recorder.ring.capacity_frames
    block = np.ones((1_024, 2), dtype=np.float32) * 0.2
    total_pushed = capacity + 4_096
    pushed = 0
    while pushed < total_pushed:
        recorder.ring.write(block)
        pushed += len(block)

    recorder.stop()
    stats = recorder.stats()
    assert stats.dropped_frames == pushed - capacity
    assert stats.overrun_events > 0

    info = sf.info(path)
    assert info.frames == capacity


def test_five_minute_recording_via_block_processor_without_xruns(tmp_path: Path) -> None:
    sample_rate_hz = 48_000
    block_size = 128
    duration_seconds = 300  # 5 simulated minutes (§59, Step 4.5 Done when)
    total_frames = sample_rate_hz * duration_seconds
    total_blocks = total_frames // block_size

    audio_cfg = AudioConfig(
        sample_rate_hz=sample_rate_hz,
        block_size_frames=block_size,
        monitor_gain=0.8,
    )
    input_ring = RingBuffer(capacity_frames=sample_rate_hz * 2, channels=1)
    rec_cfg = RecordingConfig(output_dir=tmp_path, ring_capacity_seconds=10.0)
    recorder = MasterRecorder(rec_cfg, sample_rate_hz=sample_rate_hz)
    path = recorder.open_file(now=datetime(2026, 10, 14, 21, 0))

    processor = BlockProcessor(
        input_ring=input_ring,
        input_channel_index=0,
        monitor_gain=audio_cfg.monitor_gain,
        soft_clip_threshold=audio_cfg.soft_clip_threshold,
        max_block_frames=audio_cfg.fallback_block_size_frames,
        record_ring=recorder.active_ring,
    )

    t = np.arange(block_size, dtype=np.float32) / sample_rate_hz
    indata = (0.35 * np.sin(2.0 * np.pi * 82.41 * t)).reshape(block_size, 1).astype(np.float32)
    outdata = np.zeros((block_size, 2), dtype=np.float32)
    expected_peak = float(np.max(np.abs(indata))) * audio_cfg.monitor_gain

    # Drain every 2 seconds of simulated audio (750 blocks = 96,000 frames).
    drain_every_blocks = 750
    for block_idx in range(1, total_blocks + 1):
        processor.process(
            indata,
            outdata,
            input_overflow=False,
            output_underflow=False,
        )
        if block_idx % drain_every_blocks == 0:
            recorder.drain_available()

    recorder.stop()

    assert processor.input_overflows == 0
    assert processor.output_underflows == 0
    assert recorder.dropped_frames == 0
    assert recorder.frames_written == total_frames

    info = sf.info(path)
    assert info.samplerate == sample_rate_hz
    assert info.channels == 2
    assert info.frames == total_frames

    # Spot-check start and end of the 5-minute WAV file for clean audio without NaNs or clipping.
    head, _ = sf.read(path, frames=1_024, start=0, dtype="float32", always_2d=True)
    tail, _ = sf.read(
        path,
        frames=1_024,
        start=total_frames - 1_024,
        dtype="float32",
        always_2d=True,
    )
    assert np.all(np.isfinite(head))
    assert np.all(np.isfinite(tail))
    assert float(np.max(np.abs(head))) == pytest.approx(expected_peak, rel=1e-4)
    assert float(np.max(np.abs(tail))) == pytest.approx(expected_peak, rel=1e-4)


def test_user_preferences_round_trip_and_corrupt_fallback(tmp_path: Path) -> None:
    prefs_path = tmp_path / "JamMate" / "settings.json"
    # Missing file returns defaults
    default_prefs = load_preferences(prefs_path)
    assert default_prefs == UserPreferences()

    custom = UserPreferences(
        input_device_name="USB Audio CODEC",
        output_device_name="USB Audio CODEC",
        instrument="guitar",
        block_size_frames=256,
        style="stoner_rock",
        model_size="mrt2_base",
        drums_gain=1.15,
        ai_gain=0.65,
    )
    save_preferences(custom, prefs_path)
    assert prefs_path.is_file()

    loaded = load_preferences(prefs_path)
    assert loaded == custom

    applied = loaded.apply_to_config(JamMateConfig())
    assert applied.audio.input_device_name == "USB Audio CODEC"
    assert applied.audio.output_device_name == "USB Audio CODEC"
    assert applied.analysis.instrument == "guitar"
    assert applied.audio.block_size_frames == 256
    assert applied.drums.style == "stoner_rock"
    assert applied.ai.style == "stoner_rock"
    assert applied.ai.model_size == "mrt2_base"
    assert applied.audio.drums_gain == pytest.approx(1.15)
    assert applied.audio.ai_gain == pytest.approx(0.65)
    assert UserPreferences.from_config(applied) == custom

    # Corrupt file falls back gracefully to defaults
    prefs_path.write_text("{not valid json", encoding="utf-8")
    fallback = load_preferences(prefs_path)
    assert fallback == UserPreferences()

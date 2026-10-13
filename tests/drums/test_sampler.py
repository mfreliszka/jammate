import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.config import DEFAULT_KIT_DIR, DrumConfig
from jammate.drums.sampler import (
    DrumEventRing,
    DrumPiece,
    DrumSampler,
    load_kit,
    load_sample,
)

BLOCK = 128


def _ramp_kit(length: int = 300) -> dict[DrumPiece, np.ndarray]:
    """Synthetic kit where each piece is a distinct non-zero linear ramp."""
    return {
        piece: np.linspace(0.1 * (int(piece) + 1), 0.5 * (int(piece) + 1), length, dtype=np.float32)
        for piece in DrumPiece
    }


def test_loads_default_placeholder_kit_at_48khz() -> None:
    kit = load_kit(DEFAULT_KIT_DIR, sample_rate_hz=48_000)
    assert set(kit) == set(DrumPiece)
    for sample in kit.values():
        assert sample.dtype == np.float32
        assert sample.ndim == 1
        assert sample.flags.c_contiguous
        assert sample.size > 1_000


def test_load_sample_downmixes_stereo_and_resamples_to_target_rate(tmp_path: Path) -> None:
    source_rate = 44_100
    duration_s = 0.1
    t = np.arange(round(source_rate * duration_s), dtype=np.float32) / source_rate
    tone = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    path = tmp_path / "stereo_44k.wav"
    sf.write(path, stereo, source_rate, subtype="FLOAT")

    loaded = load_sample(path, target_sample_rate_hz=48_000)

    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert loaded.shape[0] == round(48_000 * duration_s)
    assert float(np.max(np.abs(loaded))) == pytest.approx(0.75, rel=0.05)


def test_event_ring_pushes_wraps_and_drops_when_full() -> None:
    ring = DrumEventRing(capacity=4)
    for i in range(4):
        assert ring.push(i * 100, DrumPiece.KICK, 0.8)
    assert ring.pending_count() == 4
    assert not ring.push(400, DrumPiece.SNARE, 0.9)
    assert ring.dropped_events == 1

    sampler = DrumSampler(_ramp_kit(50), max_voices=4, event_ring_capacity=4)
    for i in range(4):
        assert sampler.schedule(i * 10, DrumPiece.KICK, 0.5)
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    sampler.render_block(out, 0)
    assert sampler.event_ring.pending_count() == 0

    # Wrap-around after draining.
    for i in range(4):
        assert sampler.schedule(BLOCK + i * 10, DrumPiece.SNARE, 0.5)
    assert sampler.event_ring.pending_count() == 4


def test_event_at_frame_n_renders_at_exact_offset_across_block_boundaries() -> None:
    kit = _ramp_kit(length=300)
    sampler = DrumSampler(kit, max_voices=8)
    trigger_frame = 200  # block 1 (frames 128..256), intra-block offset 72
    gain = 0.75
    sampler.schedule(trigger_frame, DrumPiece.SNARE, gain)

    blocks: list[np.ndarray] = []
    for b in range(5):
        out = np.full((BLOCK, 2), 99.0, dtype=np.float32)
        sampler.render_block(out, b * BLOCK)
        blocks.append(out)

    rendered = np.concatenate(blocks, axis=0)
    expected = np.zeros((5 * BLOCK, 2), dtype=np.float32)
    sample = kit[DrumPiece.SNARE] * gain
    expected[trigger_frame : trigger_frame + len(sample), 0] = sample
    expected[trigger_frame : trigger_frame + len(sample), 1] = sample

    np.testing.assert_allclose(rendered, expected, rtol=1e-6, atol=1e-6)
    assert sampler.events_played == 1
    assert sampler.active_voice_count() == 0


def test_simultaneous_and_staggered_events_within_one_block_sum_accurately() -> None:
    kit = _ramp_kit(length=200)
    sampler = DrumSampler(kit, max_voices=8)
    sampler.schedule(10, DrumPiece.KICK, 0.5)
    sampler.schedule(10, DrumPiece.CLOSED_HAT, 0.25)
    sampler.schedule(70, DrumPiece.SNARE, 0.8)

    out = np.zeros((BLOCK, 2), dtype=np.float32)
    sampler.render_block(out, 0)

    expected = np.zeros(BLOCK, dtype=np.float32)
    expected[10:] += kit[DrumPiece.KICK][: BLOCK - 10] * 0.5
    expected[10:] += kit[DrumPiece.CLOSED_HAT][: BLOCK - 10] * 0.25
    expected[70:] += kit[DrumPiece.SNARE][: BLOCK - 70] * 0.8

    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], expected, rtol=1e-6, atol=1e-6)
    assert sampler.active_voice_count() == 3


def test_late_event_starts_at_block_start_and_increments_late_counter() -> None:
    kit = _ramp_kit(length=64)
    sampler = DrumSampler(kit, max_voices=4)
    sampler.schedule(100, DrumPiece.TOM, 1.0)

    out = np.zeros((BLOCK, 2), dtype=np.float32)
    sampler.render_block(out, 128)

    assert sampler.late_events == 1
    np.testing.assert_allclose(out[:64, 0], kit[DrumPiece.TOM])
    assert not out[64:].any()


def test_voice_pool_overflow_steals_oldest_voice_cleanly() -> None:
    kit = _ramp_kit(length=256)
    sampler = DrumSampler(kit, max_voices=2)
    sampler.schedule(0, DrumPiece.KICK, 1.0)
    sampler.schedule(20, DrumPiece.SNARE, 1.0)
    sampler.schedule(50, DrumPiece.CRASH, 1.0)  # steals KICK at offset 50

    out = np.zeros((BLOCK, 2), dtype=np.float32)
    sampler.render_block(out, 0)

    assert sampler.stolen_voices == 1
    assert sampler.active_voice_count() == 2

    expected = np.zeros(BLOCK, dtype=np.float32)
    # KICK plays for [0, 50), then is stolen by CRASH at frame 50.
    expected[0:50] += kit[DrumPiece.KICK][0:50]
    expected[20:BLOCK] += kit[DrumPiece.SNARE][0 : BLOCK - 20]
    expected[50:BLOCK] += kit[DrumPiece.CRASH][0 : BLOCK - 50]

    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-6, atol=1e-6)


def test_callback_cost_micro_benchmark_at_128_frames_is_within_budget() -> None:
    sampler = DrumSampler.from_config(DrumConfig(), sample_rate_hz=48_000)
    out = np.zeros((BLOCK, 2), dtype=np.float32)

    # Pre-schedule hits across 500 blocks so multiple voices stay active simultaneously.
    for block_idx in range(500):
        frame = block_idx * BLOCK + (block_idx * 17) % BLOCK
        piece = DrumPiece(block_idx % len(DrumPiece))
        sampler.schedule(frame, piece, 0.8)
        if block_idx % 2 == 0:
            sampler.schedule(frame, DrumPiece.CLOSED_HAT, 0.5)

    durations_ns: list[int] = []
    for block_idx in range(500):
        t0 = time.perf_counter_ns()
        sampler.render_block(out, block_idx * BLOCK)
        durations_ns.append(time.perf_counter_ns() - t0)

    mean_us = float(np.mean(durations_ns)) / 1_000.0
    p99_us = float(np.percentile(durations_ns, 99)) / 1_000.0
    # Nominal 128-frame block period at 48 kHz is 2,667 us; sampler should stay well under 250 us.
    assert mean_us < 250.0
    assert p99_us < 1_000.0

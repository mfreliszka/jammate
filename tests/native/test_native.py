"""Unit and parity tests for the C++20 native extension and shared-memory AI bridge (Step 7.1)."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, noise, plucked_note

from jammate.ai.base import AccompanimentEngine, AccompanimentStatus, NoteTracker
from jammate.ai.native_bridge import NativeBridgeAccompaniment, SharedMemoryAudioRing
from jammate.analysis.pitch import yin, yin_lag_range, yin_window_frames
from jammate.audio.engine import BlockProcessor
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AccompanimentConfig
from jammate.drums.sampler import DrumPiece, DrumSampler
from jammate.music.events import PitchEvent, SilenceEvent
from jammate.native import (
    compile_native_extension,
    mix_drum_voices_f32,
    native_available,
    ring_copy_range_f32,
    ring_write_f32,
    yin_cmndf_f32,
    yin_cmndf_numpy,
    yin_estimate,
)

SAMPLE_RATE_HZ = 48_000


def test_native_extension_compiles_and_is_available() -> None:
    dylib_path = compile_native_extension()
    assert dylib_path.is_file()
    assert native_available() is True


@pytest.mark.parametrize(
    ("note_name", "min_hz", "max_hz"),
    [
        ("E2", 60.0, 1400.0),  # Guitar E2
        ("A2", 60.0, 1400.0),  # Guitar A2
        ("G3", 60.0, 1400.0),  # Guitar G3
        ("E1", 35.0, 400.0),  # Bass E1
    ],
)
def test_yin_cmndf_f32_cpp_vs_numpy_parity_on_voiced_notes(
    note_name: str,
    min_hz: float,
    max_hz: float,
) -> None:
    assert native_available()
    _, tau_max = yin_lag_range(SAMPLE_RATE_HZ, min_hz, max_hz)
    win_len = yin_window_frames(SAMPLE_RATE_HZ, min_hz)
    clip = plucked_note(
        note_name,
        duration_seconds=0.15,
        sample_rate_hz=SAMPLE_RATE_HZ,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    window = clip[1000 : 1000 + win_len]

    cmndf_cpp = yin_cmndf_f32(window, tau_max, use_native=True)
    cmndf_py = yin_cmndf_f32(window, tau_max, use_native=False)
    cmndf_direct = yin_cmndf_numpy(window, tau_max)

    assert cmndf_cpp.dtype == np.float32
    assert cmndf_py.dtype == np.float32
    assert cmndf_cpp.shape == (tau_max + 2,)
    np.testing.assert_array_equal(cmndf_py, cmndf_direct)
    max_diff = float(np.max(np.abs(cmndf_cpp - cmndf_py)))
    assert max_diff < 1e-6


@pytest.mark.parametrize("note_name", ["E2", "A2", "D3", "G3", "E4"])
def test_yin_estimate_cpp_vs_numpy_parity_voiced_and_unvoiced(note_name: str) -> None:
    assert native_available()
    min_hz, max_hz = 60.0, 1400.0
    threshold = 0.15
    win_len = yin_window_frames(SAMPLE_RATE_HZ, min_hz)

    clip = plucked_note(
        note_name,
        duration_seconds=0.12,
        sample_rate_hz=SAMPLE_RATE_HZ,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    window = clip[800 : 800 + win_len]

    res_cpp = yin_estimate(window, SAMPLE_RATE_HZ, min_hz, max_hz, threshold, use_native=True)
    res_py = yin_estimate(window, SAMPLE_RATE_HZ, min_hz, max_hz, threshold, use_native=False)
    res_ref = yin(window, SAMPLE_RATE_HZ, min_hz, max_hz, threshold)

    assert res_cpp.frequency_hz is not None
    assert res_py.frequency_hz is not None
    assert res_ref.frequency_hz is not None
    assert abs(res_cpp.frequency_hz - res_py.frequency_hz) < 1e-6
    assert abs(res_cpp.frequency_hz - res_ref.frequency_hz) < 1e-6
    assert abs(res_cpp.confidence - res_py.confidence) < 1e-6

    # Unvoiced white noise parity
    noise_win = noise(duration_seconds=0.1, sample_rate_hz=SAMPLE_RATE_HZ, seed=42)[:win_len]
    noise_cpp = yin_estimate(noise_win, SAMPLE_RATE_HZ, min_hz, max_hz, threshold, use_native=True)
    noise_py = yin_estimate(noise_win, SAMPLE_RATE_HZ, min_hz, max_hz, threshold, use_native=False)
    assert noise_cpp.frequency_hz is None
    assert noise_py.frequency_hz is None
    assert abs(noise_cpp.confidence - noise_py.confidence) < 1e-6


def test_ring_write_and_copy_range_f32_cpp_vs_numpy_bit_exact_wraparound() -> None:
    assert native_available()
    capacity = 256
    channels = 2
    ring_cpp = np.zeros((capacity, channels), dtype=np.float32)
    ring_py = np.zeros((capacity, channels), dtype=np.float32)

    rng = np.random.default_rng(123)
    write_pos = 0
    for step in range(10):
        chunk_len = 70 + (step * 19) % 90  # guarantees mid-buffer and wrap-around writes
        chunk = rng.uniform(-1.0, 1.0, size=(chunk_len, channels)).astype(np.float32)
        ring_write_f32(ring_cpp, write_pos, chunk, use_native=True)
        ring_write_f32(ring_py, write_pos, chunk, use_native=False)
        np.testing.assert_array_equal(ring_cpp, ring_py)

        read_len = min(chunk_len + 20, capacity)
        start_pos = max(0, write_pos + chunk_len - read_len)
        out_cpp = np.zeros((read_len, channels), dtype=np.float32)
        out_py = np.zeros((read_len, channels), dtype=np.float32)
        ring_copy_range_f32(ring_cpp, start_pos, read_len, out_cpp, use_native=True)
        ring_copy_range_f32(ring_py, start_pos, read_len, out_py, use_native=False)
        np.testing.assert_array_equal(out_cpp, out_py)

        write_pos += chunk_len


@pytest.mark.parametrize("channels", [1, 2])
def test_mix_drum_voices_f32_cpp_vs_numpy_parity(channels: int) -> None:
    assert native_available()
    rng = np.random.default_rng(99)
    samples = [
        rng.uniform(-0.8, 0.8, size=200).astype(np.float32),
        rng.uniform(-0.8, 0.8, size=90).astype(np.float32),
        rng.uniform(-0.8, 0.8, size=350).astype(np.float32),
    ]
    active_cpp = [True, True, False, True]
    active_py = list(active_cpp)
    pieces = [0, 1, 2, 2]
    gains = [0.85, 0.6, 1.0, 0.4]
    pos_cpp = [0, 40, 0, 120]  # voice 1 will exhaust mid-block (40 + 50 == 90)
    pos_py = list(pos_cpp)

    shape = (128,) if channels == 1 else (128, channels)
    out_cpp = np.zeros(shape, dtype=np.float32)
    out_py = np.zeros(shape, dtype=np.float32)
    scratch = np.zeros(128, dtype=np.float32)

    mix_drum_voices_f32(
        samples,
        active_cpp,
        pieces,
        gains,
        pos_cpp,
        out_cpp,
        16,
        112,
        use_native=True,
    )
    mix_drum_voices_f32(
        samples,
        active_py,
        pieces,
        gains,
        pos_py,
        out_py,
        16,
        112,
        use_native=False,
        mono_scratch=scratch,
    )

    assert active_cpp == active_py
    assert active_cpp[1] is False  # exhausted voice deactivated
    assert pos_cpp == pos_py
    np.testing.assert_allclose(out_cpp, out_py, rtol=1e-6, atol=1e-6)


def test_shared_memory_audio_ring_cross_attach_wraparound_and_conditioning() -> None:
    producer = SharedMemoryAudioRing(capacity_frames=512, channels=2, create=True)
    consumer: SharedMemoryAudioRing | None = None
    try:
        consumer = SharedMemoryAudioRing(
            capacity_frames=512,
            channels=2,
            name=producer.name,
            create=False,
        )
        assert consumer.capacity == 512
        assert consumer.channels == 2

        producer.set_conditioning(active_note=40, intensity=0.75, follow_improvise=0.35)
        assert consumer.active_note == 40
        assert consumer.intensity == pytest.approx(0.75, abs=1e-3)
        assert consumer.follow_improvise == pytest.approx(0.35, abs=1e-3)

        # Write across wrap-around boundary and verify lossless read.
        first_chunk = np.ones((400, 2), dtype=np.float32) * 0.25
        assert producer.write(first_chunk) == 400
        read_buf = np.zeros((400, 2), dtype=np.float32)
        assert consumer.read_into(read_buf) == 400
        np.testing.assert_allclose(read_buf, first_chunk)

        wrap_chunk = np.linspace(-0.5, 0.5, 300 * 2, dtype=np.float32).reshape(300, 2)
        assert producer.write(wrap_chunk) == 300
        assert consumer.available_read() == 300
        wrap_out = np.zeros((300, 2), dtype=np.float32)
        assert consumer.read_into(wrap_out) == 300
        np.testing.assert_allclose(wrap_out, wrap_chunk)

        # Overflow drops oldest unread frames cleanly and counts them.
        overflow = np.ones((600, 2), dtype=np.float32) * 0.5
        producer.write(np.ones((300, 2), dtype=np.float32))
        producer.write(overflow[:350])  # total unread 650 > 512 capacity -> drops 138
        drain_buf = np.zeros((512, 2), dtype=np.float32)
        read_n = consumer.read_into(drain_buf)
        assert read_n == 512
        assert consumer.dropped_frames == 138
    finally:
        if consumer is not None:
            consumer.close()
        producer.unlink()


def test_native_bridge_accompaniment_streams_into_audio_engine_and_stops_cleanly() -> None:
    def _synthetic_producer(shm_ring: SharedMemoryAudioRing) -> float:
        if shm_ring.available_read() < 3840:
            tone = np.full((1920, 2), 0.2, dtype=np.float32)
            if shm_ring.active_note is not None:
                tone *= 1.5
            shm_ring.write(tone)
        return 4.5

    bridge = NativeBridgeAccompaniment(
        AccompanimentConfig(enabled=True, style="rock", follow_improvise=0.4),
        producer_fn=_synthetic_producer,
        poll_interval_seconds=0.001,
    )
    try:
        assert isinstance(bridge, AccompanimentEngine)
        assert bridge.status == AccompanimentStatus.READY
        assert bridge.output_ring is not None

        tracker = NoteTracker(bridge)
        tracker.handle_event(
            PitchEvent(
                frame_index=0,
                timestamp_seconds=0.0,
                midi_note=40,
                frequency_hz=82.41,
                confidence=0.95,
            )
        )
        assert bridge.shm_ring.active_note == 40

        bridge.set_intensity(0.8)
        bridge.set_follow_improvise(0.6)
        bridge.set_style("funk")

        # Deterministic push + drain before starting worker thread
        pushed = np.full((1920, 2), 0.125, dtype=np.float32)
        bridge.push_external_frames(pushed, ms_per_frame=3.2)
        drained = bridge.drain_once()
        assert drained == 1920

        # Wire into BlockProcessor and verify audio callback mixes AI audio cleanly.
        input_ring = RingBuffer(capacity_frames=4096, channels=1)
        processor = BlockProcessor(
            input_ring=input_ring,
            input_channel_index=0,
            monitor_gain=0.0,
            ai_ring=bridge.output_ring,
            ai_gain=1.0,
            max_block_frames=128,
        )
        in_block = np.zeros((128, 1), dtype=np.float32)
        out_block = np.zeros((128, 2), dtype=np.float32)
        processor.process(in_block, out_block)
        np.testing.assert_allclose(out_block, 0.125, rtol=1e-6, atol=1e-6)

        # Start and stop background bridge worker thread cleanly.
        bridge.start()
        assert bridge.status == AccompanimentStatus.RUNNING
        stats = bridge.stats()
        assert stats.style == "funk"
        assert stats.active_note == 40
        assert stats.frames_generated >= 1

        tracker.handle_event(SilenceEvent(frame_index=1920, timestamp_seconds=0.04))
        assert bridge.shm_ring.active_note is None

        bridge.stop()
        assert bridge.status == AccompanimentStatus.STOPPED
    finally:
        bridge.close()


def test_native_yin_and_callback_benchmark_with_zero_numpy_allocations() -> None:
    assert native_available()
    min_hz, max_hz = 60.0, 1400.0
    _, tau_max = yin_lag_range(SAMPLE_RATE_HZ, min_hz, max_hz)
    win_len = yin_window_frames(SAMPLE_RATE_HZ, min_hz)
    clip = plucked_note(
        "A2",
        duration_seconds=0.15,
        sample_rate_hz=SAMPLE_RATE_HZ,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    window = clip[500 : 500 + win_len]

    # Warmup
    yin_cmndf_f32(window, tau_max, use_native=True)

    iters = 50
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        yin_cmndf_f32(window, tau_max, use_native=True)
    cpp_yin_us = (time.perf_counter_ns() - t0) / iters / 1_000.0
    # 1602-sample YIN CMNDF must execute well under the 10,667 us (512-frame) analysis hop budget.
    assert cpp_yin_us < 1_500.0

    # Callback drum voice mixing benchmark & allocation check.
    kit = {piece: np.linspace(0.1, -0.1, 4096, dtype=np.float32) for piece in DrumPiece}
    sampler = DrumSampler(kit, max_voices=8)
    for i, piece in enumerate(DrumPiece):
        sampler.schedule(i * 16, piece, 0.7)
    out = np.zeros((128, 2), dtype=np.float32)
    sampler.render_block(out, 0)  # warmup & activate 5 voices

    tracemalloc.start()
    try:
        snap_before = tracemalloc.take_snapshot()
        t_cb0 = time.perf_counter_ns()
        for b in range(1, 21):
            sampler.render_block(out, b * 128)
        cb_us = (time.perf_counter_ns() - t_cb0) / 20 / 1_000.0
        snap_after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    # Check that no NumPy ndarray allocations happened inside sampler.render_block.
    diffs = snap_after.compare_to(snap_before, "lineno")
    sampler_allocs = [d for d in diffs if "sampler.py" in str(d.traceback) and d.size_diff > 0]
    assert sampler_allocs == []
    assert cb_us < 200.0

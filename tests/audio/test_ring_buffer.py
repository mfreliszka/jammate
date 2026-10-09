import numpy as np
import pytest

from jammate.audio.ring_buffer import RingBuffer, RingRead


def _ramp(start: int, count: int, channels: int = 1) -> np.ndarray:
    """Frames whose value equals their absolute position (per channel offset by 0.5)."""
    positions = np.arange(start, start + count, dtype=np.float32)
    return np.stack([positions + 0.5 * channel for channel in range(channels)], axis=1)


def _write_ramp(ring: RingBuffer, start: int, count: int, block: int = 4) -> None:
    for block_start in range(start, start + count, block):
        ring.write(_ramp(block_start, min(block, start + count - block_start), ring.channels))


def test_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)
    with pytest.raises(ValueError):
        RingBuffer(8, channels=0)


def test_read_on_empty_buffer_returns_nothing() -> None:
    ring = RingBuffer(8)
    out = np.zeros((4, 1), dtype=np.float32)

    assert ring.read(out) == RingRead(start_frame=0, frame_count=0)
    assert ring.available_frames() == 0


def test_sequential_reads_return_frames_in_order() -> None:
    ring = RingBuffer(16)
    _write_ramp(ring, 0, 10)
    out = np.zeros((6, 1), dtype=np.float32)

    first = ring.read(out)
    np.testing.assert_array_equal(out[: first.frame_count], _ramp(0, 6))
    second = ring.read(out)

    assert first == RingRead(0, 6)
    assert second == RingRead(6, 4)
    np.testing.assert_array_equal(out[:4], _ramp(6, 4))
    assert ring.frames_written == 10
    assert ring.read_position == 10
    assert ring.overrun_count == 0


def test_wrap_around_preserves_data() -> None:
    ring = RingBuffer(8, channels=2)
    out = np.zeros((8, 2), dtype=np.float32)
    _write_ramp(ring, 0, 6, block=3)
    ring.read(out)

    _write_ramp(ring, 6, 6, block=3)  # crosses the end of the storage
    result = ring.read(out)

    assert result == RingRead(6, 6)
    np.testing.assert_array_equal(out[:6], _ramp(6, 6, channels=2))


def test_overrun_drops_oldest_and_counts() -> None:
    ring = RingBuffer(8)
    _write_ramp(ring, 0, 20)
    out = np.zeros((16, 1), dtype=np.float32)

    result = ring.read(out)

    assert result == RingRead(start_frame=12, frame_count=8)
    np.testing.assert_array_equal(out[:8], _ramp(12, 8))
    assert ring.overrun_count == 1
    assert ring.dropped_frames == 12
    assert ring.available_frames() == 0


def test_writes_never_block_or_fail_without_a_consumer() -> None:
    ring = RingBuffer(8)

    _write_ramp(ring, 0, 1000)

    assert ring.frames_written == 1000
    assert ring.available_frames() == 8


def test_read_latest_skips_to_newest_frames() -> None:
    ring = RingBuffer(32)
    _write_ramp(ring, 0, 20)
    out = np.zeros((5, 1), dtype=np.float32)

    result = ring.read_latest(out)

    assert result == RingRead(start_frame=15, frame_count=5)
    np.testing.assert_array_equal(out, _ramp(15, 5))
    assert ring.dropped_frames == 15
    assert ring.overrun_count == 0
    assert ring.read_position == 20


def test_read_latest_allows_overlapping_windows() -> None:
    ring = RingBuffer(32)
    window = np.zeros((8, 1), dtype=np.float32)
    _write_ramp(ring, 0, 8)
    ring.read_latest(window)

    _write_ramp(ring, 8, 4)  # one hop of 4 frames
    result = ring.read_latest(window)

    assert result == RingRead(start_frame=4, frame_count=8)
    np.testing.assert_array_equal(window, _ramp(4, 8))
    assert ring.dropped_frames == 0


def test_read_latest_before_window_is_full_returns_what_exists() -> None:
    ring = RingBuffer(32)
    _write_ramp(ring, 0, 3)
    out = np.zeros((8, 1), dtype=np.float32)

    assert ring.read_latest(out) == RingRead(0, 3)
    np.testing.assert_array_equal(out[:3], _ramp(0, 3))


def test_frames_overwritten_during_a_read_are_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the producer overwriting the oldest frames while the consumer copies them."""
    ring = RingBuffer(8)
    _write_ramp(ring, 0, 8)
    original_copy = ring._copy_range

    def copy_then_producer_writes(start: int, frame_count: int, out: np.ndarray) -> None:
        original_copy(start, frame_count, out)
        ring.write(_ramp(8, 3))  # overwrites positions 0..2 of the frames just copied

    monkeypatch.setattr(ring, "_copy_range", copy_then_producer_writes)
    out = np.zeros((8, 1), dtype=np.float32)

    result = ring.read(out)

    assert result == RingRead(start_frame=3, frame_count=5)
    np.testing.assert_array_equal(out[:5], _ramp(3, 5))
    assert ring.overrun_count == 1
    assert ring.dropped_frames == 3
    assert ring.read_position == 8

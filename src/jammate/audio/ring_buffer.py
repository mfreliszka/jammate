"""Preallocated single-producer/single-consumer ring buffer for audio frames.

The producer (normally the audio callback) only calls `write`. It never blocks, never allocates
sample storage and never looks at the consumer. When the consumer falls behind by more than the
capacity, the oldest frames are overwritten. The consumer notices on its next read, skips to the
oldest frame still valid and counts what it lost (§23).

Frame positions are absolute: frame `i` is the `i`-th frame ever written. When the producer is the
audio callback, that equals the stream's sample-frame index.

Synchronization relies on the GIL making single attribute reads and writes atomic. NumPy may
release the GIL while copying, so the producer announces the range it is about to overwrite
(`_write_end`) before copying, and the consumer re-checks that announcement after its own copy.
"""

from typing import NamedTuple

import numpy as np


class RingRead(NamedTuple):
    """Result of a consumer read: frames `[start_frame, start_frame + frame_count)`."""

    start_frame: int
    frame_count: int


class RingBuffer:
    def __init__(self, capacity_frames: int, channels: int = 1) -> None:
        if capacity_frames <= 0 or channels <= 0:
            raise ValueError("capacity_frames and channels must be positive")
        self._capacity = capacity_frames
        self._buffer = np.zeros((capacity_frames, channels), dtype=np.float32)
        # Producer-owned.
        self._frames_written = 0
        self._write_end = 0
        # Consumer-owned.
        self._read_position = 0
        self.overrun_count = 0  # times unread frames were overwritten before being copied
        self.dropped_frames = 0  # frames the consumer never received (overrun or skipped)

    @property
    def capacity_frames(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._buffer.shape[1]

    @property
    def frames_written(self) -> int:
        """Total frames written since creation (absolute position after the newest frame)."""
        return self._frames_written

    @property
    def read_position(self) -> int:
        """Absolute position of the next frame the consumer will read."""
        return self._read_position

    def available_frames(self) -> int:
        """Unread frames still in the buffer (consumer side)."""
        return min(self._frames_written - self._read_position, self._capacity)

    def write(self, frames: np.ndarray) -> None:
        """Producer: append `frames` of shape `(n, channels)`, with `n <= capacity_frames`.

        Realtime-safe: no locks, no sample allocation, no checks that could raise in the
        callback. Callers guarantee the shape.
        """
        frame_count = frames.shape[0]
        position = self._frames_written
        self._write_end = position + frame_count
        start = position % self._capacity
        first = min(frame_count, self._capacity - start)
        self._buffer[start : start + first] = frames[:first]
        if first < frame_count:
            self._buffer[: frame_count - first] = frames[first:]
        self._frames_written = position + frame_count

    def read(self, out: np.ndarray) -> RingRead:
        """Consumer: copy the next unread frames, at most `len(out)`, into `out[:frame_count]`.

        If unread frames were already overwritten, reading resumes at the oldest valid frame and
        the loss is counted as an overrun.
        """
        newest = self._frames_written
        oldest = newest - self._capacity
        start = self._read_position
        if start < oldest:
            self._record_overrun(oldest - start)
            start = oldest
        frame_count = min(out.shape[0], newest - start)
        return self._consume(start, frame_count, out)

    def read_latest(self, out: np.ndarray) -> RingRead:
        """Consumer: copy the newest frames, at most `len(out)`, and skip everything older.

        Skipping is deliberate, so skipped frames count in `dropped_frames` but not as an
        overrun. Frames that were already read can be read again (overlapping analysis windows).
        """
        newest = self._frames_written
        frame_count = min(out.shape[0], newest, self._capacity)
        start = newest - frame_count
        if start > self._read_position:
            self.dropped_frames += start - self._read_position
        return self._consume(start, frame_count, out)

    def _consume(self, start: int, frame_count: int, out: np.ndarray) -> RingRead:
        self._copy_range(start, frame_count, out)

        # The producer may have overwritten the oldest frames while we copied them.
        oldest_valid = self._write_end - self._capacity
        if start < oldest_valid:
            lost = min(oldest_valid - start, frame_count)
            out[: frame_count - lost] = out[lost:frame_count]
            self._record_overrun(lost)
            start += lost
            frame_count -= lost

        self._read_position = max(self._read_position, start + frame_count)
        return RingRead(start, frame_count)

    def _copy_range(self, start: int, frame_count: int, out: np.ndarray) -> None:
        offset = start % self._capacity
        first = min(frame_count, self._capacity - offset)
        out[:first] = self._buffer[offset : offset + first]
        if first < frame_count:
            out[first:frame_count] = self._buffer[: frame_count - first]

    def _record_overrun(self, lost_frames: int) -> None:
        self.overrun_count += 1
        self.dropped_frames += lost_frames

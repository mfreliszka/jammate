"""Realtime drum sample player and lock-free SPSC event ring (Step 3.2, §45, §46).

Samples are loaded and resampled to 48 kHz float32 at startup, before the stream opens. The
scheduler thread pushes `(frame_index, piece, gain)` events into `DrumEventRing`. On the audio
callback thread, `DrumSampler.render_block` starts voices at their exact intra-block offset and
mixes all active voices in place with vectorized NumPy and zero allocations.
"""

import math
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import soundfile as sf
from scipy import signal

from jammate.config import DEFAULT_KIT_DIR, DrumConfig
from jammate.music.events import SampleFrame
from jammate.native.kernels import get_native_lib, mix_drum_voices_f32

MAX_RENDER_BLOCK_FRAMES = 4096


class DrumPiece(IntEnum):
    """Pieces in the prototype drum kit (§45)."""

    KICK = 0
    SNARE = 1
    CLOSED_HAT = 2
    CRASH = 3
    TOM = 4


KIT_FILENAMES: dict[DrumPiece, str] = {
    DrumPiece.KICK: "kick.wav",
    DrumPiece.SNARE: "snare.wav",
    DrumPiece.CLOSED_HAT: "closed_hat.wav",
    DrumPiece.CRASH: "crash.wav",
    DrumPiece.TOM: "tom.wav",
}


class DrumEvent(NamedTuple):
    """A scheduled drum hit at an exact stream sample frame (§46)."""

    frame_index: SampleFrame
    piece: DrumPiece
    gain: float = 1.0


def load_sample(path: Path, target_sample_rate_hz: int = 48_000) -> np.ndarray:
    """Load a WAV file as a 1D contiguous float32 array at `target_sample_rate_hz`.

    Multi-channel files are downmixed to mono. Files at another sample rate are resampled with
    `scipy.signal.resample_poly` (§45). Call at startup only, never from the audio callback.
    """
    samples, file_rate_hz = sf.read(path, dtype="float32", always_2d=True)
    mono: np.ndarray = (
        samples[:, 0] if samples.shape[1] == 1 else samples.mean(axis=1, dtype=np.float32)
    )
    if file_rate_hz != target_sample_rate_hz:
        divisor = math.gcd(target_sample_rate_hz, file_rate_hz)
        up = target_sample_rate_hz // divisor
        down = file_rate_hz // divisor
        mono = signal.resample_poly(mono, up=up, down=down).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_kit(
    kit_dir: Path = DEFAULT_KIT_DIR, sample_rate_hz: int = 48_000
) -> dict[DrumPiece, np.ndarray]:
    """Load all five kit WAVs from `kit_dir` into float32 mono arrays at `sample_rate_hz`."""
    kit: dict[DrumPiece, np.ndarray] = {}
    for piece, filename in KIT_FILENAMES.items():
        path = kit_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Drum sample {path} not found. "
                "Run `uv run python scripts/make_placeholder_kit.py`."
            )
        kit[piece] = load_sample(path, sample_rate_hz)
    return kit


class DrumEventRing:
    """Bounded, preallocated single-producer/single-consumer ring for scheduled drum hits.

    The producer (scheduler worker) calls `push` in chronological `frame_index` order. The
    consumer (audio callback via `DrumSampler`) pops events whose `frame_index` falls before the
    end of the current audio block. If the ring is full, `push` drops the new event and counts it
    in `dropped_events` without overwriting unread slots.
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._frames: list[int] = [0] * capacity
        self._pieces: list[int] = [0] * capacity
        self._gains: list[float] = [0.0] * capacity
        self._write_pos = 0
        self._read_pos = 0
        self.dropped_events = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def pending_count(self) -> int:
        """Number of unread events currently in the ring."""
        return self._write_pos - self._read_pos

    def push(self, frame_index: SampleFrame, piece: DrumPiece, gain: float = 1.0) -> bool:
        """Producer: enqueue a drum hit. Returns False and increments `dropped_events` if full."""
        write_pos = self._write_pos
        if write_pos - self._read_pos >= self._capacity:
            self.dropped_events += 1
            return False
        slot = write_pos % self._capacity
        self._frames[slot] = int(frame_index)
        self._pieces[slot] = int(piece)
        self._gains[slot] = float(gain)
        self._write_pos = write_pos + 1
        return True

    def pop_due(self, before_frame: SampleFrame) -> list[DrumEvent]:
        """Consumer helper: pop and return all events with `frame_index < before_frame`."""
        due: list[DrumEvent] = []
        read_pos = self._read_pos
        write_pos = self._write_pos
        while read_pos < write_pos:
            slot = read_pos % self._capacity
            event_frame = self._frames[slot]
            if event_frame >= before_frame:
                break
            due.append(
                DrumEvent(
                    frame_index=event_frame,
                    piece=DrumPiece(self._pieces[slot]),
                    gain=self._gains[slot],
                )
            )
            read_pos += 1
        self._read_pos = read_pos
        return due


class DrumSampler:
    """Polyphonic sample player with a preallocated voice pool for the audio callback (§45, §46)."""

    def __init__(
        self,
        samples: dict[DrumPiece, np.ndarray],
        max_voices: int = 16,
        event_ring_capacity: int = 256,
        max_block_frames: int = MAX_RENDER_BLOCK_FRAMES,
    ) -> None:
        if max_voices <= 0:
            raise ValueError("max_voices must be positive")
        for piece in DrumPiece:
            if piece not in samples:
                raise ValueError(f"missing sample for {piece.name}")
            if samples[piece].ndim != 1 or samples[piece].size == 0:
                raise ValueError(f"sample for {piece.name} must be a non-empty 1D array")

        self._samples: tuple[np.ndarray, ...] = tuple(
            np.ascontiguousarray(samples[DrumPiece(i)], dtype=np.float32)
            for i in range(len(DrumPiece))
        )
        self._max_voices = max_voices
        self.event_ring = DrumEventRing(event_ring_capacity)

        # Preallocated voice pool (owned by the audio callback thread).
        self._active: list[bool] = [False] * max_voices
        self._piece: list[int] = [0] * max_voices
        self._gain: list[float] = [0.0] * max_voices
        self._sample_pos: list[int] = [0] * max_voices
        self._trigger_seq: list[int] = [0] * max_voices
        self._next_seq = 0
        self._mono_scratch = np.zeros(max_block_frames, dtype=np.float32)
        get_native_lib()

        # Callback-owned counters.
        self.events_played = 0
        self.late_events = 0
        self.stolen_voices = 0

    @classmethod
    def from_config(cls, config: DrumConfig, sample_rate_hz: int = 48_000) -> Self:
        """Load the kit from `config.kit_dir` at `sample_rate_hz` and build a sampler."""
        kit = load_kit(config.kit_dir, sample_rate_hz)
        return cls(
            samples=kit,
            max_voices=config.max_voices,
            event_ring_capacity=config.event_ring_capacity,
        )

    @property
    def max_voices(self) -> int:
        return self._max_voices

    def active_voice_count(self) -> int:
        """Number of currently playing voices."""
        count = 0
        for active in self._active:
            if active:
                count += 1
        return count

    def schedule(self, frame_index: SampleFrame, piece: DrumPiece, gain: float = 1.0) -> bool:
        """Convenience wrapper around `self.event_ring.push`."""
        return self.event_ring.push(frame_index, piece, gain)

    def render_block(self, out: np.ndarray, block_start_frame: SampleFrame) -> None:
        """Render one audio block `[block_start_frame, block_start_frame + len(out))` into `out`.

        `out` is zero-filled first, then active voices and due events are mixed in place at their
        exact intra-block sample offsets. Supports 1D `(frames,)` or 2D `(frames, channels)`
        output buffers with zero allocations.
        """
        out.fill(0.0)
        block_frames = out.shape[0]
        block_end_frame = block_start_frame + block_frames
        cursor = 0

        ring = self.event_ring
        read_pos = ring._read_pos
        write_pos = ring._write_pos
        capacity = ring._capacity
        frames_buf = ring._frames
        pieces_buf = ring._pieces
        gains_buf = ring._gains

        while read_pos < write_pos:
            slot = read_pos % capacity
            event_frame = frames_buf[slot]
            if event_frame >= block_end_frame:
                break
            piece = pieces_buf[slot]
            gain = gains_buf[slot]
            read_pos += 1
            if event_frame < block_start_frame:
                offset = 0
                self.late_events += 1
            else:
                offset = event_frame - block_start_frame
            if offset > cursor:
                self._mix_slice(out, cursor, offset)
                cursor = offset
            if gain > 0.0:
                self._start_voice(piece, gain)
            self.events_played += 1

        ring._read_pos = read_pos
        if cursor < block_frames:
            self._mix_slice(out, cursor, block_frames)

    def _mix_slice(self, out: np.ndarray, start: int, end: int) -> None:
        mix_drum_voices_f32(
            self._samples,
            self._active,
            self._piece,
            self._gain,
            self._sample_pos,
            out,
            start,
            end,
            mono_scratch=self._mono_scratch,
        )

    def _start_voice(self, piece: int, gain: float) -> None:
        chosen = -1
        for i in range(self._max_voices):
            if not self._active[i]:
                chosen = i
                break
        if chosen < 0:
            # All voices are active: steal the oldest triggered voice.
            oldest_seq = self._trigger_seq[0]
            chosen = 0
            for i in range(1, self._max_voices):
                if self._trigger_seq[i] < oldest_seq:
                    oldest_seq = self._trigger_seq[i]
                    chosen = i
            self.stolen_voices += 1

        self._next_seq += 1
        self._active[chosen] = True
        self._piece[chosen] = piece
        self._gain[chosen] = gain
        self._sample_pos[chosen] = 0
        self._trigger_seq[chosen] = self._next_seq

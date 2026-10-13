"""The internal musical clock: sample-frame ↔ (bar, beat, phase) and PLL correction (§31-§33).

Once the tempo locks (or a manual `--bpm` override is set), `BeatClock` anchors a 4/4 grid to the
audio stream's sample-frame clock (D5). Downstream components (Musical State, the drum scheduler,
AI entry, the UI beat dots) read an immutable `BeatClockSnapshot` and convert between sample
frames and musical positions without locks.

Design rules (§31-§33):
- Pure math: `frame_to_beat`, `beat_to_frame`, `next_beat_frame`, `next_bar_frame` and
  `align_anchor` are pure functions of sample frames and BPM. Beat `k` sits at
  `round(anchor_frame + k * frames_per_beat)`, so there is zero cumulative rounding drift over
  arbitrary durations.
- Start-up alignment: when the tempo first locks, `align_anchor` picks the downbeat candidate
  from the recent onsets that maximizes bar-cycle and beat-grid support (weighted by attack
  strength, with a tie-breaker for the phrase start), then refines its phase across all aligned
  onsets.
- Gradual synchronization (§33):
  - Each onset nudges `anchor_frame` by a bounded fraction of its timing error relative to the
    beat (or half-beat subdivision for off-beat 8th notes), acting as a first-order PLL.
  - Locked tempo updates are smoothed and adjust `anchor_frame` at the update's sample frame so
    that the continuous beat position is unchanged (no phase or bar jumps on BPM jitter).
  - Re-anchoring happens only when a large beat-phase error persists across `reanchor_onsets`
    consecutive onsets over at least `reanchor_seconds`.
"""

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jammate.config import BeatClockConfig
from jammate.music.events import (
    MusicalEvent,
    OnsetEvent,
    SampleFrame,
    TempoUpdate,
    frames_to_seconds,
    seconds_to_frames,
)

logger = logging.getLogger(__name__)

MAX_RECENT_ONSETS = 128
RECENT_ONSET_WINDOW_SECONDS = 10.0


@dataclass(frozen=True, kw_only=True)
class BeatPosition:
    """Musical position at a sample frame (§31, §32).

    `bar` is 1-based (1, 2, … from the anchor), `beat` is in `1..beats_per_bar`, and
    `beat_phase` is in `[0.0, 1.0)`. `total_beats` is the continuous beat count from the anchor
    (`(bar - 1) * beats_per_bar + (beat - 1) + beat_phase`).
    """

    bar: int
    beat: int
    beat_phase: float
    total_beats: float


@dataclass(frozen=True, kw_only=True)
class BeatClockSnapshot:
    """Immutable snapshot of a running `BeatClock`, safe to read from any thread."""

    anchor_frame: float
    bpm: float
    beats_per_bar: int = 4
    sample_rate_hz: int = 48_000

    @property
    def frames_per_beat(self) -> float:
        return frames_per_beat(self.bpm, self.sample_rate_hz)

    @property
    def frames_per_bar(self) -> float:
        return self.beats_per_bar * self.frames_per_beat

    def position_at(self, frame_index: SampleFrame) -> BeatPosition:
        return frame_to_beat(
            frame_index,
            self.anchor_frame,
            self.bpm,
            self.sample_rate_hz,
            self.beats_per_bar,
        )

    def beat_to_frame(self, bar: int, beat: int, beat_phase: float = 0.0) -> SampleFrame:
        return beat_to_frame(
            bar,
            beat,
            beat_phase,
            anchor_frame=self.anchor_frame,
            bpm=self.bpm,
            sample_rate_hz=self.sample_rate_hz,
            beats_per_bar=self.beats_per_bar,
        )

    def next_beat_frame(self, frame_index: SampleFrame) -> SampleFrame:
        return next_beat_frame(
            frame_index,
            self.anchor_frame,
            self.bpm,
            self.sample_rate_hz,
        )

    def next_bar_frame(self, frame_index: SampleFrame) -> SampleFrame:
        return next_bar_frame(
            frame_index,
            self.anchor_frame,
            self.bpm,
            self.sample_rate_hz,
            self.beats_per_bar,
        )


def frames_per_beat(bpm: float, sample_rate_hz: int) -> float:
    """Exact (floating-point) sample frames per beat."""
    _check_clock_params(bpm, sample_rate_hz, beats_per_bar=4)
    return 60.0 * sample_rate_hz / bpm


def beat_to_frame(
    bar: int,
    beat: int,
    beat_phase: float = 0.0,
    *,
    anchor_frame: float,
    bpm: float,
    sample_rate_hz: int,
    beats_per_bar: int = 4,
) -> SampleFrame:
    """Convert `(bar, beat, beat_phase)` to the nearest whole sample frame (halves round up)."""
    _check_clock_params(bpm, sample_rate_hz, beats_per_bar)
    if not 1 <= beat <= beats_per_bar:
        raise ValueError(f"beat must be in 1..{beats_per_bar}, got {beat}")
    if not (math.isfinite(beat_phase) and 0.0 <= beat_phase < 1.0):
        raise ValueError(f"beat_phase must be in [0.0, 1.0), got {beat_phase}")
    total_beats = (bar - 1) * beats_per_bar + (beat - 1) + beat_phase
    fpb = 60.0 * sample_rate_hz / bpm
    return math.floor(anchor_frame + total_beats * fpb + 0.5)


def frame_to_beat(
    frame_index: SampleFrame,
    anchor_frame: float,
    bpm: float,
    sample_rate_hz: int,
    beats_per_bar: int = 4,
) -> BeatPosition:
    """Convert a sample `frame_index` into `(bar, beat, beat_phase)` on the grid."""
    _check_clock_params(bpm, sample_rate_hz, beats_per_bar)
    fpb = 60.0 * sample_rate_hz / bpm
    beat_index = _beat_index_at(frame_index, anchor_frame, fpb)
    start_frame = _beat_index_to_frame(beat_index, anchor_frame, fpb)
    end_frame = _beat_index_to_frame(beat_index + 1, anchor_frame, fpb)
    span = max(1, end_frame - start_frame)
    beat_phase = (frame_index - start_frame) / span
    bar = beat_index // beats_per_bar + 1
    beat = beat_index % beats_per_bar + 1
    return BeatPosition(
        bar=bar,
        beat=beat,
        beat_phase=beat_phase,
        total_beats=beat_index + beat_phase,
    )


def next_beat_frame(
    frame_index: SampleFrame,
    anchor_frame: float,
    bpm: float,
    sample_rate_hz: int,
) -> SampleFrame:
    """Sample frame of the first beat boundary strictly after `frame_index`."""
    _check_clock_params(bpm, sample_rate_hz, beats_per_bar=4)
    fpb = 60.0 * sample_rate_hz / bpm
    beat_index = _beat_index_at(frame_index, anchor_frame, fpb)
    return _beat_index_to_frame(beat_index + 1, anchor_frame, fpb)


def next_bar_frame(
    frame_index: SampleFrame,
    anchor_frame: float,
    bpm: float,
    sample_rate_hz: int,
    beats_per_bar: int = 4,
) -> SampleFrame:
    """Sample frame of the first bar boundary (beat 1) strictly after `frame_index`."""
    pos = frame_to_beat(frame_index, anchor_frame, bpm, sample_rate_hz, beats_per_bar)
    return beat_to_frame(
        pos.bar + 1,
        1,
        0.0,
        anchor_frame=anchor_frame,
        bpm=bpm,
        sample_rate_hz=sample_rate_hz,
        beats_per_bar=beats_per_bar,
    )


def align_anchor(
    onsets: Sequence[OnsetEvent],
    bpm: float,
    sample_rate_hz: int,
    *,
    beats_per_bar: int = 4,
    tolerance_seconds: float = 0.03,
    fallback_frame: SampleFrame = 0,
) -> float:
    """Pick the Bar 1 Beat 1 anchor frame from recent `onsets` at `bpm`.

    Scores each onset as a candidate downbeat by how strongly and repeatedly onsets land on its
    bar cycle and beat grid (with a small tie-breaker for the first onset in the phrase), then
    refines the winning phase across all aligned onsets.
    """
    _check_clock_params(bpm, sample_rate_hz, beats_per_bar)
    if not onsets:
        return float(fallback_frame)

    fpb = 60.0 * sample_rate_hz / bpm
    bar_frames = beats_per_bar * fpb
    tol_frames = max(1.0, tolerance_seconds * sample_rate_hz)

    frames = np.array([onset.frame_index for onset in onsets], dtype=np.float64)
    strengths = np.array([max(onset.strength, 1e-3) for onset in onsets], dtype=np.float64)

    # Find the shared sub-beat (half-beat) phase across all onsets via circular mean, so no
    # candidate is penalized by a single onset's human timing jitter.
    half_fpb = 0.5 * fpb
    phases = np.exp(2j * np.pi * (frames - frames[0]) / half_fpb)
    mean_angle = float(np.angle(np.dot(phases, strengths)))
    base_frame = float(frames[0]) + (mean_angle / (2.0 * np.pi)) * half_fpb

    # Assign each onset near the half-beat grid to its half-beat slot in the bar (0..2*B-1).
    num_candidates = 2 * beats_per_bar
    sub_steps = np.round((frames - base_frame) / half_fpb).astype(int)
    sub_residuals = np.abs(frames - (base_frame + sub_steps * half_fpb))
    aligned = sub_residuals <= 2.5 * tol_frames
    slot_strength = np.zeros(num_candidates, dtype=np.float64)
    if np.any(aligned):
        np.add.at(slot_strength, sub_steps[aligned] % num_candidates, strengths[aligned])
    else:
        np.add.at(slot_strength, sub_steps % num_candidates, strengths)

    beat_support = np.array(
        [
            sum(slot_strength[(k + 2 * b) % num_candidates] for b in range(beats_per_bar))
            for k in range(num_candidates)
        ],
        dtype=np.float64,
    )
    phrase_start_bonus = np.zeros(num_candidates, dtype=np.float64)
    phrase_start_bonus[0] = 0.2 * strengths[0]

    total_score = slot_strength + 0.5 * beat_support + phrase_start_bonus
    best_slot = int(np.argmax(total_score))
    best_frame = base_frame + best_slot * half_fpb

    # Place Bar 1 Beat 1 at the downbeat at or just before the earliest onset.
    bars_after_first = math.ceil((best_frame - float(frames[0]) - tol_frames) / bar_frames)
    anchor = best_frame - max(0, bars_after_first) * bar_frames
    return anchor


class BeatClock:
    """Stateful musical clock driven on the analysis worker thread (§31-§33).

    Publishes an immutable `BeatClockSnapshot` on every change so other threads can read
    `snapshot` without locking.
    """

    def __init__(
        self,
        config: BeatClockConfig,
        sample_rate_hz: int,
        *,
        manual_bpm: float | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._manual_bpm = manual_bpm
        self._onsets: deque[OnsetEvent] = deque(maxlen=MAX_RECENT_ONSETS)
        self._snapshot: BeatClockSnapshot | None = None
        self._consecutive_large_errors = 0
        self._large_error_since_seconds: float | None = None
        self.corrections = 0
        self.reanchors = 0
        self.last_correction_seconds = 0.0
        if manual_bpm is not None:
            self.start_at(manual_bpm, anchor_frame=0.0)

    @property
    def snapshot(self) -> BeatClockSnapshot | None:
        """Current immutable clock snapshot, or None while unlocked. Safe on any thread."""
        return self._snapshot

    @property
    def is_running(self) -> bool:
        return self._snapshot is not None

    @property
    def bpm(self) -> float | None:
        snap = self._snapshot
        return None if snap is None else snap.bpm

    @property
    def anchor_frame(self) -> float | None:
        snap = self._snapshot
        return None if snap is None else snap.anchor_frame

    def start_at(self, bpm: float, anchor_frame: float = 0.0) -> BeatClockSnapshot:
        """Start (or re-anchor) the clock explicitly at `bpm` and `anchor_frame`."""
        _check_clock_params(bpm, self._sample_rate_hz, self._config.beats_per_bar)
        self._consecutive_large_errors = 0
        self._large_error_since_seconds = None
        snap = BeatClockSnapshot(
            anchor_frame=float(anchor_frame),
            bpm=float(bpm),
            beats_per_bar=self._config.beats_per_bar,
            sample_rate_hz=self._sample_rate_hz,
        )
        self._snapshot = snap
        return snap

    def reset(self) -> None:
        """Stop the clock when the tempo unlocks."""
        self._snapshot = None
        self._consecutive_large_errors = 0
        self._large_error_since_seconds = None

    def position_at(self, frame_index: SampleFrame) -> BeatPosition | None:
        """Musical position at `frame_index`, or None while the clock is not running."""
        snap = self._snapshot
        return None if snap is None else snap.position_at(frame_index)

    def handle_event(self, event: MusicalEvent) -> None:
        """Feed an analysis event (`OnsetEvent` or `TempoUpdate`)."""
        if isinstance(event, OnsetEvent):
            self.on_onset(event)
        elif isinstance(event, TempoUpdate):
            self.on_tempo(event)

    def on_onset(self, onset: OnsetEvent) -> float:
        """Record `onset` and, if running, apply bounded PLL phase correction (§33).

        Returns the phase shift applied to `anchor_frame` in seconds (0.0 if not running).
        """
        self._onsets.append(onset)
        cutoff_frame = onset.frame_index - seconds_to_frames(
            RECENT_ONSET_WINDOW_SECONDS, self._sample_rate_hz
        )
        while self._onsets and self._onsets[0].frame_index < cutoff_frame:
            self._onsets.popleft()

        snap = self._snapshot
        if snap is None or self._manual_bpm is not None:
            return 0.0

        fpb = snap.frames_per_beat
        total_beats = (onset.frame_index - snap.anchor_frame) / fpb
        beat_error_beats = ((total_beats + 0.5) % 1.0) - 0.5

        if abs(beat_error_beats) > self._config.reanchor_error_beats:
            self._consecutive_large_errors += 1
            if self._large_error_since_seconds is None:
                self._large_error_since_seconds = onset.timestamp_seconds
            elapsed = onset.timestamp_seconds - self._large_error_since_seconds
            if (
                self._consecutive_large_errors >= self._config.reanchor_onsets
                and elapsed >= self._config.reanchor_seconds
            ):
                return self._reanchor_from_recent(snap, onset.frame_index)
            # Single off-beat note (e.g. an 8th note on the "and"): measure microtiming error
            # against the half-beat subdivision so 8ths never pull the beat toward the off-beat.
            error_beats = ((total_beats + 0.25) % 0.5) - 0.25
        else:
            self._consecutive_large_errors = 0
            self._large_error_since_seconds = None
            error_beats = beat_error_beats

        error_frames = error_beats * fpb
        max_frames = self._config.max_phase_correction_seconds * self._sample_rate_hz
        correction_frames = max(
            -max_frames,
            min(max_frames, self._config.phase_correction_gain * error_frames),
        )
        self._snapshot = BeatClockSnapshot(
            anchor_frame=snap.anchor_frame + correction_frames,
            bpm=snap.bpm,
            beats_per_bar=snap.beats_per_bar,
            sample_rate_hz=snap.sample_rate_hz,
        )
        self.corrections += 1
        self.last_correction_seconds = frames_to_seconds(
            round(correction_frames), self._sample_rate_hz
        )
        return correction_frames / self._sample_rate_hz

    def on_tempo(self, update: TempoUpdate) -> None:
        """Start the clock when tempo locks, smooth BPM while locked, or reset on unlock (§33)."""
        if self._manual_bpm is not None:
            return
        if not update.locked or update.bpm is None:
            if self._snapshot is not None:
                logger.info("BeatClock stopped (tempo unlocked).")
            self.reset()
            return

        snap = self._snapshot
        if snap is None:
            anchor = align_anchor(
                list(self._onsets),
                update.bpm,
                self._sample_rate_hz,
                beats_per_bar=self._config.beats_per_bar,
                tolerance_seconds=self._config.alignment_tolerance_seconds,
                fallback_frame=update.frame_index,
            )
            self.start_at(update.bpm, anchor)
            logger.info(
                "BeatClock started at %.1f BPM (anchor frame %.0f).",
                update.bpm,
                anchor,
            )
            return

        # Smooth BPM and adjust anchor_frame so the continuous beat count at update.frame_index
        # does not jump (§33).
        new_bpm = snap.bpm + self._config.tempo_smoothing * (update.bpm - snap.bpm)
        old_fpb = snap.frames_per_beat
        new_fpb = 60.0 * self._sample_rate_hz / new_bpm
        total_beats = (update.frame_index - snap.anchor_frame) / old_fpb
        new_anchor = update.frame_index - total_beats * new_fpb
        self._snapshot = BeatClockSnapshot(
            anchor_frame=new_anchor,
            bpm=new_bpm,
            beats_per_bar=snap.beats_per_bar,
            sample_rate_hz=snap.sample_rate_hz,
        )

    def _reanchor_from_recent(self, snap: BeatClockSnapshot, current_frame: SampleFrame) -> float:
        recent = list(self._onsets)[-self._config.reanchor_onsets :]
        new_anchor_raw = align_anchor(
            recent,
            snap.bpm,
            self._sample_rate_hz,
            beats_per_bar=snap.beats_per_bar,
            tolerance_seconds=self._config.alignment_tolerance_seconds,
            fallback_frame=current_frame,
        )
        # Preserve the current bar number so bar counting does not reset to 1 mid-jam: shift the
        # new anchor by whole bars to stay within half a bar of the previous anchor.
        bar_frames = snap.frames_per_bar
        delta_frames = ((new_anchor_raw - snap.anchor_frame + 0.5 * bar_frames) % bar_frames) - (
            0.5 * bar_frames
        )
        new_anchor = snap.anchor_frame + delta_frames
        self._snapshot = BeatClockSnapshot(
            anchor_frame=new_anchor,
            bpm=snap.bpm,
            beats_per_bar=snap.beats_per_bar,
            sample_rate_hz=snap.sample_rate_hz,
        )
        self._consecutive_large_errors = 0
        self._large_error_since_seconds = None
        self.reanchors += 1
        shift_seconds = delta_frames / self._sample_rate_hz
        self.last_correction_seconds = shift_seconds
        logger.info("BeatClock re-anchored by %+.1f ms.", shift_seconds * 1000.0)
        return shift_seconds


def _beat_index_to_frame(beat_index: int, anchor_frame: float, fpb: float) -> SampleFrame:
    return math.floor(anchor_frame + beat_index * fpb + 0.5)


def _beat_index_at(frame_index: SampleFrame, anchor_frame: float, fpb: float) -> int:
    beat_index = math.floor((frame_index - anchor_frame) / fpb)
    if frame_index >= _beat_index_to_frame(beat_index + 1, anchor_frame, fpb):
        beat_index += 1
    elif frame_index < _beat_index_to_frame(beat_index, anchor_frame, fpb):
        beat_index -= 1
    return beat_index


def _check_clock_params(bpm: float, sample_rate_hz: int, beats_per_bar: int) -> None:
    if not (math.isfinite(bpm) and bpm > 0):
        raise ValueError(f"bpm must be positive, got {bpm}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if beats_per_bar < 1:
        raise ValueError(f"beats_per_bar must be >= 1, got {beats_per_bar}")

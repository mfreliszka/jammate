"""Manual escape hatches, tap tempo, and multi-meter helpers (Step 6.1, §58, §149).

Provides:
- `TapTempoTracker`: collects monotonic tap timestamps (`3..8` taps, auto-reset after `2.5 s`
  idle), rejects outlier inter-tap intervals, computes BPM clamped to `min_bpm..max_bpm`, and
  aligns `BeatClock` beat-1 phase to the latest tap.
- `ManualOverrides` frozen state (`auto_bpm`, `manual_bpm`, `meter`, `manual_key`).
- Multi-meter helpers (`"4/4"`, `"3/4"`, `"6/8"`, `parse_meter`, `format_meter`,
  `meter_beats_per_bar`, `meter_steps_per_bar`).
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from jammate.analysis.key import parse_key_string
from jammate.analysis.notes import NOTE_NAMES
from jammate.music.beat_clock import BeatClock, BeatClockSnapshot
from jammate.music.events import SampleFrame

type MeterName = Literal["4/4", "3/4", "6/8"]
type MeterTuple = tuple[int, int]

SUPPORTED_METERS: tuple[MeterName, ...] = ("4/4", "3/4", "6/8")
METER_TUPLES: dict[MeterName, MeterTuple] = {
    "4/4": (4, 4),
    "3/4": (3, 4),
    "6/8": (6, 8),
}
METER_BEATS_PER_BAR: dict[MeterName, int] = {
    "4/4": 4,
    "3/4": 3,
    "6/8": 6,
}
METER_STEPS_PER_BAR: dict[MeterName, int] = {
    "4/4": 16,
    "3/4": 12,
    "6/8": 12,
}

# Standard UI dropdown options for key override (`None` = Auto, plus 12 pitch classes x minor/major)
KEY_OVERRIDE_OPTIONS: tuple[tuple[str, str | None], ...] = (
    ("Auto", None),
    *tuple(
        item
        for note in ("E", "A", "D", "G", "C", "F", "F#", "G#", "A#", "B", "C#", "D#")
        for item in ((f"{note} minor", f"{note} minor"), (f"{note} major", f"{note} major"))
        if note in NOTE_NAMES
    ),
)


def parse_meter(meter: str | tuple[int, int]) -> MeterTuple:
    """Parse `'4/4'`, `'3/4'`, `'6/8'` or `(num, den)` into a validated `MeterTuple`."""
    if isinstance(meter, tuple):
        if len(meter) == 2 and meter in METER_TUPLES.values():
            return (int(meter[0]), int(meter[1]))
        raise ValueError(f"unsupported meter {meter!r}; expected one of (4, 4), (3, 4), (6, 8)")
    cleaned = meter.strip()
    if cleaned in METER_TUPLES:
        return METER_TUPLES[cleaned]  # type: ignore[index]
    raise ValueError(f"unsupported meter {meter!r}; expected one of {SUPPORTED_METERS}")


def format_meter(meter: str | tuple[int, int]) -> MeterName:
    """Format a meter string or `(num, den)` tuple as `'4/4'`, `'3/4'`, or `'6/8'`."""
    num, den = parse_meter(meter)
    formatted = f"{num}/{den}"
    if formatted in METER_TUPLES:
        return formatted  # type: ignore[return-value]
    raise ValueError(f"unsupported meter {meter!r}")


def meter_beats_per_bar(meter: str | tuple[int, int]) -> int:
    """Return `beats_per_bar` (`4` for `4/4`, `3` for `3/4`, `6` for `6/8`)."""
    return METER_BEATS_PER_BAR[format_meter(meter)]


def meter_steps_per_bar(meter: str | tuple[int, int]) -> int:
    """Return grid steps per bar (`16` for `4/4`, `12` for `3/4` and `6/8`)."""
    return METER_STEPS_PER_BAR[format_meter(meter)]


def validate_manual_key(manual_key: str | None) -> str | None:
    """Validate and normalize a manual key string (`'E minor'`, `'A major'`, or `None`)."""
    if manual_key is None or not manual_key.strip() or manual_key.strip().lower() == "auto":
        return None
    _pc, key_name, mode = parse_key_string(manual_key)
    return f"{key_name} {mode}"


@dataclass(frozen=True, slots=True)
class ManualOverrides:
    """Immutable snapshot of manual studio overrides (Step 6.1, §58, §149)."""

    bpm: float | None = None
    meter: MeterTuple = (4, 4)
    key: str | None = None
    auto_bpm: bool = True

    def __post_init__(self) -> None:
        norm_meter = parse_meter(self.meter)
        object.__setattr__(self, "meter", norm_meter)
        if self.key is not None:
            parsed = validate_manual_key(self.key)
            object.__setattr__(self, "key", parsed)
        if self.bpm is not None:
            object.__setattr__(self, "auto_bpm", False)

    @property
    def manual_bpm(self) -> float | None:
        return self.bpm

    @property
    def manual_key(self) -> str | None:
        return self.key

    @property
    def meter_name(self) -> MeterName:
        return format_meter(self.meter)

    @property
    def beats_per_bar(self) -> int:
        return meter_beats_per_bar(self.meter)

    @property
    def steps_per_bar(self) -> int:
        return meter_steps_per_bar(self.meter)

    @property
    def effective_manual_bpm(self) -> float | None:
        return None if self.auto_bpm else self.bpm


class TapTempoTracker:
    """Collects monotonic tap timestamps (`2..8` taps) and computes robust BPM (§58, §149).

    - Automatically resets tap history if the gap since the previous tap exceeds
      `idle_reset_seconds` (`2.5 s`).
    - Rejects outlier inter-tap intervals deviating by more than `outlier_tolerance_ratio`
      from the median interval when `>= 3` intervals are present.
    - Clamps the resulting BPM to `[min_bpm, max_bpm]`.
    """

    def __init__(
        self,
        *,
        min_taps: int = 2,
        max_taps: int = 8,
        idle_reset_seconds: float = 2.5,
        max_gap_seconds: float | None = None,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        outlier_tolerance_ratio: float = 0.35,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_gap_seconds is not None:
            idle_reset_seconds = max_gap_seconds
        if min_taps < 2:
            raise ValueError(f"min_taps must be >= 2, got {min_taps}")
        if max_taps < min_taps:
            raise ValueError(f"max_taps ({max_taps}) must be >= min_taps ({min_taps})")
        if idle_reset_seconds <= 0.0:
            raise ValueError(f"idle_reset_seconds must be > 0, got {idle_reset_seconds}")
        if not (0.0 < min_bpm < max_bpm):
            raise ValueError(f"invalid BPM bounds: {min_bpm}..{max_bpm}")
        self._min_taps = min_taps
        self._max_taps = max_taps
        self._idle_reset_seconds = idle_reset_seconds
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._outlier_tolerance_ratio = outlier_tolerance_ratio
        self._monotonic = monotonic
        self._taps: deque[float] = deque(maxlen=max_taps)
        self._estimated_bpm: float | None = None

    @property
    def taps(self) -> tuple[float, ...]:
        return tuple(self._taps)

    @property
    def tap_count(self) -> int:
        return len(self._taps)

    @property
    def estimated_bpm(self) -> float | None:
        return self._estimated_bpm

    @property
    def current_bpm(self) -> float | None:
        return self._estimated_bpm

    def reset(self) -> None:
        """Clear all buffered tap timestamps and the estimated BPM."""
        self._taps.clear()
        self._estimated_bpm = None

    def tap(self, timestamp_seconds: float | None = None) -> float | None:
        """Record a tap and return the clamped BPM once `>= min_taps` are collected."""
        now = float(self._monotonic() if timestamp_seconds is None else timestamp_seconds)
        if self._taps:
            gap = now - self._taps[-1]
            if gap <= 0.0 or gap > self._idle_reset_seconds:
                self._taps.clear()
                self._estimated_bpm = None

        self._taps.append(now)
        if len(self._taps) < self._min_taps:
            self._estimated_bpm = None
            return None

        tap_list = list(self._taps)
        intervals = [tap_list[i + 1] - tap_list[i] for i in range(len(tap_list) - 1)]
        med = statistics.median(intervals)
        if len(intervals) >= 3 and med > 1e-6:
            inliers = [
                dt for dt in intervals if abs(dt - med) / med <= self._outlier_tolerance_ratio
            ]
            if len(inliers) >= 2:
                intervals = inliers

        mean_dt = sum(intervals) / len(intervals)
        if mean_dt <= 1e-6:
            return self._estimated_bpm
        raw_bpm = 60.0 / mean_dt
        self._estimated_bpm = max(self._min_bpm, min(self._max_bpm, raw_bpm))
        return self._estimated_bpm

    def apply_tap_to_clock(
        self,
        clock: BeatClock,
        frame_index: SampleFrame,
        timestamp_seconds: float | None = None,
    ) -> BeatClockSnapshot | None:
        """Record a tap and, once locked, align `clock` beat-1 phase to `frame_index`."""
        bpm = self.tap(timestamp_seconds)
        if bpm is None:
            return None
        return clock.set_manual_bpm(bpm, anchor_frame=float(frame_index), current_frame=frame_index)

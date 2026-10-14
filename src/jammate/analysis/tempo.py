"""Tempo estimation and locking (§28, §29).

Runs on the analysis worker thread, once per onset; it never sees audio. Estimation is a pure
function of onset times in seconds (`estimate_tempo`), so tests drive it with synthetic onset
lists. `TempoTracker` adds the lock/unlock hysteresis and the gradual adaptation.

Estimation looks at the onsets of the last `window_seconds`. Their weights fall off with age
(`recency_seconds`), so the evidence rolls forward with the playing. For a grid of period p, it
measures two things:

- Alignment (`mean_resultants`). Every onset becomes a phase on a circle of circumference p. The
  alignment is the length of the onsets' weighted mean resultant, |sum w·exp(2πi·t/p)| / sum w:
  1 when all onsets sit on one grid of period p, near 0 when they spread around it. Squared, it
  is the sum over all pairs of onsets of w_i·w_j·cos(2π·Δt/p), so inter-onset intervals that are
  whole multiples of p raise it and the others lower it. The resultant's angle is the grid's
  phase.
- Fit (`grid_fit`). The weighted share of onsets within about `timing_tolerance_seconds` of that
  phased grid, minus the share that random onsets would reach: a fine grid is hit by chance, so
  it can score little. Its tolerance is absolute, like human timing errors. The alignment's is
  relative: an onset 77 ms off a 0.6 s grid still counts 0.7 there. So by alignment, a grid of
  three beats per bar fits a 3+3+2 riff almost as well as the real one; by fit it doesn't.

Candidates, with half/double normalization. A beat period B is scored by the summed fit of its
grid and of its 8th, 16th and 32nd subdivisions (B, B/2, B/4, B/8, down to MIN_PULSE_SECONDS).
Whichever of these the riff fills, its grid shows up in the sum; a period that fits only by
coincidence (3:2 against the played grid, say) gets little support at any level. Beats are scored
only within one octave around a reference tempo, so B never competes with 2B or B/2, which fit
8th-note playing equally well: the octave is the normalization (D9). Before a lock the reference
is `preferred_bpm`, so 8th notes at 92 BPM read as 92, not 184. Once locked it is the locked
tempo, so a half-time passage can't halve it.

The best beat is interpolated between candidate steps: on the recorded riffs, within ±1 BPM. Its
`clarity` is the alignment at its best-filled level: 0.6-0.97 on the recorded riffs and chords.
A few random onsets always fit some grid, so `significance` (effective onset count x clarity
squared, the Rayleigh statistic) also says how many onsets back the estimate.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from jammate.config import AnalysisConfig, TempoConfig
from jammate.music.events import OnsetEvent, SampleFrame, TempoUpdate, frames_to_seconds

STEPS_PER_OCTAVE = 144  # beat candidates 0.5% apart
EDGE_STEPS = 1  # a candidate past each end of the octave, so a peak at its edge interpolates
GRID_LEVELS = 4  # the beat and its 8th, 16th and 32nd subdivisions
MIN_PULSE_SECONDS = 0.07  # finer grids drown in timing jitter (16ths at 214 BPM)
MAX_ONSETS_PER_SECOND = 16  # bounds the onset history (onsets are at least 80 ms apart)


@dataclass(frozen=True)
class TempoEstimate:
    bpm: float
    clarity: float  # 0-1: alignment of the onsets with the beat's grid at its best-filled level
    significance: float  # effective onset count x clarity squared


@dataclass(frozen=True)
class TempoState:
    """What the tracker currently believes (§28). `bpm` is None until the first estimate."""

    bpm: float | None
    confidence: float
    locked: bool


def mean_resultants(
    onset_seconds: np.ndarray, weights: np.ndarray, periods_seconds: np.ndarray
) -> np.ndarray:
    """Per period, the weighted mean of exp(2πi·t/p) over the onsets. Its length is the grid
    alignment (0-1), its angle the grid's phase."""
    phases = np.exp(np.outer(2j * np.pi / periods_seconds, onset_seconds))
    return (phases @ weights) / np.sum(weights)


def grid_fit(
    onset_seconds: np.ndarray,
    weights: np.ndarray,
    periods_seconds: np.ndarray,
    resultants: np.ndarray,
    tolerance_seconds: float,
) -> np.ndarray:
    """Per period, the weighted share of onsets near the grid of that period, phased by
    `resultants` (from `mean_resultants`), beyond the share that random onsets would reach.
    Between 0 (no better than chance) and 1 minus that chance share."""
    grid_seconds = np.angle(resultants) / (2.0 * np.pi) * periods_seconds
    half_periods = periods_seconds[:, np.newaxis] / 2.0
    offsets = onset_seconds - grid_seconds[:, np.newaxis] + half_periods
    offsets = offsets % (2.0 * half_periods) - half_periods
    near = np.exp(-0.5 * (offsets / tolerance_seconds) ** 2) @ weights / np.sum(weights)
    # The expected `near` of an onset at a random position.
    chance = (
        math.sqrt(2.0 * math.pi)
        * tolerance_seconds
        / periods_seconds
        * erf(periods_seconds / (2.0 * math.sqrt(2.0) * tolerance_seconds))
    )
    return np.maximum(near - chance, 0.0)


def estimate_tempo(
    onset_seconds: np.ndarray, reference_bpm: float, config: TempoConfig
) -> TempoEstimate | None:
    """The tempo of `onset_seconds` (ascending; the last one is now) in the octave around
    `reference_bpm`, or None while the recent onsets are too few or too sparse to show a pulse.
    See the module docstring."""
    now_seconds = onset_seconds[-1]
    recent = onset_seconds[onset_seconds > now_seconds - config.window_seconds] - now_seconds
    if len(recent) < config.min_onsets:
        return None
    if np.median(np.diff(recent)) > config.max_median_interval_seconds:
        return None
    weights = np.exp(recent / config.recency_seconds)

    # Keep the whole octave inside min_bpm-max_bpm.
    reference_bpm = min(
        max(reference_bpm, config.min_bpm * math.sqrt(2)), config.max_bpm / math.sqrt(2)
    )
    half = STEPS_PER_OCTAVE // 2 + EDGE_STEPS
    beats_seconds = 60.0 / reference_bpm * 2.0 ** (np.arange(-half, half + 1) / STEPS_PER_OCTAVE)
    periods_seconds = beats_seconds / (2.0 ** np.arange(GRID_LEVELS))[:, np.newaxis]
    usable = periods_seconds >= MIN_PULSE_SECONDS
    periods = periods_seconds[usable]
    resultants = mean_resultants(recent, weights, periods)
    alignment = np.zeros(periods_seconds.shape)
    alignment[usable] = np.abs(resultants)
    fit = np.zeros(periods_seconds.shape)
    fit[usable] = grid_fit(recent, weights, periods, resultants, config.timing_tolerance_seconds)
    score = fit.sum(axis=0)

    # The edge candidates only serve the interpolation, so the result stays in the octave.
    best = EDGE_STEPS + int(np.argmax(score[EDGE_STEPS:-EDGE_STEPS]))
    offset_steps = _parabolic_peak(score[best - 1], score[best], score[best + 1])
    beat_seconds = beats_seconds[best] * 2.0 ** (offset_steps / STEPS_PER_OCTAVE)
    bpm = min(max(60.0 / beat_seconds, config.min_bpm), config.max_bpm)
    clarity = float(alignment[:, best].max())
    effective_onsets = float(np.sum(weights) ** 2 / np.sum(weights * weights))
    return TempoEstimate(bpm=bpm, clarity=clarity, significance=effective_onsets * clarity**2)


def _parabolic_peak(before: float, peak: float, after: float) -> float:
    """Offset (-0.5 to 0.5 steps) of the vertex of the parabola through three samples. Clamped:
    at the octave's edge the middle sample need not be the highest."""
    curvature = before - 2.0 * peak + after
    if curvature >= 0.0:
        return 0.0
    return min(0.5, max(-0.5, 0.5 * (before - after) / curvature))


class TempoTracker:
    """Turns per-onset estimates into a stable tempo (§28, §29).

    - Searching (unlocked): each estimate is made in the octave around the current run of
      estimates, or around `preferred_bpm` when there is none, and compared with the run. An
      estimate agrees when it is within `agreement_percent` and at least `min_clarity`. A run
      that lasts `lock_seconds` and `lock_estimates` estimates locks the tempo. The reported tempo
      is the latest estimate.
    - Locked: estimates are made in the octave around the locked tempo. An agreeing one moves the
      tempo `adaptation` of the way towards it, so slow drift is followed and one odd onset barely
      moves it. Disagreeing ones leave it alone. After `unlock_seconds` with no agreeing estimate,
      the lock is dropped and the search restarts from the latest onsets.
    - Silence changes nothing: no onsets, no evidence either way.
    - Confidence moves `confidence_rate` of the way towards each estimate's clarity if it agreed,
      and towards 0 if it didn't.

    Emits a `TempoUpdate` when the lock changes, the tempo moved `report_bpm_step` or the
    confidence `report_confidence_step` since the last update.
    """

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._config = config.tempo
        self._sample_rate_hz = sample_rate_hz
        self._onset_seconds: deque[float] = deque(
            maxlen=math.ceil(self._config.window_seconds * MAX_ONSETS_PER_SECOND)
        )
        self._manual_bpm: float | None = (
            float(self._config.manual_bpm) if self._config.manual_bpm is not None else None
        )
        if self._manual_bpm is not None:
            self._bpm: float | None = self._manual_bpm
            self._locked = True
            self._confidence = 1.0
        else:
            self._bpm = None
            self._locked = False
            self._confidence = 0.0
        self._run_estimates = 0  # agreeing estimates in the current run (searching)
        self._run_start_seconds = 0.0
        self._disagreeing_since_seconds: float | None = None  # locked only
        self._reported = self.state

    @property
    def manual_bpm(self) -> float | None:
        return self._manual_bpm

    @property
    def state(self) -> TempoState:
        return TempoState(bpm=self._bpm, confidence=self._confidence, locked=self._locked)

    def set_manual_bpm(self, manual_bpm: float | None) -> TempoState:
        """Set or clear a manual BPM override (Step 6.1, §31, §149)."""
        if manual_bpm is None:
            if self._manual_bpm is not None:
                self._manual_bpm = None
                self._bpm = None
                self._locked = False
                self._confidence = 0.0
                self._run_estimates = 0
                self._disagreeing_since_seconds = None
                self._reported = self.state
            return self.state
        clamped = float(manual_bpm)
        if not self._config.min_bpm <= clamped <= self._config.max_bpm:
            raise ValueError(
                f"manual_bpm must be in [{self._config.min_bpm}, {self._config.max_bpm}], "
                f"got {manual_bpm}"
            )
        self._manual_bpm = clamped
        self._bpm = clamped
        self._locked = True
        self._confidence = 1.0
        self._disagreeing_since_seconds = None
        self._reported = self.state
        return self.state

    def reset(self) -> TempoState:
        """Clear accumulated onset history and tempo search state while preserving manual BPM."""
        self._onset_seconds.clear()
        self._run_estimates = 0
        self._run_start_seconds = 0.0
        self._disagreeing_since_seconds = None
        if self._manual_bpm is not None:
            self._bpm = self._manual_bpm
            self._locked = True
            self._confidence = 1.0
        else:
            self._bpm = None
            self._locked = False
            self._confidence = 0.0
        self._reported = self.state
        return self.state

    def update(self, onset: OnsetEvent | None, frame_index: SampleFrame) -> TempoUpdate | None:
        """Feed this hop's onset, if any; `frame_index` is the hop's position, used for the
        update's timestamp."""
        if onset is None or self._manual_bpm is not None:
            return None
        now_seconds = frames_to_seconds(onset.frame_index, self._sample_rate_hz)
        self._onset_seconds.append(now_seconds)
        while self._onset_seconds[0] <= now_seconds - self._config.window_seconds:
            self._onset_seconds.popleft()
        onset_seconds = np.fromiter(self._onset_seconds, dtype=float)
        if self._locked:
            self._track(onset_seconds, now_seconds)
        else:
            self._search(onset_seconds, now_seconds)
        return self._report(frame_index)

    def _agrees(self, estimate: TempoEstimate, bpm: float) -> bool:
        config = self._config
        return (
            estimate.clarity >= config.min_clarity
            and abs(estimate.bpm / bpm - 1.0) * 100.0 <= config.agreement_percent
        )

    def _search(self, onset_seconds: np.ndarray, now_seconds: float) -> None:
        config = self._config
        if self._run_estimates > 0 and self._bpm is not None:
            estimate = estimate_tempo(onset_seconds, self._bpm, config)
            if estimate is None:
                return
            if self._agrees(estimate, self._bpm):
                self._bpm = estimate.bpm
                self._run_estimates += 1
                self._move_confidence(estimate.clarity)
                if (
                    self._run_estimates >= config.lock_estimates
                    and now_seconds - self._run_start_seconds >= config.lock_seconds
                    and estimate.significance >= config.lock_significance
                ):
                    self._locked = True
                    self._disagreeing_since_seconds = None
                return
        self._start_run(onset_seconds, now_seconds)

    def _start_run(self, onset_seconds: np.ndarray, now_seconds: float) -> None:
        estimate = estimate_tempo(onset_seconds, self._config.preferred_bpm, self._config)
        if estimate is None:
            return
        self._bpm = estimate.bpm
        self._move_confidence(0.0)
        clear = estimate.clarity >= self._config.min_clarity
        self._run_estimates = 1 if clear else 0
        self._run_start_seconds = now_seconds

    def _track(self, onset_seconds: np.ndarray, now_seconds: float) -> None:
        assert self._bpm is not None
        config = self._config
        estimate = estimate_tempo(onset_seconds, self._bpm, config)
        if estimate is None:
            return
        if self._agrees(estimate, self._bpm):
            self._bpm += config.adaptation * (estimate.bpm - self._bpm)
            self._disagreeing_since_seconds = None
            self._move_confidence(estimate.clarity)
            return
        self._move_confidence(0.0)
        if self._disagreeing_since_seconds is None:
            self._disagreeing_since_seconds = now_seconds
        elif now_seconds - self._disagreeing_since_seconds >= config.unlock_seconds:
            self._locked = False
            self._start_run(onset_seconds, now_seconds)

    def _move_confidence(self, support: float) -> None:
        self._confidence += self._config.confidence_rate * (support - self._confidence)

    def _report(self, frame_index: SampleFrame) -> TempoUpdate | None:
        config = self._config
        state = self.state
        last = self._reported
        if state.bpm is None:
            return None
        changed = (
            state.locked != last.locked
            or last.bpm is None
            or abs(state.bpm - last.bpm) >= config.report_bpm_step
            or abs(state.confidence - last.confidence) >= config.report_confidence_step
        )
        if not changed:
            return None
        self._reported = state
        return TempoUpdate(
            frame_index=frame_index,
            timestamp_seconds=frames_to_seconds(frame_index, self._sample_rate_hz),
            bpm=state.bpm,
            confidence=min(1.0, max(0.0, state.confidence)),
            locked=state.locked,
        )

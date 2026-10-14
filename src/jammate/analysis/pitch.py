"""Monophonic pitch detection: band-limiting, a level gate and YIN (§15, §25).

Runs on the analysis worker thread, never in the audio callback. `yin` is a pure function;
`PitchDetector` adds the per-hop state (filter memory and the sliding analysis window).

YIN (de Cheveigné & Kawahara, 2002) is implemented here in NumPy instead of using
`librosa.yin`, because librosa returns no confidence and its pYIN works on whole clips, not
a stream. The confidence is `1 - aperiodicity` at the chosen period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from jammate.audio.levels import SILENCE_FLOOR_DBFS, amplitude_to_dbfs
from jammate.config import AnalysisConfig
from jammate.music.events import MidiNote, SampleFrame

if TYPE_CHECKING:
    from jammate.analysis.guitar import GuitarChordAnalyzer

# Light band-limiting before YIN: the high-pass corner removes DC and rumble, the low-pass corner
# removes hiss and pick noise far above the highest fundamental. Both are relative to the range.
HIGHPASS_RATIO = 0.5  # times pitch_min_hz
LOWPASS_RATIO = 2.0  # times pitch_max_hz
MAX_LOWPASS_NYQUIST_FRACTION = 0.9
FILTER_ORDER = 2


@dataclass(frozen=True)
class YinResult:
    frequency_hz: float | None  # None: no period was clearly periodic (aperiodicity >= threshold)
    confidence: float  # 1 - aperiodicity of the best period, 0.0-1.0


@dataclass(frozen=True)
class PitchFrame:
    """One hop's analysis. The window ends just before `frame_index` (the stream position)."""

    frame_index: SampleFrame
    frequency_hz: float | None  # None when gated (too quiet) or unvoiced
    confidence: float
    rms_dbfs: float  # band-limited window RMS: in-band energy only, so DC or rumble never count
    peak_dbfs: float  # unfiltered peak of this hop's new frames (for clipping)
    # The recent playing level: follows rms_dbfs up at once and falls back slowly. Sounds far
    # below it are by-products of playing (see `new_note_gate_db`).
    playing_level_dbfs: float
    chord_notes: tuple[MidiNote, ...] = ()
    chord_type: str | None = None

    def is_far_below_playing_level(self, gate_db: float) -> bool:
        return self.rms_dbfs < self.playing_level_dbfs - gate_db


def next_playing_level(previous_dbfs: float, rms_dbfs: float, release_db: float) -> float:
    """The playing level after one hop: up to `rms_dbfs` at once, down by `release_db` at most."""
    return max(rms_dbfs, previous_dbfs - release_db)


def yin_lag_range(sample_rate_hz: int, min_hz: float, max_hz: float) -> tuple[int, int]:
    """Shortest and longest period, in frames, that YIN searches."""
    return max(2, math.floor(sample_rate_hz / max_hz)), math.ceil(sample_rate_hz / min_hz)


def yin_window_frames(sample_rate_hz: int, min_hz: float) -> int:
    """Analysis window: the longest period twice (integration window + lag), plus 2 frames for
    the parabolic interpolation."""
    return 2 * math.ceil(sample_rate_hz / min_hz) + 2


def yin(
    window: np.ndarray, sample_rate_hz: int, min_hz: float, max_hz: float, threshold: float
) -> YinResult:
    """Estimate the fundamental frequency of `window` (1-D, at least `yin_window_frames` long).

    Picks the first lag whose cumulative mean normalized difference falls below `threshold`,
    then the bottom of that dip. Preferring the first dip avoids sub-octave errors; requiring a
    clear dip rejects noise and chords.
    """
    tau_min, tau_max = yin_lag_range(sample_rate_hz, min_hz, max_hz)
    x = window.astype(np.float64)
    integration_frames = len(x) - tau_max - 1
    if integration_frames < tau_max:
        raise ValueError(f"window of {len(x)} frames is too short for {min_hz} Hz")

    # Difference function d(tau) = sum_j (x[j] - x[j + tau])^2 over the integration window,
    # from energies and an FFT cross-correlation.
    fft_frames = 1 << (len(x) + integration_frames).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(x, fft_frames) * np.conj(np.fft.rfft(x[:integration_frames], fft_frames)),
        fft_frames,
    )[: tau_max + 2]
    cumulative_energy = np.concatenate(([0.0], np.cumsum(x * x)))
    lags = np.arange(tau_max + 2)
    shifted_energy = cumulative_energy[lags + integration_frames] - cumulative_energy[lags]
    difference = np.maximum(shifted_energy[0] + shifted_energy - 2.0 * correlation, 0.0)

    # Cumulative mean normalized difference: d'(0) = 1, d'(tau) = d(tau) / mean(d[1..tau]).
    running_sum = np.cumsum(difference[1:])
    normalized = np.ones_like(difference)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized[1:] = np.where(running_sum > 0, difference[1:] * lags[1:] / running_sum, 1.0)

    search = normalized[tau_min : tau_max + 1]
    below = np.flatnonzero(search < threshold)
    if below.size == 0:
        best = tau_min + int(np.argmin(search))
        return YinResult(None, _confidence(normalized[best]))
    tau = tau_min + int(below[0])
    while tau < tau_max and normalized[tau + 1] < normalized[tau]:
        tau += 1

    # Parabolic interpolation through the dip for a sub-frame period.
    before, at, after = normalized[tau - 1], normalized[tau], normalized[tau + 1]
    curvature = before - 2.0 * at + after
    offset = 0.5 * (before - after) / curvature if curvature > 0 else 0.0
    period_frames = tau + max(-0.5, min(0.5, offset))
    return YinResult(sample_rate_hz / period_frames, _confidence(at))


def _confidence(aperiodicity: float) -> float:
    return float(min(1.0, max(0.0, 1.0 - aperiodicity)))


class PitchDetector:
    """Per-hop pitch estimates for a mono stream.

    Each hop's frames are band-limited (with filter state kept across hops) and appended to a
    sliding window. The window's RMS is gated at `level_gate_dbfs`; above the gate, YIN runs on
    the window. When `config.instrument == "guitar"`, `GuitarChordAnalyzer` refines the result to
    correct power-chord subharmonics and extract polyphonic chord roots/tones.
    """

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._min_hz, self._max_hz = config.pitch_range_hz
        self.window_frames = yin_window_frames(sample_rate_hz, self._min_hz)
        if config.hop_frames > self.window_frames:
            raise ValueError("hop_frames must not exceed the analysis window")
        nyquist_hz = sample_rate_hz / 2
        lowpass_hz = min(LOWPASS_RATIO * self._max_hz, MAX_LOWPASS_NYQUIST_FRACTION * nyquist_hz)
        self._sos = signal.butter(
            FILTER_ORDER,
            [HIGHPASS_RATIO * self._min_hz, lowpass_hz],
            btype="bandpass",
            fs=sample_rate_hz,
            output="sos",
        )
        self._filter_state = np.zeros((self._sos.shape[0], 2))
        self._release_db_per_hop = (
            config.playing_level_release_db_per_second * config.hop_frames / sample_rate_hz
        )
        self._playing_level_dbfs = SILENCE_FLOOR_DBFS
        self._window = np.zeros(self.window_frames)
        self._guitar_analyzer: GuitarChordAnalyzer | None = None
        if config.instrument == "guitar":
            from jammate.analysis.guitar import GuitarChordAnalyzer

            self._guitar_analyzer = GuitarChordAnalyzer(config, sample_rate_hz)

    def process_hop(self, hop: np.ndarray, end_frame: SampleFrame) -> PitchFrame:
        """Analyze the window after appending `hop` (1-D) whose last frame is `end_frame - 1`."""
        count = len(hop)
        filtered, self._filter_state = signal.sosfilt(self._sos, hop, zi=self._filter_state)
        window = self._window
        window[:-count] = window[count:]
        window[-count:] = filtered
        if self._guitar_analyzer is not None:
            self._guitar_analyzer.push_hop(filtered)

        rms_dbfs = amplitude_to_dbfs(math.sqrt(float(np.mean(window**2))))
        peak_dbfs = amplitude_to_dbfs(float(np.max(np.abs(hop)))) if count else SILENCE_FLOOR_DBFS
        self._playing_level_dbfs = next_playing_level(
            self._playing_level_dbfs, rms_dbfs, self._release_db_per_hop
        )
        levels = (rms_dbfs, peak_dbfs, self._playing_level_dbfs)
        if rms_dbfs < self._config.level_gate_dbfs:
            return PitchFrame(end_frame, None, 0.0, *levels)
        result = yin(
            window,
            self._sample_rate_hz,
            self._min_hz,
            self._max_hz,
            self._config.yin_threshold,
        )
        if self._guitar_analyzer is not None:
            chord_est = self._guitar_analyzer.analyze(
                result,
                rms_dbfs=rms_dbfs,
                playing_level_dbfs=self._playing_level_dbfs,
            )
            if chord_est is not None:
                chord_type = None if chord_est.chord_type == "single" else chord_est.chord_type
                return PitchFrame(
                    end_frame,
                    chord_est.root_hz,
                    chord_est.confidence,
                    *levels,
                    chord_notes=chord_est.chord_notes,
                    chord_type=chord_type,
                )
        return PitchFrame(end_frame, result.frequency_hz, result.confidence, *levels)

"""Onset detection: when a new note or strum was attacked (§27).

Runs on the analysis worker thread, one hop at a time, in two stages.

Detection. The detection function per hop is

    spectral flux of log-compressed band magnitudes  +  level_rise_weight * hop level rise (dB)

- The flux compares the latest analysis window with the one FLUX_LAG_HOPS earlier: a strum
  spreads over 20-50 ms, and a one-hop (10 ms) difference sees only a slice of it.
- The spectrum is grouped into log-spaced bands (BANDS_PER_OCTAVE), so steady noise, whose
  single-bin magnitudes fluctuate randomly, averages out within each band, while an attack still
  raises many bands at once.
- The log compression makes the flux respond to relative change, so soft and loud playing are
  treated alike, while bands near the noise floor contribute almost nothing.
- The hop level rise catches re-attacks of the same note, which barely change the spectrum. The
  level is floored at FLOOR_DBFS, so a rise out of digital silence doesn't dwarf everything else.

An onset is a local peak of the detection function above an adaptive threshold (median + k·MAD
of the recent values, never below `min_detection`), at least `refractory_seconds` after the
previous onset, followed by a window level of at least `min_level_dbfs` and no more than
`new_note_gate_db` below the recent playing level (fret noise, pull-offs onto open strings and
touch noise).

Timing. Peak picking needs the next hop, so each onset is reported one hop after its peak. Its
frame index is then refined within the peak's analysis window, in REFINE_BLOCK_FRAMES blocks:
the earliest block whose energy rise over the REFINE_LOOKBACK_BLOCKS before it comes within
REFINE_TOLERANCE_DB of the largest rise in the window. Within one note, the waveform's own peaks
and troughs make block energy swing by tens of dB, but those swings come after the attack. The
result is accurate to a few milliseconds and stays within the detection window.
"""

import math
import statistics
from collections import deque

import numpy as np

from jammate.analysis.pitch import PitchFrame
from jammate.audio.levels import amplitude_to_dbfs
from jammate.config import AnalysisConfig
from jammate.music.events import OnsetEvent, SampleFrame, frames_to_seconds, seconds_to_frames

FFT_FRAMES = 1024  # 21 ms at 48 kHz
FLUX_MIN_HZ = 40.0
FLUX_MAX_HZ = 8000.0  # pick attacks are broadband; above this there is mostly hiss
BANDS_PER_OCTAVE = 6
FLUX_LAG_HOPS = 2
# log10(1 + COMPRESSION * magnitude), magnitude normalized so a full-scale sine peaks at 1.0:
# bands below about -60 dB contribute almost nothing.
COMPRESSION = 1000.0
FLOOR_DBFS = -80.0
REFINE_BLOCK_FRAMES = 64  # 1.3 ms
REFINE_LOOKBACK_BLOCKS = 2
REFINE_TOLERANCE_DB = 6.0
MIN_POWER = 1e-12


class OnsetDetector:
    def __init__(self, analysis_config: AnalysisConfig, sample_rate_hz: int) -> None:
        config = analysis_config.onset
        hop_frames = analysis_config.hop_frames
        if hop_frames > FFT_FRAMES:
            raise ValueError(f"hop_frames must not exceed {FFT_FRAMES}")
        self._config = config
        self._gate_db = analysis_config.new_note_gate_db
        self._sample_rate_hz = sample_rate_hz
        self._refractory_frames = seconds_to_frames(config.refractory_seconds, sample_rate_hz)
        window_hops = max(3, round(config.threshold_window_seconds * sample_rate_hz / hop_frames))
        self._history: deque[float] = deque(maxlen=window_hops)
        self._taper = np.hanning(FFT_FRAMES)
        self._magnitude_scale = 2.0 / float(np.sum(self._taper))
        self._bins, self._band_starts, self._band_sizes = _bands(sample_rate_hz)
        self._past_log_magnitudes: deque[np.ndarray] = deque(
            [np.zeros(len(self._band_starts))] * FLUX_LAG_HOPS, maxlen=FLUX_LAG_HOPS
        )
        self._previous_hop_dbfs = FLOOR_DBFS
        # Raw input: the peak's analysis window, the hop after it and the lookback before it.
        self._hop_frames = hop_frames
        self._recent = np.zeros(
            FFT_FRAMES + hop_frames + REFINE_LOOKBACK_BLOCKS * REFINE_BLOCK_FRAMES
        )
        # The previous hop, waiting for this hop to show whether it was a peak.
        self._candidate_value = 0.0
        self._before_candidate_value = 0.0
        self._candidate_threshold = math.inf
        self._last_onset_frame: SampleFrame | None = None

    def process_hop(self, hop: np.ndarray, frame: PitchFrame) -> OnsetEvent | None:
        """Feed one hop (1-D, raw input) with the pitch detector's `frame` for the same hop
        (levels and position). Returns the onset found at the previous hop, if any."""
        count = len(hop)
        self._recent[:-count] = self._recent[count:]
        self._recent[-count:] = hop

        window = self._recent[-FFT_FRAMES:] * self._taper
        power = np.abs(np.fft.rfft(window))[self._bins] ** 2
        band_magnitude = np.sqrt(np.add.reduceat(power, self._band_starts) / self._band_sizes)
        log_magnitude = np.log10(1.0 + COMPRESSION * self._magnitude_scale * band_magnitude)
        flux = float(np.sum(np.maximum(log_magnitude - self._past_log_magnitudes[0], 0.0)))
        self._past_log_magnitudes.append(log_magnitude)

        hop_dbfs = max(FLOOR_DBFS, amplitude_to_dbfs(math.sqrt(float(np.mean(hop * hop)))))
        level_rise_db = max(0.0, hop_dbfs - self._previous_hop_dbfs)
        self._previous_hop_dbfs = hop_dbfs
        value = flux + self._config.level_rise_weight * level_rise_db

        onset = self._pick_candidate(value, frame)

        threshold = self._config.min_detection
        if len(self._history) >= 3:
            median = statistics.median(self._history)
            mad = statistics.median(abs(v - median) for v in self._history)
            threshold = max(threshold, median + self._config.threshold_mads * mad)
        self._history.append(value)
        self._before_candidate_value = self._candidate_value
        self._candidate_value = value
        self._candidate_threshold = threshold
        return onset

    def _pick_candidate(self, next_value: float, next_frame: PitchFrame) -> OnsetEvent | None:
        value = self._candidate_value
        is_peak = value >= self._before_candidate_value and value > next_value
        if not is_peak or value <= self._candidate_threshold:
            return None
        if next_frame.rms_dbfs < self._config.min_level_dbfs:
            return None
        if next_frame.is_far_below_playing_level(self._gate_db):
            return None
        attack_offset = self._attack_offset()
        frame_index = max(0, next_frame.frame_index - len(self._recent) + attack_offset)
        if (
            self._last_onset_frame is not None
            and frame_index - self._last_onset_frame < self._refractory_frames
        ):
            return None
        self._last_onset_frame = frame_index
        return OnsetEvent(
            frame_index=frame_index,
            timestamp_seconds=frames_to_seconds(frame_index, self._sample_rate_hz),
            strength=float(np.max(np.abs(self._recent[attack_offset:]))),
        )

    def _attack_offset(self) -> int:
        """Offset in `_recent` of the attack: see the module docstring (Timing)."""
        window_end = len(self._recent) - self._hop_frames  # the peak hop's window
        region = self._recent[: window_end - window_end % REFINE_BLOCK_FRAMES]
        first = len(region) % REFINE_BLOCK_FRAMES  # keep whole blocks, aligned to the end
        blocks = region[first:].reshape(-1, REFINE_BLOCK_FRAMES)
        energy_db = np.maximum(
            10.0 * np.log10(np.maximum(np.mean(blocks * blocks, axis=1), MIN_POWER)), FLOOR_DBFS
        )
        before = np.lib.stride_tricks.sliding_window_view(energy_db[:-1], REFINE_LOOKBACK_BLOCKS)
        rises = energy_db[REFINE_LOOKBACK_BLOCKS:] - before.max(axis=1)
        attack = int(np.flatnonzero(rises >= rises.max() - REFINE_TOLERANCE_DB)[0])
        return first + (REFINE_LOOKBACK_BLOCKS + attack) * REFINE_BLOCK_FRAMES


def _bands(sample_rate_hz: int) -> tuple[slice, np.ndarray, np.ndarray]:
    """FFT bins from FLUX_MIN_HZ to FLUX_MAX_HZ, grouped BANDS_PER_OCTAVE per octave. Low bins
    narrower than a band keep a band each. Returns the bin slice, each band's first bin within
    it (for `np.add.reduceat`) and the bins per band."""
    bin_hz = sample_rate_hz / FFT_FRAMES
    first = max(1, math.ceil(FLUX_MIN_HZ / bin_hz))
    last = math.floor(FLUX_MAX_HZ / bin_hz)
    frequencies_hz = np.arange(first, last + 1) * bin_hz
    band_of_bin = np.floor(BANDS_PER_OCTAVE * np.log2(frequencies_hz / FLUX_MIN_HZ)).astype(int)
    starts = np.flatnonzero(np.diff(band_of_bin, prepend=-1))
    sizes = np.diff(np.append(starts, len(band_of_bin)))
    return slice(first, last + 1), starts, sizes

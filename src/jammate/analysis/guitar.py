"""Polyphonic electric guitar power-chord and chord root extraction (Step 5.1, §53, §54, D7, D12).

Runs on the analysis worker thread when `config.instrument == "guitar"`, never in the audio
callback. Complements monophonic YIN (`analysis/pitch.py`) in two ways:

1. **Power-chord subharmonic & harmonic correction (when YIN is voiced):**
   - A power chord's root (`f0`) and fifth (`~1.5 * f0`) share a composite period of `2 / f0`
     (frequency `0.5 * f0`), causing raw YIN to report an octave below the root (`E1..D2`, below
     standard guitar `E2 = 82.4 Hz`) when searching down to `60 Hz` (or `35 Hz`). By checking
     whether the fundamental at `f_yin` is absent (`a1 << a2`) while `2 * f_yin` and `3 * f_yin`
     carry the chord energy, `GuitarChordAnalyzer` corrects the subharmonic trap to the true root.
   - Conversely, when a full chord (`root + fifth + octave`, e.g. `A2 + E3 + A3`) causes YIN to
     lock onto the octave (`2 * f0 = A3`) or fifth (`1.5 * f0 = E3`), `GuitarChordAnalyzer` checks
     whether both the lower root (`0.5 * f_yin`) and lower fifth (`0.75 * f_yin`) have true local
     spectral peaks, recovering the true bass root (`A2`) without ever affecting single notes.

2. **Polyphonic chord root & quality extraction (when YIN is unvoiced or a chord is played):**
   Multi-string open/barre major and minor chords often have YIN aperiodicity above the monophonic
   threshold (`frequency_hz = None`). Using parabolic local-maximum peak detection
   (`_local_peak_near`) so adjacent semitones never bleed across the Hann mainlobe,
   `GuitarChordAnalyzer` extracts the chord root (`root_midi`, `root_hz`), quality (`"power"`,
   `"major"`, `"minor"`), and active chord tones (`chord_notes`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from jammate.analysis.notes import hz_to_midi_float, midi_to_hz
from jammate.analysis.pitch import YinResult
from jammate.config import AnalysisConfig
from jammate.music.events import MidiNote

type ChordType = Literal["single", "power", "major", "minor"]

FFT_FRAMES = 8192
CHORD_WINDOW_SECONDS = 0.085  # 4080 frames at 48 kHz: sine mainlobe half-width = 17.6 Hz
LOWEST_GUITAR_ROOT_MIDI: MidiNote = 40  # E2 (82.41 Hz, standard tuning lowest open string)
HIGHEST_GUITAR_ROOT_MIDI: MidiNote = 64  # E4 (329.63 Hz)
SUBHARMONIC_CEILING_HZ = 78.0  # Below E2 (82.41 Hz): where 0.5*f0 subharmonics (E1..D2) land
PEAK_SEARCH_CENTS = 38.0
MIN_SPECTRAL_CREST = 12.0  # rejects broadband white/pink noise and pick scrapes
MIN_CHORD_CONFIDENCE = 0.82
UNVOICED_CHORD_MAX_DROP_DB = 4.5  # only recover unvoiced chords at active strum level, not decay

_FIFTH_RATIO = 2.0 ** (7.0 / 12.0)  # 1.498307
_MAJ3_RATIO = 2.0 ** (4.0 / 12.0)  # 1.259921
_MIN3_RATIO = 2.0 ** (3.0 / 12.0)  # 1.189207


@dataclass(frozen=True, slots=True)
class GuitarChordEstimate:
    """Per-hop guitar pitch/chord analysis result."""

    root_midi: MidiNote
    root_hz: float
    confidence: float
    chord_notes: tuple[MidiNote, ...]
    chord_type: ChordType


class GuitarChordAnalyzer:
    """Streaming harmonic analyzer for electric guitar single notes, power chords, and triads."""

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._min_hz, self._max_hz = config.pitch_range_hz
        self._buffer_frames = max(
            round(CHORD_WINDOW_SECONDS * sample_rate_hz),
            2 * math.ceil(sample_rate_hz / self._min_hz) + 2,
        )
        self._buffer = np.zeros(self._buffer_frames, dtype=np.float64)
        # Sine window has mainlobe half-width 1.5 * fs / N = 17.6 Hz (< 23.4 Hz between B2 and D3)
        # and 2.0 full beat cycles of B2-D3 (23.36 Hz), cleanly resolving G2, B2, and D3.
        self._window = np.sin(
            np.pi * (np.arange(self._buffer_frames, dtype=np.float64) + 0.5) / self._buffer_frames
        )
        self._bin_hz = sample_rate_hz / FFT_FRAMES
        self._search_ratio = 2.0 ** (PEAK_SEARCH_CENTS / 1200.0)

        min_midi = max(LOWEST_GUITAR_ROOT_MIDI, math.ceil(hz_to_midi_float(self._min_hz)))
        max_midi = min(HIGHEST_GUITAR_ROOT_MIDI, math.floor(hz_to_midi_float(self._max_hz)))
        self._candidate_roots: tuple[MidiNote, ...] = tuple(range(min_midi, max_midi + 1))
        self._active_frames = 0

    def reset(self) -> None:
        """Clear the internal sliding window."""
        self._buffer.fill(0.0)
        self._active_frames = 0

    def push_hop(self, filtered_hop: np.ndarray) -> None:
        """Append one band-limited hop to the chord analysis sliding window."""
        count = len(filtered_hop)
        if count >= self._buffer_frames:
            self._buffer[:] = filtered_hop[-self._buffer_frames :]
        elif count > 0:
            self._buffer[:-count] = self._buffer[count:]
            self._buffer[-count:] = filtered_hop
        self._active_frames = min(self._buffer_frames, self._active_frames + count)

    def analyze(
        self,
        yin_result: YinResult,
        *,
        rms_dbfs: float,
        playing_level_dbfs: float,
    ) -> GuitarChordEstimate | None:
        """Refine `yin_result` for guitar: fix power-chord subharmonics or extract chord roots."""
        if rms_dbfs < self._config.level_gate_dbfs:
            self._active_frames = 0
            return None

        if 0 < self._active_frames < self._buffer_frames:
            n_active = self._active_frames
            active_win = np.sin(np.pi * (np.arange(n_active, dtype=np.float64) + 0.5) / n_active)
            windowed = self._buffer[-n_active:] * active_win
        else:
            windowed = self._buffer * self._window
        spectrum = np.abs(np.fft.rfft(windowed, n=FFT_FRAMES))
        low_bin = max(1, int(self._min_hz / self._bin_hz))
        high_bin = min(len(spectrum) - 1, int(1200.0 / self._bin_hz))
        band = spectrum[low_bin:high_bin]
        median_level = float(np.median(band)) if band.size > 0 else 0.0

        if yin_result.frequency_hz is not None:
            return self._analyze_voiced(
                yin_result.frequency_hz,
                yin_result.confidence,
                spectrum,
                band,
                median_level,
            )

        # Unvoiced YIN: only attempt polyphonic chord recovery once enough signal fills the
        # spectral window and RMS is within UNVOICED_CHORD_MAX_DROP_DB of playing level.
        if self._active_frames < self._buffer_frames // 2:
            return None
        if rms_dbfs < playing_level_dbfs - UNVOICED_CHORD_MAX_DROP_DB:
            return None
        return self._analyze_unvoiced_chord(spectrum, band, median_level)

    def _analyze_voiced(
        self,
        f_yin: float,
        confidence: float,
        spectrum: np.ndarray,
        band: np.ndarray,
        median_level: float,
    ) -> GuitarChordEstimate:
        f_root = f_yin
        lowest_root_hz = midi_to_hz(LOWEST_GUITAR_ROOT_MIDI) * 0.96

        # 1. Check for the D7 power-chord subharmonic trap (YIN locked onto 0.5 * f_root).
        # On a standard-tuned guitar (lowest root E2 = 82.41 Hz), 0.5 * f_root for E5..D5 lands
        # below SUBHARMONIC_CEILING_HZ (78 Hz: E1..D2), where no guitar string fundamental exists.
        if f_yin < SUBHARMONIC_CEILING_HZ and 2.0 * f_yin <= self._max_hz:
            a1 = self._local_peak_near(spectrum, f_yin)
            a2 = self._local_peak_near(spectrum, 2.0 * f_yin)
            a3 = self._local_peak_near(spectrum, 2.0 * f_yin * _FIFTH_RATIO)
            if a2 > 0.0 and a3 >= 0.18 * a2 and a1 < 0.25 * a2:
                f_root = 2.0 * f_yin
                root_midi = self._clamp_midi(round(hz_to_midi_float(f_root)))
                chord_type, chord_notes = self._classify_chord_quality(
                    spectrum, root_midi, midi_to_hz(root_midi), median_level, force_chord=True
                )
                return GuitarChordEstimate(
                    root_midi=root_midi,
                    root_hz=f_root,
                    confidence=max(confidence, MIN_CHORD_CONFIDENCE),
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                )

        # 2. Check if YIN locked onto the octave (2 * f_root) of a chord whose true bass root
        # (0.5 * f_yin >= E2) AND fifth (0.5 * f_yin * 1.5) both have distinct local peaks.
        # A single note at f_yin has zero energy at 0.5 * f_yin and 0.75 * f_yin.
        f_below_octave = 0.5 * f_yin
        if f_below_octave >= lowest_root_hz:
            a_yin = self._local_peak_near(spectrum, f_yin)
            a_low_root = self._local_peak_near(spectrum, f_below_octave)
            a_low_fifth = self._local_peak_near(spectrum, f_below_octave * _FIFTH_RATIO)
            if (
                a_yin > 0.0
                and a_low_root >= 0.18 * a_yin
                and a_low_fifth >= 0.18 * a_yin
                and a_low_root >= 4.5 * max(median_level, 1e-9)
                and a_low_fifth >= 4.5 * max(median_level, 1e-9)
            ):
                f_root = f_below_octave
                root_midi = self._clamp_midi(round(hz_to_midi_float(f_root)))
                chord_type, chord_notes = self._classify_chord_quality(
                    spectrum, root_midi, midi_to_hz(root_midi), median_level, force_chord=True
                )
                return GuitarChordEstimate(
                    root_midi=root_midi,
                    root_hz=f_root,
                    confidence=max(confidence, MIN_CHORD_CONFIDENCE),
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                )

        # 3. Check if YIN locked onto the fifth (1.5 * f_root) or shared 3rd harmonic (3 * f_root)
        # of a lower power chord whose root is >= E2 (82.41 Hz).
        f_below_fifth = f_yin / _FIFTH_RATIO
        if f_below_fifth >= lowest_root_hz:
            a_yin = self._local_peak_near(spectrum, f_yin)
            a_root = self._local_peak_near(spectrum, f_below_fifth)
            a_root_oct = self._local_peak_near(spectrum, 2.0 * f_below_fifth)
            a_sub = self._local_peak_near(spectrum, 0.5 * f_below_fifth)
            if (
                a_yin > 0.0
                and a_root >= 0.40 * a_yin
                and a_root_oct >= 0.30 * a_yin
                and a_sub < 0.15 * a_root
                and a_root >= 4.5 * max(median_level, 1e-9)
            ):
                f_root = f_below_fifth
                root_midi = self._clamp_midi(round(hz_to_midi_float(f_root)))
                chord_type, chord_notes = self._classify_chord_quality(
                    spectrum, root_midi, midi_to_hz(root_midi), median_level, force_chord=True
                )
                return GuitarChordEstimate(
                    root_midi=root_midi,
                    root_hz=f_root,
                    confidence=max(confidence, MIN_CHORD_CONFIDENCE),
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                )

        f_below_third_harm = f_yin / 3.0
        if f_below_third_harm >= lowest_root_hz:
            a_yin = self._local_peak_near(spectrum, f_yin)
            a_root = self._local_peak_near(spectrum, f_below_third_harm)
            a_fifth = self._local_peak_near(spectrum, f_below_third_harm * _FIFTH_RATIO)
            a_oct = self._local_peak_near(spectrum, 2.0 * f_below_third_harm)
            if (
                a_yin > 0.0
                and a_root >= 0.35 * a_yin
                and a_fifth >= 0.28 * a_yin
                and a_oct >= 0.30 * a_yin
                and a_fifth >= 4.5 * max(median_level, 1e-9)
            ):
                f_root = f_below_third_harm
                root_midi = self._clamp_midi(round(hz_to_midi_float(f_root)))
                chord_type, chord_notes = self._classify_chord_quality(
                    spectrum, root_midi, midi_to_hz(root_midi), median_level, force_chord=True
                )
                return GuitarChordEstimate(
                    root_midi=root_midi,
                    root_hz=f_root,
                    confidence=max(confidence, MIN_CHORD_CONFIDENCE),
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                )

        # 4. If YIN locked onto a subharmonic below E2 (with no fundamental peak) or weakly onto
        # an upper string of a multi-string chord (confidence < 0.88) while a lower guitar string
        # (>= E2) has a strong peak below 0.88 * f_yin, delegate to the polyphonic chord finder.
        has_lower_string = False
        if confidence < 0.88:
            a_yin_fund = self._local_peak_near(spectrum, f_yin)
            low_bin = max(1, int(lowest_root_hz / self._bin_hz))
            below_yin_bin = min(len(spectrum) - 1, int((0.88 * f_yin) / self._bin_hz))
            if below_yin_bin > low_bin + 2:
                lower_max = float(np.max(spectrum[low_bin:below_yin_bin]))
                if lower_max >= 0.25 * max(a_yin_fund, 1e-9) and lower_max >= 5.0 * max(
                    median_level, 1e-9
                ):
                    has_lower_string = True
        if f_yin < SUBHARMONIC_CEILING_HZ or has_lower_string:
            chord_est = self._analyze_unvoiced_chord(spectrum, band, median_level)
            if chord_est is not None:
                return chord_est

        # 5. Normal voiced note at f_root: check if a fifth/third is also strongly present.
        root_midi = self._clamp_midi(round(hz_to_midi_float(f_root)))
        chord_type, chord_notes = self._classify_chord_quality(
            spectrum, root_midi, f_root, median_level, force_chord=False
        )
        return GuitarChordEstimate(
            root_midi=root_midi,
            root_hz=f_root,
            confidence=confidence,
            chord_notes=chord_notes,
            chord_type=chord_type,
        )

    def _analyze_unvoiced_chord(
        self,
        spectrum: np.ndarray,
        band: np.ndarray,
        median_level: float,
    ) -> GuitarChordEstimate | None:
        if band.size == 0:
            return None

        max_peak = float(np.max(band))
        if max_peak <= 1e-6 or median_level <= 0.0:
            return None

        # Reject steady white/pink noise and broadband pick noise.
        crest = max_peak / median_level
        if crest < MIN_SPECTRAL_CREST:
            return None

        low_bin = max(1, int(self._min_hz / self._bin_hz))
        fund_high_bin = min(len(spectrum) - 1, int(360.0 / self._bin_hz))
        max_fund_peak = float(np.max(spectrum[low_bin:fund_high_bin]))
        if max_fund_peak <= 1e-6:
            return None

        candidates: dict[MidiNote, tuple[float, float]] = {}
        for midi_note in self._candidate_roots:
            f0 = midi_to_hz(midi_note)
            f5 = f0 * _FIFTH_RATIO

            # Require true local spectral maxima at the root, octave, and fifth so adjacent
            # semitones (e.g. F2 next to E2) never match the slope of a neighbor's mainlobe.
            a_root = self._local_peak_near(spectrum, f0)
            a_oct = self._local_peak_near(spectrum, 2.0 * f0)
            a_fifth = self._local_peak_near(spectrum, f5)
            a_fifth_2 = self._local_peak_near(spectrum, 2.0 * f5)
            a_fifth_3 = self._local_peak_near(spectrum, 3.0 * f5)
            a_harm3 = self._local_peak_near(spectrum, 3.0 * f0)
            a_harm4 = self._local_peak_near(spectrum, 4.0 * f0)

            # In a 6-string chord, upper octaves/fifths are reinforced by 2-3 strings while the
            # lowest bass root is 1 string (~0.10..0.20 of max_fund_peak).
            if a_root < 0.10 * max_fund_peak or a_oct < 0.20 * a_root:
                continue
            if a_root < 4.0 * max(median_level, 1e-9):
                continue
            if a_fifth < 0.16 * a_root and a_fifth_2 < 0.25 * a_oct:
                continue
            if a_fifth < 0.08 * max_fund_peak:
                continue
            if a_fifth_3 < 0.05 * max(a_root, a_oct) and a_fifth < 0.24 * a_root:
                continue
            if a_fifth < 3.5 * max(median_level, 1e-9):
                continue

            root_series = 1.25 * a_root + 0.85 * a_oct + 0.55 * a_harm3 + 0.35 * a_harm4
            fifth_series = 0.95 * a_fifth + 0.45 * a_fifth_3
            maj3_score = self._third_support(spectrum, f0, _MAJ3_RATIO)
            min3_score = self._third_support(spectrum, f0, _MIN3_RATIO)
            third_series = max(maj3_score, min3_score)

            score = root_series + fifth_series + 0.45 * third_series
            candidates[midi_note] = (score, a_root)

        if not candidates:
            return None

        best_midi = max(candidates, key=lambda m: candidates[m][0])
        best_score, _ = candidates[best_midi]

        # If a valid chord candidate exists an octave (-12) or fifth (-7) below best_midi,
        # prefer the lower root (upper strings cannot produce fundamentals at 0.5*f0 and 0.75*f0).
        for interval in (12, 7):
            lower_midi = best_midi - interval
            if lower_midi in candidates:
                lower_score, lower_root_amp = candidates[lower_midi]
                if lower_score >= 0.20 * best_score and lower_root_amp >= 0.10 * max_fund_peak:
                    best_midi = lower_midi
                    best_score = lower_score

        root_hz = self._refine_frequency(spectrum, midi_to_hz(best_midi))
        chord_type, chord_notes = self._classify_chord_quality(
            spectrum, best_midi, root_hz, median_level, force_chord=True
        )
        confidence = min(
            0.98, max(MIN_CHORD_CONFIDENCE, (crest - MIN_SPECTRAL_CREST) / 25.0 + 0.82)
        )
        return GuitarChordEstimate(
            root_midi=best_midi,
            root_hz=root_hz,
            confidence=confidence,
            chord_notes=chord_notes,
            chord_type=chord_type,
        )

    def _classify_chord_quality(
        self,
        spectrum: np.ndarray,
        root_midi: MidiNote,
        f0: float,
        median_level: float,
        *,
        force_chord: bool,
    ) -> tuple[ChordType, tuple[MidiNote, ...]]:
        a_root = max(
            self._local_peak_near(spectrum, f0),
            0.65 * self._local_peak_near(spectrum, 2.0 * f0),
        )
        if a_root <= 1e-6:
            return ("single", ())

        f5 = f0 * _FIFTH_RATIO
        a_fifth = self._local_peak_near(spectrum, f5)
        a_fifth_3 = self._local_peak_near(spectrum, 3.0 * f5)
        bg_ref = max(median_level, 1e-9)

        # f5 = 1.4983 * f0 is >= 41.2 Hz away from 1*f0 and 2*f0 for all guitar roots >= E2,
        # completely outside the 17.6 Hz sine mainlobe. A single note at f0 has no peak at f5.
        has_fifth = force_chord or (
            a_fifth >= 0.16 * a_root
            and a_fifth >= 4.0 * bg_ref
            and (a_fifth_3 >= 0.05 * a_root or a_fifth >= 0.25 * a_root)
        )
        if not has_fifth:
            return ("single", ())

        fifth_midi = min(127, root_midi + 7)
        maj3_score = self._third_support(spectrum, f0, _MAJ3_RATIO)
        min3_score = self._third_support(spectrum, f0, _MIN3_RATIO)

        if (
            maj3_score >= 0.18 * a_root
            and maj3_score >= 3.5 * bg_ref
            and maj3_score > 1.10 * min3_score
        ):
            third_midi = min(127, root_midi + 4)
            return ("major", (root_midi, third_midi, fifth_midi))
        if (
            min3_score >= 0.18 * a_root
            and min3_score >= 3.5 * bg_ref
            and min3_score > 1.10 * maj3_score
        ):
            third_midi = min(127, root_midi + 3)
            return ("minor", (root_midi, third_midi, fifth_midi))

        return ("power", (root_midi, fifth_midi))

    def _third_support(self, spectrum: np.ndarray, f0: float, ratio: float) -> float:
        """Measure support for a major or minor third using its 2nd-octave (`2 * ratio * f0`,
        +15/+16 semitones) and, for roots >= G2, 1st-octave (`1 * ratio * f0`) partials.
        Never use `4 * ratio * f0` because `4 * 2^(4/12) = 5.04 * f0` collides with the 5th
        harmonic (`5 * f0`) of the root string itself."""
        p2 = self._local_peak_near(spectrum, 2.0 * f0 * ratio)
        p1 = self._local_peak_near(spectrum, f0 * ratio) if f0 >= 95.0 else 0.0
        return max(p2, 0.85 * p1)

    def _local_peak_near(self, spectrum: np.ndarray, freq_hz: float) -> float:
        """Return the magnitude of the strongest true local maximum within `PEAK_SEARCH_CENTS`
        of `freq_hz`, or `0.0` if the spectrum is merely sloping from an adjacent note."""
        if freq_hz <= 0.0 or freq_hz >= self._sample_rate_hz / 2:
            return 0.0
        lo = max(1, math.floor(freq_hz / self._search_ratio / self._bin_hz))
        hi = min(len(spectrum) - 2, math.ceil(freq_hz * self._search_ratio / self._bin_hz))
        if lo > hi:
            return 0.0
        best_amp = 0.0
        for k in range(lo, hi + 1):
            mid = float(spectrum[k])
            left = float(spectrum[k - 1])
            right = float(spectrum[k + 1])
            if mid >= left and mid >= right and mid > 0.0:
                denom = left - 2.0 * mid + right
                offset = 0.5 * (left - right) / denom if denom < 0 else 0.0
                peak_hz = (k + max(-0.5, min(0.5, offset))) * self._bin_hz
                if (
                    peak_hz > 0.0
                    and abs(1200.0 * math.log2(peak_hz / freq_hz)) <= PEAK_SEARCH_CENTS
                ):
                    peak_amp = mid - 0.25 * (left - right) * offset
                    if peak_amp > best_amp:
                        best_amp = peak_amp
        return best_amp

    def _refine_frequency(self, spectrum: np.ndarray, target_hz: float) -> float:
        """Refine `target_hz` from the local peak of the 2nd or 1st harmonic for sub-Hz accuracy."""
        for harmonic in (2, 1):
            freq = target_hz * harmonic
            lo = max(1, math.floor(freq / self._search_ratio / self._bin_hz))
            hi = min(len(spectrum) - 2, math.ceil(freq * self._search_ratio / self._bin_hz))
            if lo > hi:
                continue
            best_k = -1
            best_mid = 0.0
            for k in range(lo, hi + 1):
                mid = float(spectrum[k])
                if (
                    mid >= float(spectrum[k - 1])
                    and mid >= float(spectrum[k + 1])
                    and mid > best_mid
                ):
                    best_mid = mid
                    best_k = k
            if best_k < 1:
                continue
            left = float(spectrum[best_k - 1])
            mid = float(spectrum[best_k])
            right = float(spectrum[best_k + 1])
            denom = left - 2.0 * mid + right
            offset = 0.5 * (left - right) / denom if denom < 0 and mid > 0 else 0.0
            refined = (best_k + max(-0.5, min(0.5, offset))) * self._bin_hz / harmonic
            if abs(1200.0 * math.log2(refined / target_hz)) <= PEAK_SEARCH_CENTS:
                return refined
        return target_hz

    @staticmethod
    def _clamp_midi(note: int) -> MidiNote:
        return max(0, min(127, note))

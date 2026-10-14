"""Key and tonal-center estimation over a rolling musical window (Step 5.2, §40, §90, §144, §145).

Runs on the analysis worker thread, never in the audio callback. Maintains an exponentially
decaying 12-bin pitch-class histogram (`C, C#, ..., B`) alongside a tonic-anchor histogram that
weights:
- note duration (sustained notes outweigh brief chromatic passing tones);
- metric placement (`downbeat_weight = 1.5x` near beat-1 downbeats or phrase starts after silence);
- phrase-ending stability (`phrase_end_weight = 1.3x` when a note resolves into a silence gap
  `>= phrase_end_silence_seconds`);
- onset velocity / attack strength and polyphonic guitar chord tones from Step 5.1;
- lowest-note bass root prior (guitar and bass riffs anchor the tonic at the bottom of the riff).

Scores all 12 pitch classes against minor/pentatonic/blues and major tonal profiles and applies
switching hysteresis (`switch_margin`, `min_confidence`) so chromatic passing tones never flip
a locked key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from jammate.analysis.notes import NOTE_NAMES
from jammate.config import AnalysisConfig, Instrument
from jammate.music.events import (
    KeyUpdate,
    MidiNote,
    OnsetEvent,
    PitchEvent,
    SampleFrame,
    SilenceEvent,
    frames_to_seconds,
)

type KeyMode = Literal["minor", "major"]
type BassRegister = Literal["low", "mid", "high"]

# Relative interval weights (0..11 semitones above candidate root) for minor/pentatonic/blues
# and major/Mixolydian guitar & bass playing (§145).
_MINOR_PROFILE = np.array(
    [1.00, -0.35, 0.25, 0.90, -0.45, 0.60, 0.25, 0.70, 0.25, 0.20, 0.55, -0.20],
    dtype=np.float64,
)
_MAJOR_PROFILE = np.array(
    [1.00, -0.45, 0.40, -0.60, 0.95, 0.50, -0.40, 0.70, -0.35, 0.45, 0.20, 0.40],
    dtype=np.float64,
)

_FLAT_TO_SHARP: dict[str, str] = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    """Current tonal-center estimate (§90, §145)."""

    root_pc: int | None = None
    key_name: str | None = None
    mode: KeyMode | None = None
    confidence: float = 0.0

    @property
    def label(self) -> str | None:
        """Human-readable label such as `'E minor'` or `'C major'`, or `None` when unknown."""
        if self.key_name is None:
            return None
        if self.mode is None:
            return self.key_name
        return f"{self.key_name} {self.mode}"


def classify_register(midi_note: MidiNote | None, instrument: Instrument = "bass") -> BassRegister:
    """Classify `midi_note` into `'low'`, `'mid'`, or `'high'` for the active instrument (§40)."""
    if midi_note is None:
        return "low"
    if instrument == "guitar":
        if midi_note < 52:  # E2 (40) .. D#3 (51)
            return "low"
        if midi_note < 64:  # E3 (52) .. D#4 (63)
            return "mid"
        return "high"  # E4 (64)+ lead / solo register
    # Bass: E1 (28) .. G4 (67)
    if midi_note <= 40:  # E1 (28) .. E2 (40)
        return "low"
    if midi_note < 52:  # F2 (41) .. D#3 (51)
        return "mid"
    return "high"  # E3 (52)+ high bass register


def parse_key_string(key_str: str) -> tuple[int, str, KeyMode]:
    """Parse a key string such as `'E minor'`, `'Em'`, `'C major'`, `'F#m'`, or `'A'`."""
    cleaned = key_str.strip()
    if not cleaned:
        raise ValueError("Key string must not be empty")
    parts = cleaned.split()
    token = parts[0]
    mode: KeyMode = "minor"
    if len(parts) >= 2:
        mode_str = parts[1].lower()
        if mode_str in ("major", "maj", "mjr"):
            mode = "major"
        elif mode_str in ("minor", "min", "m"):
            mode = "minor"
        else:
            raise ValueError(f"Unrecognized key mode in {key_str!r}")
    elif len(token) >= 2 and token.endswith("m") and not token.lower().endswith("maj"):
        token = token[:-1]
        mode = "minor"
    elif len(token) > 3 and token.lower().endswith("maj"):
        token = token[:-3]
        mode = "major"

    note_upper = token.upper()
    note_upper = _FLAT_TO_SHARP.get(note_upper, note_upper)
    if note_upper not in NOTE_NAMES:
        raise ValueError(f"Unrecognized key root in {key_str!r}")
    root_pc = NOTE_NAMES.index(note_upper)
    return root_pc, NOTE_NAMES[root_pc], mode


class KeyEstimator:
    """Streaming 12-bin pitch-class and tonal-center estimator (§90, §145)."""

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int = 48_000) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._window_seconds = (
            config.key_window_seconds
            if config.key_window_seconds != 8.0
            else config.key.window_seconds
        )
        self._hop_seconds = config.hop_frames / sample_rate_hz

        self._pitch_hist = np.zeros(12, dtype=np.float64)
        self._tonic_hist = np.zeros(12, dtype=np.float64)
        self._major_chord_hist = np.zeros(12, dtype=np.float64)
        self._minor_chord_hist = np.zeros(12, dtype=np.float64)

        self._active_note: MidiNote | None = None
        self._active_chord_notes: tuple[MidiNote, ...] = ()
        self._active_chord_type: str | None = None
        self._last_voiced_note: MidiNote | None = None
        self._silence_seconds: float = 1.0  # start ready for a phrase-initial note
        self._phrase_end_boosted: bool = True
        self._lowest_midi_seen: MidiNote | None = None
        self._recent_onset_strength: float = 0.7

        self._locked_root_pc: int | None = None
        self._locked_mode: KeyMode | None = None
        self._estimate = KeyEstimate()
        self._last_reported: tuple[str | None, KeyMode | None, float, BassRegister] = (
            None,
            None,
            0.0,
            "low",
        )
        self._manual_override: KeyEstimate | None = None
        if config.key.manual_key is not None:
            self.set_manual_key(config.key.manual_key)

    @property
    def estimate(self) -> KeyEstimate:
        """Current `KeyEstimate`."""
        if self._manual_override is not None:
            return self._manual_override
        return self._estimate

    def set_manual_key(self, manual_key: str | None) -> KeyEstimate:
        """Set or clear a manual key override (e.g. `'E minor'`, `'C major'`, or `None`)."""
        if manual_key is None:
            self._manual_override = None
            return self._estimate
        root_pc, key_name, mode = parse_key_string(manual_key)
        self._manual_override = KeyEstimate(
            root_pc=root_pc,
            key_name=key_name,
            mode=mode,
            confidence=1.0,
        )
        return self._manual_override

    def reset(self) -> None:
        """Reset all accumulated harmonic history while preserving any manual key override."""
        self._pitch_hist.fill(0.0)
        self._tonic_hist.fill(0.0)
        self._major_chord_hist.fill(0.0)
        self._minor_chord_hist.fill(0.0)
        self._active_note = None
        self._active_chord_notes = ()
        self._active_chord_type = None
        self._last_voiced_note = None
        self._silence_seconds = 1.0
        self._phrase_end_boosted = True
        self._lowest_midi_seen = None
        self._recent_onset_strength = 0.7
        self._locked_root_pc = None
        self._locked_mode = None
        self._estimate = KeyEstimate()
        self._last_reported = (None, None, 0.0, "low")

    def observe_note(
        self,
        midi_note: MidiNote,
        duration_seconds: float = 0.25,
        *,
        velocity: float = 0.8,
        is_downbeat: bool = False,
        is_phrase_end: bool = False,
        chord_notes: tuple[MidiNote, ...] = (),
        chord_type: str | None = None,
    ) -> KeyEstimate:
        """Directly feed a stabilized note observation (for unit tests and offline use)."""
        dt = max(0.01, float(duration_seconds))
        self._decay(dt)

        pc = midi_note % 12
        vel_scale = 0.75 + 0.50 * min(1.0, max(0.0, velocity))
        metric_scale = self._config.key.downbeat_weight if is_downbeat else 1.0
        phrase_scale = self._config.key.phrase_end_weight if is_phrase_end else 1.0

        weight = dt * vel_scale * metric_scale * phrase_scale
        self._pitch_hist[pc] += weight

        if self._lowest_midi_seen is None or midi_note <= self._lowest_midi_seen:
            self._lowest_midi_seen = midi_note
            self._tonic_hist[pc] += 0.35 * weight
        if is_downbeat:
            self._tonic_hist[pc] += 0.45 * weight
        if is_phrase_end:
            self._tonic_hist[pc] += 0.40 * weight
        if self._silence_seconds >= self._config.key.phrase_end_silence_seconds:
            self._tonic_hist[pc] += 0.35 * weight
            self._silence_seconds = 0.0

        self._accumulate_chord_tones(pc, chord_notes, chord_type, weight)
        self._last_voiced_note = midi_note
        self._recompute_estimate()
        return self.estimate

    def update_hop(
        self,
        frame_index: SampleFrame,
        note_event: PitchEvent | SilenceEvent | None,
        onset_event: OnsetEvent | None,
        *,
        is_downbeat: bool = False,
    ) -> KeyUpdate | None:
        """Process one analysis hop and emit a `KeyUpdate` when the key or confidence changes."""
        self._decay(self._hop_seconds)

        if onset_event is not None:
            self._recent_onset_strength = min(1.0, max(0.1, onset_event.strength))

        if isinstance(note_event, PitchEvent):
            self._active_note = note_event.midi_note
            self._active_chord_notes = note_event.chord_notes
            self._active_chord_type = note_event.chord_type
            self._last_voiced_note = note_event.midi_note

            pc = note_event.midi_note % 12
            vel_scale = 0.75 + 0.50 * self._recent_onset_strength
            phrase_start = self._silence_seconds >= self._config.key.phrase_end_silence_seconds
            metric_scale = (
                self._config.key.downbeat_weight if (is_downbeat or phrase_start) else 1.0
            )
            attack_weight = 0.14 * vel_scale * metric_scale
            self._pitch_hist[pc] += attack_weight

            if self._lowest_midi_seen is None or note_event.midi_note <= self._lowest_midi_seen:
                self._lowest_midi_seen = note_event.midi_note
                self._tonic_hist[pc] += 0.40 * attack_weight
            if is_downbeat or phrase_start:
                self._tonic_hist[pc] += 0.50 * attack_weight

            self._accumulate_chord_tones(
                pc, note_event.chord_notes, note_event.chord_type, attack_weight
            )
            self._silence_seconds = 0.0
            self._phrase_end_boosted = False
        elif isinstance(note_event, SilenceEvent):
            self._active_note = None
            self._active_chord_notes = ()
            self._active_chord_type = None

        if self._active_note is not None:
            pc = self._active_note % 12
            sustain_weight = self._hop_seconds
            self._pitch_hist[pc] += sustain_weight
            if self._lowest_midi_seen is not None and self._active_note <= self._lowest_midi_seen:
                self._tonic_hist[pc] += 0.30 * sustain_weight
            self._accumulate_chord_tones(
                pc, self._active_chord_notes, self._active_chord_type, sustain_weight
            )
            self._silence_seconds = 0.0
        else:
            self._silence_seconds += self._hop_seconds
            if (
                not self._phrase_end_boosted
                and self._last_voiced_note is not None
                and self._silence_seconds >= self._config.key.phrase_end_silence_seconds
            ):
                last_pc = self._last_voiced_note % 12
                boost = 0.18 * (self._config.key.phrase_end_weight - 1.0 + 1.0)
                self._pitch_hist[last_pc] += boost
                self._tonic_hist[last_pc] += boost
                self._phrase_end_boosted = True

        self._recompute_estimate()
        est = self.estimate
        register = classify_register(
            self._active_note if self._active_note is not None else self._last_voiced_note,
            self._config.instrument,
        )

        prev_key, prev_mode, prev_conf, prev_reg = self._last_reported
        key_changed = (est.key_name != prev_key) or (est.mode != prev_mode)
        conf_changed = abs(est.confidence - prev_conf) >= self._config.key.report_confidence_step
        reg_changed = register != prev_reg and self._active_note is not None

        if est.key_name is None and prev_key is None and not reg_changed:
            return None

        if key_changed or conf_changed or reg_changed:
            self._last_reported = (est.key_name, est.mode, est.confidence, register)
            return KeyUpdate(
                frame_index=frame_index,
                timestamp_seconds=frames_to_seconds(frame_index, self._sample_rate_hz),
                key_name=est.key_name,
                mode=est.mode,
                confidence=est.confidence,
                bass_register=register,
            )
        return None

    def _decay(self, dt_seconds: float) -> None:
        if dt_seconds <= 0.0 or self._window_seconds <= 0.0:
            return
        factor = math.exp(-dt_seconds / self._window_seconds)
        self._pitch_hist *= factor
        self._tonic_hist *= factor
        self._major_chord_hist *= factor
        self._minor_chord_hist *= factor

    def _accumulate_chord_tones(
        self,
        root_pc: int,
        chord_notes: tuple[MidiNote, ...],
        chord_type: str | None,
        base_weight: float,
    ) -> None:
        tone_weight = self._config.key.chord_tone_weight * base_weight
        for note in chord_notes:
            pc = note % 12
            if pc != root_pc:
                self._pitch_hist[pc] += tone_weight
        if chord_type == "major":
            self._major_chord_hist[root_pc] += base_weight
        elif chord_type == "minor":
            self._minor_chord_hist[root_pc] += base_weight

    def _recompute_estimate(self) -> None:
        total_pitch = float(np.sum(self._pitch_hist))
        if total_pitch < 0.12:
            self._estimate = KeyEstimate()
            return

        p = self._pitch_hist / total_pitch
        total_tonic = float(np.sum(self._tonic_hist))
        t = self._tonic_hist / total_tonic if total_tonic > 1e-9 else p

        scores: dict[tuple[int, KeyMode], float] = {}
        for root_pc in range(12):
            rel = np.roll(p, -root_pc)
            d0 = float(rel[0])
            root_gate = min(1.0, d0 / 0.15)
            tonic_prior = float(t[root_pc])

            scale_min = float(np.dot(_MINOR_PROFILE, rel))
            if self._minor_chord_hist[root_pc] > 0.05 * total_pitch:
                scale_min += 0.25
            if rel[3] < 0.05 and rel[4] < 0.05:
                scale_min += 0.04  # modal rock/power-chord default
            third_bonus_min = 0.18 if rel[3] >= 0.08 else 0.0
            scores[(root_pc, "minor")] = (
                0.62 * scale_min + 0.38 * tonic_prior + third_bonus_min
            ) * root_gate

            scale_maj = float(np.dot(_MAJOR_PROFILE, rel))
            if self._major_chord_hist[root_pc] > 0.05 * total_pitch:
                scale_maj += 0.25
            third_bonus_maj = 0.22 if rel[4] >= 0.08 else 0.0
            scores[(root_pc, "major")] = (
                0.62 * scale_maj + 0.38 * tonic_prior + third_bonus_maj
            ) * root_gate

        best_pair = max(scores, key=lambda k: scores[k])
        best_root, best_mode = best_pair
        best_score = scores[best_pair]

        # Runner-up across different roots (or opposite mode on the same root)
        other_scores = [s for pair, s in scores.items() if pair != best_pair]
        runner_up = max(other_scores) if other_scores else 0.0
        separation = max(0.0, best_score - runner_up)
        evidence_factor = min(1.0, total_pitch / 0.55)
        raw_conf = min(
            1.0,
            max(0.0, (0.55 * min(1.0, best_score) + 1.15 * separation) * evidence_factor),
        )

        if self._locked_root_pc is None or self._locked_mode is None:
            if raw_conf >= self._config.key.min_confidence:
                self._locked_root_pc = best_root
                self._locked_mode = best_mode
            else:
                self._estimate = KeyEstimate(confidence=raw_conf)
                return
        else:
            locked_pair = (self._locked_root_pc, self._locked_mode)
            locked_score = scores[locked_pair]
            if (
                best_pair != locked_pair
                and best_score >= locked_score + self._config.key.switch_margin
                and raw_conf >= self._config.key.min_confidence
            ):
                self._locked_root_pc = best_root
                self._locked_mode = best_mode
            else:
                best_root = self._locked_root_pc
                best_mode = self._locked_mode
                best_score = locked_score
                other_locked = [s for pair, s in scores.items() if pair != (best_root, best_mode)]
                sep_locked = max(0.0, best_score - (max(other_locked) if other_locked else 0.0))
                raw_conf = min(
                    1.0,
                    max(
                        self._config.key.min_confidence,
                        (0.55 * min(1.0, best_score) + 1.15 * sep_locked) * evidence_factor,
                    ),
                )

        self._estimate = KeyEstimate(
            root_pc=best_root,
            key_name=NOTE_NAMES[best_root],
            mode=best_mode,
            confidence=raw_conf,
        )

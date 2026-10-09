"""Note stabilization: turns per-hop pitch estimates into note changes (§26).

Raw pitch flickers (E2 E2 F2 E2), glides at the attack, passes through other notes in a slide
and jumps octaves. A new note is committed only when the last `min_note_seconds` of estimates
are all voiced, agree within `stability_cents`, sit within `tuning_tolerance_cents` of one note,
and are no more than `new_note_gate_db` below the recent playing level. That level follows the
window RMS up at once and falls back at `playing_level_release_db_per_second`; it keeps quiet
by-products of playing (harmonics after muting, open strings ringing after a fretted note,
touch noise) from becoming notes, without cutting off a note that decays naturally. Then:
- the current note is kept while the pitch stays within `hysteresis_cents` of it;
- an octave jump from the current note needs `octave_confirm_seconds` of the same evidence;
- re-picking the current note emits nothing (onsets cover re-attacks, Step 2.3);
- `silence_hold_seconds` of unvoiced hops end the note with a `SilenceEvent`.

Runs on the analysis worker thread. Pure logic: test it with hand-made `PitchFrame` sequences.
"""

import statistics
from collections import deque
from typing import NamedTuple

from jammate.analysis.notes import (
    CENTS_PER_SEMITONE,
    SEMITONES_PER_OCTAVE,
    hz_to_midi_float,
    midi_to_hz,
)
from jammate.analysis.pitch import PitchFrame
from jammate.audio.levels import SILENCE_FLOOR_DBFS
from jammate.config import AnalysisConfig
from jammate.music.events import (
    MidiNote,
    PitchEvent,
    SampleFrame,
    SilenceEvent,
    frames_to_seconds,
    seconds_to_frames,
)


class _VoicedHop(NamedTuple):
    frame_index: SampleFrame
    midi_float: float
    frequency_hz: float
    confidence: float


class NoteStabilizer:
    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._min_note_hops = self._hops(config.min_note_seconds, config.hop_frames)
        self._octave_hops = max(
            self._min_note_hops, self._hops(config.octave_confirm_seconds, config.hop_frames)
        )
        self._silence_hold_frames = seconds_to_frames(config.silence_hold_seconds, sample_rate_hz)
        self._stability_semitones = config.stability_cents / CENTS_PER_SEMITONE
        self._tuning_tolerance_semitones = config.tuning_tolerance_cents / CENTS_PER_SEMITONE
        self._hysteresis_semitones = config.hysteresis_cents / CENTS_PER_SEMITONE
        self._new_note_gate_db = config.new_note_gate_db
        self._release_db_per_hop = (
            config.playing_level_release_db_per_second * config.hop_frames / sample_rate_hz
        )
        self._playing_level_dbfs = SILENCE_FLOOR_DBFS
        self._recent: deque[_VoicedHop] = deque(maxlen=self._octave_hops)
        self._current_note: MidiNote | None = None
        self._unvoiced_since: SampleFrame | None = None

    def _hops(self, seconds: float, hop_frames: int) -> int:
        return max(1, round(seconds * self._sample_rate_hz / hop_frames))

    @property
    def current_note(self) -> MidiNote | None:
        return self._current_note

    def reset(self) -> None:
        """Forget the evidence after a gap in the input. The current note is kept; it ends
        through the normal silence path if the gap was silent."""
        self._recent.clear()

    def update(self, frame: PitchFrame) -> PitchEvent | SilenceEvent | None:
        """Feed one hop's estimate. Returns an event when the note musically changes."""
        self._playing_level_dbfs = max(
            frame.rms_dbfs, self._playing_level_dbfs - self._release_db_per_hop
        )
        if frame.frequency_hz is None:
            return self._update_unvoiced(frame.frame_index)
        self._unvoiced_since = None
        if frame.rms_dbfs < self._playing_level_dbfs - self._new_note_gate_db:
            self._recent.clear()  # too quiet to start a note, but it doesn't end the current one
            return None
        self._recent.append(
            _VoicedHop(
                frame.frame_index,
                hz_to_midi_float(frame.frequency_hz),
                frame.frequency_hz,
                frame.confidence,
            )
        )

        stable = self._stable_note(self._min_note_hops)
        if stable is None:
            return None
        note, median_midi = stable
        current = self._current_note
        if current is not None:
            if note == current or abs(median_midi - current) <= self._hysteresis_semitones:
                return None
            if abs(note - current) == SEMITONES_PER_OCTAVE:
                confirmed = self._stable_note(self._octave_hops)
                if confirmed is None or confirmed[0] != note:
                    return None
                return self._commit(note, self._octave_hops)
        return self._commit(note, self._min_note_hops)

    def _update_unvoiced(self, frame_index: SampleFrame) -> SilenceEvent | None:
        self._recent.clear()
        if self._current_note is None:
            return None
        if self._unvoiced_since is None:
            self._unvoiced_since = frame_index
        if frame_index - self._unvoiced_since < self._silence_hold_frames:
            return None
        silence_start = self._unvoiced_since
        self._current_note = None
        self._unvoiced_since = None
        return SilenceEvent(
            frame_index=silence_start,
            timestamp_seconds=frames_to_seconds(silence_start, self._sample_rate_hz),
        )

    def _stable_note(self, hop_count: int) -> tuple[MidiNote, float] | None:
        """The note the last `hop_count` voiced hops agree on, with their median MIDI value."""
        if len(self._recent) < hop_count:
            return None
        midi_values = [hop.midi_float for hop in list(self._recent)[-hop_count:]]
        if max(midi_values) - min(midi_values) > self._stability_semitones:
            return None  # gliding: a slide, bend or attack
        median_midi = statistics.median(midi_values)
        note = round(median_midi)
        if abs(median_midi - note) > self._tuning_tolerance_semitones:
            return None  # between two notes
        try:
            midi_to_hz(note)
        except ValueError:
            return None
        return note, median_midi

    def _commit(self, note: MidiNote, hop_count: int) -> PitchEvent:
        evidence = list(self._recent)[-hop_count:]
        start = evidence[0].frame_index
        self._current_note = note
        return PitchEvent(
            frame_index=start,
            timestamp_seconds=frames_to_seconds(start, self._sample_rate_hz),
            midi_note=note,
            frequency_hz=statistics.median(hop.frequency_hz for hop in evidence),
            confidence=statistics.median(hop.confidence for hop in evidence),
        )

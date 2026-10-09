"""Frequency ↔ MIDI note ↔ note name conversions. Pure functions, safe on any thread.

Notes are sounding pitch in scientific pitch notation, with A4 = 440 Hz = MIDI 69 (D6):
82.41 Hz → 40 → "E2". Open strings in standard tuning (the 12th fret is an octave higher):
- 4-string bass: E1 A1 D2 G2 (MIDI 28 33 38 43);
- guitar (D7): E2 A2 D3 G3 B3 E4 (MIDI 40 45 50 55 59 64).

Names use sharps (`midi_to_name(42) == "F#2"`). `name_to_midi` also accepts flats ("Gb2").
"""

import math
import operator
import re

from jammate.music.events import HIGHEST_MIDI_NOTE, LOWEST_MIDI_NOTE, MidiNote

A4_FREQUENCY_HZ = 440.0
A4_MIDI_NOTE: MidiNote = 69
SEMITONES_PER_OCTAVE = 12
CENTS_PER_SEMITONE = 100.0
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_NATURAL_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL_SEMITONES = {"": 0, "#": 1, "b": -1}
_NAME_PATTERN = re.compile(r"([A-Ga-g])([#b]?)(-?\d+)")


def hz_to_midi_float(frequency_hz: float) -> float:
    """Fractional MIDI note number: 82.41 Hz → 40.0017. Not limited to the 0-127 range."""
    if not (math.isfinite(frequency_hz) and frequency_hz > 0):
        raise ValueError(f"frequency_hz must be a positive finite number, got {frequency_hz}")
    return A4_MIDI_NOTE + SEMITONES_PER_OCTAVE * math.log2(frequency_hz / A4_FREQUENCY_HZ)


def hz_to_midi(frequency_hz: float) -> MidiNote:
    """Nearest MIDI note (exactly 50 cents rounds up): 82.41 Hz → 40."""
    return _checked_midi_note(math.floor(hz_to_midi_float(frequency_hz) + 0.5))


def midi_to_hz(midi_note: MidiNote) -> float:
    """Equal-tempered frequency of a MIDI note: 40 → 82.41 Hz."""
    midi_note = _checked_midi_note(midi_note)
    return A4_FREQUENCY_HZ * 2.0 ** ((midi_note - A4_MIDI_NOTE) / SEMITONES_PER_OCTAVE)


def midi_to_name(midi_note: MidiNote) -> str:
    """Note name with octave: 40 → "E2", 42 → "F#2", 60 → "C4"."""
    octave, pitch_class = divmod(_checked_midi_note(midi_note), SEMITONES_PER_OCTAVE)
    return f"{NOTE_NAMES[pitch_class]}{octave - 1}"


def name_to_midi(name: str) -> MidiNote:
    """Parse a note name: "E2" → 40, "F#2" and "Gb2" → 42. The letter may be lowercase."""
    match = _NAME_PATTERN.fullmatch(name.strip())
    if match is None:
        raise ValueError(f"not a note name like 'E2', 'F#2' or 'Bb1': {name!r}")
    letter, accidental, octave = match.groups()
    pitch_class = _NATURAL_PITCH_CLASSES[letter.upper()] + _ACCIDENTAL_SEMITONES[accidental]
    return _checked_midi_note((int(octave) + 1) * SEMITONES_PER_OCTAVE + pitch_class)


def cents_offset(frequency_hz: float, midi_note: MidiNote | None = None) -> float:
    """Cents from `midi_note` (default: the nearest note) to `frequency_hz`; positive = sharp.

    Against the nearest note the result lies in [-50, 50).
    """
    if midi_note is None:
        midi_note = hz_to_midi(frequency_hz)
    return CENTS_PER_SEMITONE * (hz_to_midi_float(frequency_hz) - _checked_midi_note(midi_note))


def _checked_midi_note(midi_note: MidiNote) -> MidiNote:
    """Return `midi_note` as a plain int (NumPy integers allowed, floats rejected) in 0-127."""
    midi_note = operator.index(midi_note)
    if not LOWEST_MIDI_NOTE <= midi_note <= HIGHEST_MIDI_NOTE:
        raise ValueError(
            f"MIDI note must be {LOWEST_MIDI_NOTE}-{HIGHEST_MIDI_NOTE}, got {midi_note}"
        )
    return midi_note

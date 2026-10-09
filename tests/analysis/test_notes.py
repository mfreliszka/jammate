import math

import numpy as np
import pytest

from jammate.analysis.notes import (
    cents_offset,
    hz_to_midi,
    hz_to_midi_float,
    midi_to_hz,
    midi_to_name,
    name_to_midi,
)

# (MIDI note, name, frequency in Hz): a 4-string bass's open strings, then the 12th fret.
BASS_STRINGS = [
    (28, "E1", 41.2034),
    (33, "A1", 55.0),
    (38, "D2", 73.4162),
    (43, "G2", 97.9989),
    (40, "E2", 82.4069),
    (45, "A2", 110.0),
    (50, "D3", 146.8324),
    (55, "G3", 195.9977),
]
# A guitar's open strings (D7), then the 12th fret.
GUITAR_STRINGS = [
    (40, "E2", 82.4069),
    (45, "A2", 110.0),
    (50, "D3", 146.8324),
    (55, "G3", 195.9977),
    (59, "B3", 246.9417),
    (64, "E4", 329.6276),
    (52, "E3", 164.8138),
    (57, "A3", 220.0),
    (62, "D4", 293.6648),
    (67, "G4", 391.9954),
    (71, "B4", 493.8833),
    (76, "E5", 659.2551),
]


def test_low_e_guitar_register_example() -> None:
    assert hz_to_midi(82.41) == 40
    assert midi_to_name(hz_to_midi(82.41)) == "E2"


@pytest.mark.parametrize(("midi_note", "name", "frequency_hz"), BASS_STRINGS + GUITAR_STRINGS)
def test_string_notes(midi_note: int, name: str, frequency_hz: float) -> None:
    assert midi_to_hz(midi_note) == pytest.approx(frequency_hz, abs=1e-4)
    assert hz_to_midi(frequency_hz) == midi_note
    assert midi_to_name(midi_note) == name
    assert name_to_midi(name) == midi_note


def test_reference_pitches() -> None:
    assert midi_to_hz(69) == 440.0
    assert midi_to_name(69) == "A4"
    assert midi_to_name(60) == "C4"
    assert midi_to_name(0) == "C-1"
    assert midi_to_name(127) == "G9"
    assert hz_to_midi_float(880.0) == pytest.approx(81.0)


def test_every_midi_note_round_trips() -> None:
    for midi_note in range(128):
        assert name_to_midi(midi_to_name(midi_note)) == midi_note
        assert hz_to_midi(midi_to_hz(midi_note)) == midi_note


def test_names_use_sharps() -> None:
    assert [midi_to_name(n) for n in range(40, 46)] == ["E2", "F2", "F#2", "G2", "G#2", "A2"]


@pytest.mark.parametrize(
    ("name", "midi_note"),
    [("F#2", 42), ("Gb2", 42), ("Bb1", 34), ("A#1", 34), ("eb2", 39), (" e1 ", 28), ("Cb4", 59)],
)
def test_name_to_midi_accepts_sharps_flats_and_lowercase(name: str, midi_note: int) -> None:
    assert name_to_midi(name) == midi_note


@pytest.mark.parametrize("name", ["", "E", "H2", "E##2", "E2.5", "Ab9", "Cb-1", "40"])
def test_name_to_midi_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        name_to_midi(name)


def test_rounds_to_the_nearest_note() -> None:
    e2_hz = midi_to_hz(40)

    assert hz_to_midi(e2_hz * 2 ** (49 / 1200)) == 40
    assert hz_to_midi(e2_hz * 2 ** (51 / 1200)) == 41
    assert hz_to_midi(e2_hz * 2 ** (-49 / 1200)) == 40
    assert hz_to_midi(e2_hz * 2 ** (-51 / 1200)) == 39


def test_cents_offset() -> None:
    e2_hz = midi_to_hz(40)

    assert cents_offset(82.41) == pytest.approx(0.065, abs=0.001)
    assert cents_offset(e2_hz * 2 ** (30 / 1200)) == pytest.approx(30.0)
    assert cents_offset(e2_hz * 2 ** (-20 / 1200)) == pytest.approx(-20.0)
    # Against an explicit note instead of the nearest one: F2 is 100 cents above E2.
    assert cents_offset(midi_to_hz(41), midi_note=40) == pytest.approx(100.0)


@pytest.mark.parametrize("frequency_hz", [0.0, -82.41, math.nan, math.inf])
def test_rejects_invalid_frequencies(frequency_hz: float) -> None:
    with pytest.raises(ValueError):
        hz_to_midi_float(frequency_hz)
    with pytest.raises(ValueError):
        cents_offset(frequency_hz)


def test_rejects_notes_outside_the_midi_range() -> None:
    for midi_note in (-1, 128):
        with pytest.raises(ValueError):
            midi_to_name(midi_note)
        with pytest.raises(ValueError):
            midi_to_hz(midi_note)
    with pytest.raises(ValueError):
        hz_to_midi(5.0)  # about MIDI -8
    with pytest.raises(ValueError):
        hz_to_midi(20_000.0)  # about MIDI 135


def test_accepts_numpy_numbers_but_not_float_notes() -> None:
    assert hz_to_midi(np.float32(82.41)) == 40
    assert midi_to_name(np.int64(40)) == "E2"
    assert type(hz_to_midi(np.float64(82.41))) is int
    with pytest.raises(TypeError):
        midi_to_name(40.0)  # type: ignore[arg-type]

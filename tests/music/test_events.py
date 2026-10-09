import dataclasses
import math
from collections.abc import Callable

import pytest

from jammate.music.events import (
    IntensityUpdate,
    OnsetEvent,
    PitchEvent,
    SilenceEvent,
    TempoUpdate,
    frames_to_seconds,
    seconds_to_frames,
)

AT_ONE_SECOND = {"frame_index": 48_000, "timestamp_seconds": 1.0}


def _pitch(**overrides: object) -> PitchEvent:
    values: dict[str, object] = {
        **AT_ONE_SECOND,
        "midi_note": 40,
        "frequency_hz": 82.41,
        "confidence": 0.9,
    }
    values.update(overrides)
    return PitchEvent(**values)  # type: ignore[arg-type]


def test_frames_to_seconds_and_back() -> None:
    assert frames_to_seconds(48_000, 48_000) == 1.0
    assert frames_to_seconds(128, 48_000) == pytest.approx(0.0026667, abs=1e-7)
    assert seconds_to_frames(1.5, 48_000) == 72_000
    assert seconds_to_frames(0.040, 48_000) == 1920  # one MRT2 frame
    for frame_index in (0, 1, 127, 384_000, 48_000 * 3600):
        assert seconds_to_frames(frames_to_seconds(frame_index, 48_000), 48_000) == frame_index


def test_seconds_to_frames_rounds_halves_up() -> None:
    assert seconds_to_frames(2.5 / 48_000, 48_000) == 3
    assert seconds_to_frames(2.4 / 48_000, 48_000) == 2
    assert seconds_to_frames(3.5 / 48_000, 48_000) == 4  # not banker's rounding


def test_conversions_reject_a_non_positive_sample_rate() -> None:
    with pytest.raises(ValueError):
        frames_to_seconds(1, 0)
    with pytest.raises(ValueError):
        seconds_to_frames(1.0, -48_000)


def test_events_are_frozen_and_compare_by_value() -> None:
    event = _pitch()

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.midi_note = 41  # type: ignore[misc]
    assert event == _pitch()
    assert hash(event) == hash(_pitch())
    assert event != _pitch(midi_note=41)


def test_events_require_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        OnsetEvent(48_000, 1.0, 0.5)  # type: ignore[misc]


def test_valid_events_construct() -> None:
    assert OnsetEvent(**AT_ONE_SECOND, strength=0.0).strength == 0.0
    assert TempoUpdate(**AT_ONE_SECOND, bpm=92.0, confidence=0.84, locked=True).locked
    assert TempoUpdate(**AT_ONE_SECOND, bpm=None, confidence=0.0, locked=False).bpm is None
    assert IntensityUpdate(**AT_ONE_SECOND, intensity=1.0).intensity == 1.0
    assert SilenceEvent(frame_index=0, timestamp_seconds=0.0).frame_index == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_index": -1},
        {"timestamp_seconds": -0.1},
        {"timestamp_seconds": math.nan},
        {"midi_note": -1},
        {"midi_note": 128},
        {"frequency_hz": 0.0},
        {"frequency_hz": math.inf},
        {"confidence": 1.01},
        {"confidence": math.nan},
    ],
)
def test_pitch_event_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _pitch(**overrides)


@pytest.mark.parametrize(
    "make_event",
    [
        lambda: OnsetEvent(**AT_ONE_SECOND, strength=-0.1),
        lambda: OnsetEvent(**AT_ONE_SECOND, strength=math.nan),
        lambda: TempoUpdate(**AT_ONE_SECOND, bpm=0.0, confidence=0.5, locked=False),
        lambda: TempoUpdate(**AT_ONE_SECOND, bpm=92.0, confidence=-0.1, locked=False),
        lambda: TempoUpdate(**AT_ONE_SECOND, bpm=None, confidence=0.9, locked=True),
        lambda: IntensityUpdate(**AT_ONE_SECOND, intensity=1.5),
        lambda: SilenceEvent(frame_index=-128, timestamp_seconds=0.0),
    ],
)
def test_other_events_reject_invalid_values(make_event: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        make_event()

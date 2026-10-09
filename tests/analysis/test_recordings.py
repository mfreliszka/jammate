"""The analysis pipeline on the developer's guitar recordings (truth: tests/data/README.md)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.analysis.notes import NOTE_NAMES, midi_to_name
from jammate.analysis.worker import EmittedEvent, analyze_offline
from jammate.config import AnalysisConfig
from jammate.music.events import PitchEvent, SilenceEvent

DATA = Path(__file__).resolve().parents[1] / "data"
GUITAR = AnalysisConfig(instrument="guitar")


def _analyze(name: str) -> list[EmittedEvent]:
    samples, sample_rate_hz = sf.read(DATA / name, dtype="float32")
    events, stats = analyze_offline(samples, GUITAR, sample_rate_hz)
    assert stats.errors == 0
    return events


def _timeline(events: list[EmittedEvent]) -> list[tuple[str, float]]:
    return [
        (
            midi_to_name(e.event.midi_note) if isinstance(e.event, PitchEvent) else "--",
            e.event.timestamp_seconds,
        )
        for e in events
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "guitar_open_strings.wav",
            [("E4", 1.52), ("B3", 4.75), ("G3", 7.61), ("D3", 10.76), ("A2", 13.86), ("E2", 16.70)],
        ),
        (
            "guitar_12th_fret.wav",
            [("E5", 0.60), ("B4", 3.35), ("G4", 6.70), ("D4", 10.26), ("A3", 13.54), ("E3", 17.20)],
        ),
    ],
)
def test_held_notes_are_detected_once_each_with_silence_between(
    name: str, expected: list[tuple[str, float]]
) -> None:
    timeline = _timeline(_analyze(name))

    notes = [(note, seconds) for note, seconds in timeline if note != "--"]
    assert [note for note, _ in notes] == [note for note, _ in expected]
    # Reported from the first stable estimate: shortly after the pick, never before it. The
    # 12th-fret D4 starts 46 cents sharp and is reported once its glide is within tolerance.
    for (_, detected), (_, start) in zip(notes, expected, strict=True):
        assert 0.0 <= detected - start < 0.2
    # Note, silence, note, ... (the 12th-fret take ends while its last note still rings).
    alternating = [name for note, _ in expected for name in (note, "--")]
    assert [note for note, _ in timeline] in (alternating, alternating[:-1])


@pytest.mark.parametrize(
    "name", ["guitar_riff_e2_g2_a2_92bpm.wav", "guitar_riff_e2_g2_a2_120bpm.wav"]
)
def test_riffs_report_each_note_change_without_slide_notes(name: str) -> None:
    events = _analyze(name)

    notes = [midi_to_name(e.event.midi_note) for e in events if isinstance(e.event, PitchEvent)]
    assert notes == ["E2", "G2", "A2", "E2", "G2", "A2"]
    assert isinstance(events[-1].event, SilenceEvent)


def test_chords_produce_no_wrong_notes() -> None:
    sections = [  # (start seconds, chord pitch classes), open major chords
        (0.0, {"E", "G#", "B"}),
        (3.5, {"G", "B", "D"}),
        (5.85, {"A", "C#", "E"}),
        (8.2, {"E", "G#", "B"}),
        (10.7, {"G", "B", "D"}),
        (13.0, {"A", "C#", "E"}),
    ]
    events = _analyze("guitar_chords_e_g_a_100bpm.wav")

    for emitted in events:
        if isinstance(emitted.event, PitchEvent):
            seconds = emitted.event.timestamp_seconds
            chord = [pcs for start, pcs in sections if start <= seconds][-1]
            assert NOTE_NAMES[emitted.event.midi_note % 12] in chord, (
                f"{midi_to_name(emitted.event.midi_note)} at {seconds:.2f} s"
            )


def test_decision_latency_is_under_100_ms() -> None:
    events = _analyze("guitar_riff_e2_g2_a2_120bpm.wav")

    latencies = [e.emitted_frame - e.event.frame_index for e in events]
    assert max(latencies) / 48_000 < 0.1
    assert np.median(latencies) / 48_000 == pytest.approx(0.062, abs=0.01)

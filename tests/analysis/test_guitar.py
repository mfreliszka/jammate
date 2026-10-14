"""Unit tests for electric guitar power-chord & chord root extraction (Step 5.1, D7, D12)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.ai.base import TOKEN_HELD, TOKEN_OFF, TOKEN_ONSET, NoteTracker
from jammate.ai.null import NullAccompaniment
from jammate.analysis.guitar import GuitarChordAnalyzer
from jammate.analysis.notes import midi_to_hz, name_to_midi
from jammate.analysis.pitch import PitchDetector, PitchFrame, yin
from jammate.analysis.stabilizer import NoteStabilizer
from jammate.audio.engine import AudioEngine
from jammate.audio.levels import amplitude_to_dbfs
from jammate.config import AnalysisConfig, JamMateConfig
from jammate.music.events import MidiNote, PitchEvent, SilenceEvent
from jammate.session import JamSession
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, SAMPLE_RATE_HZ, plucked_note, silence

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _make_chord_signal(
    notes: list[str | int],
    duration_seconds: float = 0.5,
    *,
    overdrive_drive: float = 0.0,
) -> np.ndarray:
    """Synthesize a strummed guitar chord by summing plucked guitar strings."""
    layers = [
        plucked_note(
            n,
            duration_seconds,
            amplitude=0.45,
            harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
        )
        for n in notes
    ]
    mix = np.sum(layers, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak > 0.0:
        mix *= 0.6 / peak
    if overdrive_drive > 0.0:
        mix = np.tanh(overdrive_drive * mix).astype(np.float32)
        peak2 = float(np.max(np.abs(mix)))
        if peak2 > 0.0:
            mix *= 0.6 / peak2
    return mix


def _feed_hops(
    detector: PitchDetector,
    samples: np.ndarray,
    hop_frames: int,
) -> list[PitchFrame]:
    frames: list[PitchFrame] = []
    for end in range(hop_frames, len(samples) + 1, hop_frames):
        hop = samples[end - hop_frames : end]
        frames.append(detector.process_hop(hop, end))
    return frames


@pytest.mark.parametrize(
    ("chord_label", "notes", "expected_root", "overdrive"),
    [
        ("E5_2note_clean", ["E2", "B2"], "E2", 0.0),
        ("E5_3note_clean", ["E2", "B2", "E3"], "E2", 0.0),
        ("E5_3note_overdrive", ["E2", "B2", "E3"], "E2", 2.5),
        ("A5_3note_clean", ["A2", "E3", "A3"], "A2", 0.0),
        ("A5_3note_overdrive", ["A2", "E3", "A3"], "A2", 2.5),
        ("G5_3note_clean", ["G2", "D3", "G3"], "G2", 0.0),
        ("G5_3note_overdrive", ["G2", "D3", "G3"], "G2", 2.5),
        ("D5_3note_clean", ["D3", "A3", "D4"], "D3", 0.0),
        ("D5_3note_overdrive", ["D3", "A3", "D4"], "D3", 2.5),
    ],
)
def test_power_chords_resolve_subharmonic_trap_and_extract_root_and_fifth(
    chord_label: str,
    notes: list[str],
    expected_root: str,
    overdrive: float,
) -> None:
    del chord_label
    cfg = AnalysisConfig(instrument="guitar")
    analyzer = GuitarChordAnalyzer(cfg, SAMPLE_RATE_HZ)
    samples = _make_chord_signal(notes, duration_seconds=0.4, overdrive_drive=overdrive)
    # Grab a steady window 80 ms after the attack
    start = int(0.08 * SAMPLE_RATE_HZ)
    window = samples[start : start + 4096]
    analyzer.push_hop(window)

    # Raw YIN extended down to 35 Hz often locks onto the 0.5*f0 subharmonic (e.g. E1=28 for E5)
    raw_yin = yin(window, SAMPLE_RATE_HZ, 35.0, 1400.0, cfg.yin_threshold)
    rms_dbfs = amplitude_to_dbfs(float(np.sqrt(np.mean(window * window))))
    estimate = analyzer.analyze(
        raw_yin,
        rms_dbfs=rms_dbfs,
        playing_level_dbfs=rms_dbfs,
    )

    expected_midi = name_to_midi(expected_root)
    assert estimate is not None
    assert estimate.root_midi == expected_midi
    assert estimate.chord_type == "power"
    assert expected_midi in estimate.chord_notes
    assert (expected_midi + 7) in estimate.chord_notes


@pytest.mark.parametrize(
    ("chord_name", "voicing", "expected_root", "expected_types"),
    [
        ("E_major", ["E2", "B2", "E3", "G#3", "B3", "E4"], "E2", ("major", "power")),
        ("E_minor", ["E2", "B2", "E3", "G3", "B3", "E4"], "E2", ("minor", "power")),
        ("G_major", ["G2", "B2", "D3", "G3", "B3", "G4"], "G2", ("major", "power")),
        ("A_major", ["A2", "E3", "A3", "C#4", "E4"], "A2", ("major", "power")),
        ("A_minor", ["A2", "E3", "A3", "C4", "E4"], "A2", ("minor", "power")),
        ("D_major", ["D3", "A3", "D4", "F#4"], "D3", ("major", "power")),
    ],
)
def test_open_major_and_minor_guitar_chords_detect_root_and_chord_tones(
    chord_name: str,
    voicing: list[str],
    expected_root: str,
    expected_types: tuple[str, ...],
) -> None:
    del chord_name
    cfg = AnalysisConfig(instrument="guitar")
    detector = PitchDetector(cfg, SAMPLE_RATE_HZ)
    stabilizer = NoteStabilizer(cfg, SAMPLE_RATE_HZ)

    signal = np.concatenate(
        [
            silence(0.05),
            _make_chord_signal(voicing, duration_seconds=0.45),
            silence(0.1),
        ]
    )
    events: list[PitchEvent] = []
    for hop in _feed_hops(detector, signal, cfg.hop_frames):
        ev = stabilizer.update(hop)
        if isinstance(ev, PitchEvent):
            events.append(ev)

    expected_midi = name_to_midi(expected_root)
    assert events, f"Expected at least one PitchEvent for {expected_root}"
    assert events[0].midi_note == expected_midi
    assert events[0].chord_type in expected_types
    assert len(events[0].chord_notes) >= 2
    assert expected_midi in events[0].chord_notes


def test_note_tracker_emits_multi_note_pianoroll_for_guitar_chords() -> None:
    tracker = NoteTracker()
    tracker.handle_event(
        PitchEvent(
            frame_index=4_800,
            timestamp_seconds=0.1,
            midi_note=40,
            frequency_hz=midi_to_hz(40),
            confidence=0.9,
            chord_notes=(40, 47, 52),
            chord_type="power",
        )
    )
    assert tracker.active_note == 40
    assert tracker.active_chord_notes == (40, 47, 52)

    roll1 = tracker.consume_pianoroll()
    assert roll1.shape == (128,)
    for note in (40, 47, 52):
        assert roll1[note] == TOKEN_ONSET
    assert roll1[45] == TOKEN_OFF

    roll2 = tracker.consume_pianoroll()
    for note in (40, 47, 52):
        assert roll2[note] == TOKEN_HELD

    tracker.handle_event(SilenceEvent(frame_index=24_000, timestamp_seconds=0.5))
    assert tracker.active_note is None
    assert tracker.active_chord_notes == ()
    roll3 = tracker.consume_pianoroll()
    assert np.all(roll3 == TOKEN_OFF)


def test_session_switches_instrument_and_updates_musical_state_chord_fields(
    fake_stream,
) -> None:
    del fake_stream
    session = JamSession(
        JamMateConfig(analysis=AnalysisConfig(instrument="guitar")),
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        accompaniment_factory=lambda ai_cfg, _sr: NullAccompaniment(
            style=ai_cfg.style,
            follow_improvise=ai_cfg.follow_improvise,
        ),
    )
    assert session.instrument == "guitar"

    session._handle_event(
        PitchEvent(
            frame_index=4_800,
            timestamp_seconds=0.1,
            midi_note=45,
            frequency_hz=110.0,
            confidence=0.91,
            chord_notes=(45, 52, 57),
            chord_type="power",
        )
    )
    assert session.state.current_note == 45
    assert session.state.chord_notes == (45, 52, 57)
    assert session.state.chord_type == "power"

    session.set_instrument("bass")
    assert session.instrument == "bass"
    assert session.worker.instrument == "bass"


def test_recorded_guitar_chords_e_g_a_detects_roots_without_octave_down_errors() -> None:
    wav_path = DATA_DIR / "guitar_chords_e_g_a_100bpm.wav"
    samples, sample_rate_hz = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples[:, 0]

    cfg = AnalysisConfig(instrument="guitar")
    detector = PitchDetector(cfg, sample_rate_hz)
    stabilizer = NoteStabilizer(cfg, sample_rate_hz)

    detected_notes: list[MidiNote] = []
    for hop in _feed_hops(detector, samples, cfg.hop_frames):
        ev = stabilizer.update(hop)
        if isinstance(ev, PitchEvent):
            detected_notes.append(ev.midi_note)

    # Every detected root on a standard-tuned guitar must be >= E2 (MIDI 40)
    assert detected_notes
    assert min(detected_notes) >= name_to_midi("E2")
    # The E, G, A chord progression roots (E2=40, G2=43, A2=45) are all detected
    detected_pcs = {n % 12 for n in detected_notes}
    assert {name_to_midi("E2") % 12, name_to_midi("G2") % 12, name_to_midi("A2") % 12}.issubset(
        detected_pcs
    )

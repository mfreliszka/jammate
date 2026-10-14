"""Unit tests for key/tonal-center estimation and register classification (Step 5.2, §90, §145)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jammate.analysis.key import (
    KeyEstimator,
    classify_register,
    parse_key_string,
)
from jammate.analysis.notes import name_to_midi
from jammate.analysis.worker import analyze_offline
from jammate.config import AnalysisConfig, KeyConfig, MusicalStateConfig
from jammate.music.events import KeyUpdate, PitchEvent
from jammate.music.state import MusicalStateManager, replay
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, riff

SAMPLE_RATE_HZ = 48_000
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ANALYZE_WAV_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_wav.py"


def _load_analyze_wav() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_wav_key_test", ANALYZE_WAV_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_wav_key_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("style_name", "note_names", "expected_key", "expected_mode"),
    [
        ("e_minor_riff", ["E2", "G2", "A2", "B2", "E2", "G2", "A2", "E2"], "E", "minor"),
        ("a_minor_pentatonic", ["A2", "C3", "D3", "E3", "G3", "A2", "C3", "A2"], "A", "minor"),
        ("blues_in_e", ["E2", "G2", "A2", "A#2", "B2", "G2", "E2", "E2"], "E", "minor"),
        ("c_major_bass", ["C2", "E2", "G2", "F2", "G2", "A2", "G2", "C2"], "C", "major"),
        ("c_major_guitar", ["C3", "D3", "E3", "F3", "G3", "E3", "C3", "C3"], "C", "major"),
    ],
)
def test_key_estimator_identifies_key_and_mode_on_riffs(
    style_name: str,
    note_names: list[str],
    expected_key: str,
    expected_mode: str,
) -> None:
    del style_name
    estimator = KeyEstimator(AnalysisConfig(instrument="guitar"), SAMPLE_RATE_HZ)

    for idx, name in enumerate(note_names * 2):
        is_downbeat = idx % len(note_names) == 0
        is_phrase_end = idx % len(note_names) == len(note_names) - 1
        est = estimator.observe_note(
            name_to_midi(name),
            duration_seconds=0.30 if is_downbeat else 0.22,
            velocity=0.85 if is_downbeat else 0.70,
            is_downbeat=is_downbeat,
            is_phrase_end=is_phrase_end,
        )

    assert est.key_name == expected_key
    assert est.mode == expected_mode
    assert est.confidence >= 0.65
    assert est.label == f"{expected_key} {expected_mode}"


def test_single_chromatic_passing_tones_do_not_flip_locked_key() -> None:
    estimator = KeyEstimator(AnalysisConfig(instrument="guitar"), SAMPLE_RATE_HZ)
    e_minor_notes = ["E2", "G2", "A2", "B2", "E2"]
    for _ in range(3):
        for idx, name in enumerate(e_minor_notes):
            estimator.observe_note(
                name_to_midi(name),
                duration_seconds=0.25,
                is_downbeat=(idx == 0),
            )

    locked = estimator.estimate
    assert locked.key_name == "E"
    assert locked.mode == "minor"

    # Inject brief chromatic passing tones (F2, D#2, C#3)
    for passing in ("F2", "D#2", "C#3"):
        after = estimator.observe_note(
            name_to_midi(passing),
            duration_seconds=0.08,
            velocity=0.5,
            is_downbeat=False,
        )
        assert after.key_name == "E"
        assert after.mode == "minor"


def test_chord_tones_disambiguate_major_and_minor_triads() -> None:
    estimator = KeyEstimator(AnalysisConfig(instrument="guitar"), SAMPLE_RATE_HZ)
    # Strum A major (A2=45, C#3=49, E3=52)
    est_maj = estimator.observe_note(
        45,
        duration_seconds=0.6,
        is_downbeat=True,
        chord_notes=(45, 49, 52),
        chord_type="major",
    )
    assert est_maj.key_name == "A"
    assert est_maj.mode == "major"

    estimator.reset()
    # Strum A minor (A2=45, C3=48, E3=52)
    est_min = estimator.observe_note(
        45,
        duration_seconds=0.6,
        is_downbeat=True,
        chord_notes=(45, 48, 52),
        chord_type="minor",
    )
    assert est_min.key_name == "A"
    assert est_min.mode == "minor"


def test_manual_key_override_and_parsing() -> None:
    assert parse_key_string("E minor") == (4, "E", "minor")
    assert parse_key_string("Em") == (4, "E", "minor")
    assert parse_key_string("C major") == (0, "C", "major")
    assert parse_key_string("Bb major") == (10, "A#", "major")
    assert parse_key_string("F#m") == (6, "F#", "minor")

    cfg = AnalysisConfig(instrument="guitar", key=KeyConfig(manual_key="C major"))
    estimator = KeyEstimator(cfg, SAMPLE_RATE_HZ)
    assert estimator.estimate.key_name == "C"
    assert estimator.estimate.mode == "major"
    assert estimator.estimate.confidence == 1.0

    # Clear manual override
    estimator.set_manual_key(None)
    assert estimator.estimate.key_name is None


@pytest.mark.parametrize(
    ("instrument", "note_name", "expected_register"),
    [
        ("guitar", "E2", "low"),
        ("guitar", "A2", "low"),
        ("guitar", "D#3", "low"),
        ("guitar", "E3", "mid"),
        ("guitar", "B3", "mid"),
        ("guitar", "E4", "high"),
        ("guitar", "A4", "high"),
        ("bass", "E1", "low"),
        ("bass", "A1", "low"),
        ("bass", "E2", "low"),
        ("bass", "G2", "mid"),
        ("bass", "D3", "mid"),
        ("bass", "E3", "high"),
        ("bass", "G3", "high"),
    ],
)
def test_classify_register_for_guitar_and_bass(
    instrument: str, note_name: str, expected_register: str
) -> None:
    midi_note = name_to_midi(note_name)
    assert classify_register(midi_note, instrument) == expected_register  # type: ignore[arg-type]


def test_musical_state_updates_key_and_bass_register() -> None:
    manager = MusicalStateManager(MusicalStateConfig(instrument="guitar"))
    # High guitar note E4 (64)
    state = manager.apply(
        PitchEvent(
            frame_index=4800,
            timestamp_seconds=0.1,
            midi_note=64,
            frequency_hz=329.63,
            confidence=0.95,
        )
    )
    assert state.bass_register == "high"

    state = manager.apply(
        KeyUpdate(
            frame_index=9600,
            timestamp_seconds=0.2,
            key_name="E",
            mode="minor",
            confidence=0.88,
            bass_register="high",
        )
    )
    assert state.estimated_key == "E"
    assert state.estimated_mode == "minor"
    assert state.harmonic_confidence == pytest.approx(0.88)
    assert state.key_label == "E minor"


def test_end_to_end_synthetic_riffs_populate_musical_state_key() -> None:
    rendered_em = riff(
        ["E2", "G2", "A2", "E2"],
        bpm=120,
        step_beats=0.5,
        repeats=4,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    events_em, _ = analyze_offline(
        rendered_em.samples, AnalysisConfig(instrument="guitar"), SAMPLE_RATE_HZ
    )
    state_em = replay(e.event for e in events_em)
    assert state_em.estimated_key == "E"
    assert state_em.estimated_mode == "minor"
    assert state_em.harmonic_confidence >= 0.65

    rendered_cmaj = riff(
        ["C2", "E2", "G2", "C2"],
        bpm=120,
        step_beats=0.5,
        repeats=4,
    )
    events_cmaj, _ = analyze_offline(
        rendered_cmaj.samples, AnalysisConfig(instrument="bass"), SAMPLE_RATE_HZ
    )
    state_cmaj = replay(e.event for e in events_cmaj)
    assert state_cmaj.estimated_key == "C"
    assert state_cmaj.estimated_mode == "major"
    assert state_cmaj.harmonic_confidence >= 0.65


def test_analyze_wav_reports_e_minor_on_recorded_guitar_riff(
    capsys: pytest.CaptureFixture[str],
) -> None:
    analyze_wav = _load_analyze_wav()
    wav_path = DATA_DIR / "guitar_riff_e2_g2_a2_120bpm.wav"
    rc = analyze_wav.main([str(wav_path), "--instrument", "guitar"])
    assert rc == 0
    out = capsys.readouterr().out
    summary_line = out.splitlines()[-1]
    assert "Key: E minor, confidence " in summary_line

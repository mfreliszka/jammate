"""Tests for `scripts/render_jam.py` (Step 3.5)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import soundfile as sf

from jammate.config import AnalysisConfig, DrumConfig, JamMateConfig, TempoConfig
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, riff

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_jam.py"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_jam", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_jam"] = module
    spec.loader.exec_module(module)
    return module


render_jam = _load_script()


def test_render_jam_offline_on_synthetic_riff_enters_and_renders_stereo() -> None:
    rendered_riff = riff(
        ["E2", "G2", "A2", "E2"],
        bpm=100.0,
        step_beats=0.5,
        repeats=8,
        lead_in_seconds=0.3,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    config = JamMateConfig(
        analysis=AnalysisConfig(instrument="guitar"),
        drums=DrumConfig(style="rock"),
    )
    stereo, summary = render_jam.render_jam_offline(rendered_riff.samples, config)

    assert stereo.shape == (len(rendered_riff.samples), 2)
    assert stereo.dtype == "float32"
    assert summary.tempo_locked is True
    assert summary.locked_bpm == pytest.approx(100.0, abs=1.5)
    assert summary.drummer_entered is True
    assert summary.entry_bar is not None and summary.entry_bar >= 2
    assert summary.events_scheduled > 0
    assert summary.dropped_events == 0
    assert summary.stolen_voices == 0
    assert summary.peak_dbfs <= 0.0


def test_render_jam_cli_writes_stereo_wav_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    in_path = DATA_DIR / "guitar_riff_e2_g2_a2_92bpm.wav"
    out_path = tmp_path / "jam_92.wav"

    rc = render_jam.main(
        [
            str(in_path),
            str(out_path),
            "--instrument",
            "guitar",
            "--style",
            "stoner_rock",
        ]
    )
    assert rc == 0
    assert out_path.exists()

    info = sf.info(out_path)
    assert info.samplerate == 48_000
    assert info.channels == 2

    stdout = capsys.readouterr().out
    assert "Wrote " in stdout
    assert "Drummer:  entered bar" in stdout
    assert "dropped 0" in stdout


def test_render_jam_manual_bpm_enters_at_bar_1() -> None:
    rendered_riff = riff(
        ["E2", "G2"],
        bpm=92.0,
        step_beats=1.0,
        repeats=2,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    config = JamMateConfig(
        analysis=AnalysisConfig(instrument="guitar", tempo=TempoConfig(manual_bpm=92.0)),
        drums=DrumConfig(style="blues"),
    )
    _, summary = render_jam.render_jam_offline(rendered_riff.samples, config)
    assert summary.tempo_locked is True
    assert summary.locked_bpm == pytest.approx(92.0)
    assert summary.drummer_entered is True
    assert summary.entry_bar == 1
    assert summary.events_scheduled > 0


def test_render_jam_rejects_bad_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.wav"
    out_path = tmp_path / "out.wav"
    assert render_jam.main([str(missing), str(out_path)]) == 1
    assert "not found" in capsys.readouterr().err

    assert render_jam.main([str(missing), str(out_path), "--drums-gain", "9.0"]) == 2
    assert "invalid configuration" in capsys.readouterr().err

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

from tests.signals import GUITAR_HARMONIC_AMPLITUDES, plucked_note, riff, silence

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_wav.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_wav", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_wav"] = module
    spec.loader.exec_module(module)
    return module


analyze_wav = _load_script()


def _write_wav(path: Path, sample_rate_hz: int = 48_000) -> Path:
    tone = plucked_note("A2", 1.0, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    sf.write(path, np.concatenate([silence(0.2), tone, silence(0.3)]), sample_rate_hz)
    return path


def test_prints_the_event_timeline_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_wav(tmp_path / "a2.wav")

    assert analyze_wav.main([str(path), "--instrument", "guitar"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "guitar 60-1400 Hz" in lines[0]
    timeline = [line.split("s  ", 1)[1].split()[0] for line in lines[1:-1]]
    assert timeline == ["onset", "state", "A2", "state", "--", "state"]
    assert lines[1].startswith("   0.199 s    onset  peak   -6.0 dBFS")
    assert "decided" in lines[3]
    assert lines[-1].startswith("1 notes (A2 x1), 1 silences. 1 onsets. Intensity max")
    assert "Tempo: no estimate." in lines[-1]


def test_tempo_locks_are_on_the_timeline_and_in_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rendered = riff(
        ["E2", "G2", "A2", "E2"],
        bpm=100,
        step_beats=0.5,
        repeats=6,
        lead_in_seconds=0.5,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )
    path = tmp_path / "riff.wav"
    sf.write(path, rendered.samples, 48_000)

    analyze_wav.main([str(path), "--instrument", "guitar"])
    timeline = capsys.readouterr().out
    analyze_wav.main([str(path), "--instrument", "guitar", "--tempo"])
    every_update = capsys.readouterr().out

    assert timeline.count(" tempo ") == 1
    assert " BPM locked  conf " in timeline
    assert "Tempo: locked at 100.0 BPM" in timeline.splitlines()[-1]
    assert "first locked at" in timeline.splitlines()[-1]
    assert every_update.count(" tempo ") > 3
    assert " BPM search  conf " in every_update


def test_hops_option_prints_every_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_wav(tmp_path / "a2.wav")

    analyze_wav.main([str(path), "--instrument", "guitar", "--hops"])

    output = capsys.readouterr().out
    assert output.count("    hop  ") == 150  # 1.5 s of 10 ms hops
    assert "unvoiced" in output


def test_rejects_other_sample_rates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_wav(tmp_path / "a2_44k.wav", sample_rate_hz=44_100)

    assert analyze_wav.main([str(path)]) == 1
    assert "expects 48000 Hz" in capsys.readouterr().err


def test_rejects_invalid_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_wav(tmp_path / "a2.wav")

    assert analyze_wav.main([str(path), "--yin-threshold", "2"]) == 2

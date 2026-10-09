from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.analysis.onset import OnsetDetector
from jammate.analysis.worker import analyze_offline
from jammate.config import AnalysisConfig
from jammate.music.events import OnsetEvent
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, noise, plucked_note, riff, silence

SR = 48_000
GUITAR = AnalysisConfig(instrument="guitar")
BASS = AnalysisConfig(instrument="bass")
HOP = GUITAR.hop_frames
DATA = Path(__file__).resolve().parents[1] / "data"


def _onsets(samples: np.ndarray, config: AnalysisConfig = GUITAR) -> list[OnsetEvent]:
    events, _ = analyze_offline(samples, config, SR)
    return [e.event for e in events if isinstance(e.event, OnsetEvent)]


def _with_noise_floor(samples: np.ndarray) -> np.ndarray:
    return samples + noise(len(samples) / SR, rms_dbfs=-65.0)


@pytest.mark.parametrize(
    ("instrument", "bpm", "step_beats", "gate"),
    [
        ("guitar", 92, 0.5, 0.9),
        ("guitar", 120, 0.5, 0.9),
        ("guitar", 120, 0.25, 0.9),  # 16ths
        ("guitar", 120, 0.5, 1.0),  # legato: no gap between notes
        ("guitar", 77, 1.0, 0.9),
        ("bass", 92, 0.5, 0.9),
        ("bass", 120, 0.5, 0.9),
        ("bass", 120, 0.25, 0.9),
        ("bass", 120, 0.5, 1.0),
    ],
)
def test_synthetic_riff_onsets_are_found_within_one_hop(
    instrument: str, bpm: float, step_beats: float, gate: float
) -> None:
    if instrument == "guitar":
        notes, config = ["E2", "G2", "A2", "E2", "E2", "D3"], GUITAR
        timbre = {"harmonic_amplitudes": GUITAR_HARMONIC_AMPLITUDES}
    else:
        notes, config, timbre = ["E1", "G1", "A1", "E1", "E1", "D2"], BASS, {}
    rendered = riff(
        notes,
        bpm=bpm,
        step_beats=step_beats,
        repeats=4,
        gate=gate,
        lead_in_seconds=0.3,
        tail_seconds=0.5,
        **timbre,
    )

    onsets = _onsets(_with_noise_floor(rendered.samples), config)

    # Every attack once (repeated notes too), each within one hop of the truth.
    assert len(onsets) == len(rendered.onset_frames)
    for onset, true_frame in zip(onsets, rendered.onset_frames, strict=True):
        assert abs(onset.frame_index - true_frame) <= HOP


def test_onset_strength_is_the_attack_peak() -> None:
    samples = np.concatenate([silence(0.2), plucked_note("A2", 0.5, amplitude=0.25)])

    (onset,) = _onsets(_with_noise_floor(samples))

    assert onset.strength == pytest.approx(0.25, abs=0.01)
    assert onset.timestamp_seconds == pytest.approx(0.2, abs=0.002)


def test_one_attack_registers_once_within_the_refractory_period() -> None:
    # Pick contact, then the note 35 ms later, as in the 120 BPM recording.
    click = plucked_note("A2", 0.035, amplitude=0.3)
    note = plucked_note("A2", 0.5, amplitude=0.5)
    samples = np.concatenate([silence(0.2), click, note, silence(0.2)])

    assert len(_onsets(_with_noise_floor(samples))) == 1


@pytest.mark.parametrize("rms_dbfs", [-30.0, -40.0, -65.0])
def test_steady_noise_has_no_onsets_after_it_starts(rms_dbfs: float) -> None:
    onsets = _onsets(noise(10.0, rms_dbfs=rms_dbfs, seed=7))

    assert [onset.frame_index for onset in onsets] in ([], [0])


def test_extremely_loud_noise_gives_only_rare_onsets() -> None:
    # White noise at -20 dBFS (far louder than any real hiss): an occasional false onset.
    onsets = _onsets(noise(10.0, rms_dbfs=-20.0, seed=7))

    assert len([onset for onset in onsets if onset.frame_index > 0]) <= 3


def test_attacks_below_the_minimum_level_are_ignored() -> None:
    quiet = plucked_note("A2", 0.5, amplitude=10 ** (-56 / 20))  # window RMS below -50 dBFS
    samples = np.concatenate([silence(0.2), quiet, silence(0.2), quiet])

    assert _onsets(_with_noise_floor(samples)) == []


def test_attacks_far_below_the_playing_level_are_ignored() -> None:
    loud = plucked_note("A2", 0.5, amplitude=0.5)
    soft = plucked_note("D3", 0.5, amplitude=0.5 * 10 ** (-25 / 20))  # 25 dB down: a pull-off
    samples = np.concatenate([silence(0.2), loud, silence(0.1), soft, silence(0.2)])

    onsets = _onsets(_with_noise_floor(samples))

    assert len(onsets) == 1


def test_detector_rejects_a_hop_longer_than_its_window() -> None:
    with pytest.raises(ValueError, match="hop_frames"):
        OnsetDetector(AnalysisConfig(instrument="guitar", hop_frames=2048), SR)


@pytest.mark.parametrize(
    ("name", "bpm", "first_note_seconds"),
    [("guitar_riff_e2_g2_a2_92bpm.wav", 92, 2.67), ("guitar_riff_e2_g2_a2_120bpm.wav", 120, 1.22)],
)
def test_recorded_riffs_give_one_onset_per_note(
    name: str, bpm: float, first_note_seconds: float
) -> None:
    samples, sample_rate_hz = sf.read(DATA / name, dtype="float32")
    events, _ = analyze_offline(samples, GUITAR, sample_rate_hz)
    onsets = [e.event.timestamp_seconds for e in events if isinstance(e.event, OnsetEvent)]

    playing = [t for t in onsets if t >= first_note_seconds - 0.1]
    # 48 picked 8th notes, plus at most a few double-registered picks and slide noises.
    assert 48 <= len(playing) <= 52
    eighth_seconds = 60 / bpm / 2
    assert np.median(np.diff(playing)) == pytest.approx(eighth_seconds, rel=0.03)

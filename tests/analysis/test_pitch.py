import math

import numpy as np
import pytest

from jammate.analysis.notes import cents_offset, midi_to_hz, name_to_midi
from jammate.analysis.pitch import (
    PitchDetector,
    yin,
    yin_lag_range,
    yin_window_frames,
)
from jammate.config import AnalysisConfig
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, noise, plucked_note, silence

SR = 48_000
GUITAR = AnalysisConfig(instrument="guitar")
BASS = AnalysisConfig(instrument="bass")


def _window(samples: np.ndarray, min_hz: float, offset_seconds: float = 0.1) -> np.ndarray:
    start = round(offset_seconds * SR)
    return samples[start : start + yin_window_frames(SR, min_hz)]


def test_window_and_lag_range_follow_the_pitch_range() -> None:
    assert yin_lag_range(SR, 60.0, 1400.0) == (34, 800)
    assert yin_window_frames(SR, 60.0) == 1602  # 33 ms
    assert yin_window_frames(SR, 35.0) == 2746  # 57 ms


def _cents_between(measured_hz: float, expected_hz: float) -> float:
    return 1200 * math.log2(measured_hz / expected_hz)


@pytest.mark.parametrize("frequency_hz", [41.2, 55.0, 82.41, 146.8, 329.6, 659.3, 1318.5])
def test_yin_finds_pure_tones_within_a_few_cents(frequency_hz: float) -> None:
    tone = 0.5 * np.sin(2 * math.pi * frequency_hz * np.arange(SR) / SR)

    result = yin(_window(tone, 35.0), SR, 35.0, 1400.0, threshold=0.2)

    assert result.frequency_hz is not None
    assert abs(_cents_between(result.frequency_hz, frequency_hz)) < 3
    assert result.confidence > 0.95


@pytest.mark.parametrize("note", ["E2", "A2", "E3", "E4", "E5"])
def test_yin_is_not_fooled_by_a_strong_second_harmonic(note: str) -> None:
    # Right after the attack, the guitar timbre's spectral peak is an octave above the note.
    tone = plucked_note(note, 0.5, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)

    result = yin(_window(tone, 60.0, offset_seconds=0.01), SR, 60.0, 1400.0, threshold=0.2)

    assert result.frequency_hz is not None
    assert abs(_cents_between(result.frequency_hz, midi_to_hz(name_to_midi(note)))) < 10


def test_yin_reports_noise_and_silence_as_unvoiced() -> None:
    noisy = yin(_window(noise(1.0, rms_dbfs=-20.0), 60.0), SR, 60.0, 1400.0, threshold=0.2)
    silent = yin(_window(silence(1.0), 60.0), SR, 60.0, 1400.0, threshold=0.2)

    assert noisy.frequency_hz is None
    assert noisy.confidence < 0.8
    assert silent.frequency_hz is None


def test_yin_rejects_a_window_too_short_for_the_lowest_pitch() -> None:
    with pytest.raises(ValueError, match="too short"):
        yin(np.zeros(1000), SR, 35.0, 400.0, threshold=0.2)


def _run_detector(detector: PitchDetector, samples: np.ndarray, hop: int = 480) -> list:
    frames = []
    for start in range(0, len(samples) - hop + 1, hop):
        frames.append(detector.process_hop(samples[start : start + hop], start + hop))
    return frames


def test_detector_tracks_a_note_and_reports_levels_and_positions() -> None:
    detector = PitchDetector(GUITAR, SR)
    tone = plucked_note("A2", 1.0, amplitude=0.5, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)

    frames = _run_detector(detector, tone)

    assert [frame.frame_index for frame in frames[:3]] == [480, 960, 1440]
    settled = frames[10:60]  # 0.1-0.6 s: the window is full of the note
    assert all(frame.frequency_hz is not None for frame in settled)
    assert all(abs(cents_offset(frame.frequency_hz, 45)) < 10 for frame in settled)
    assert max(frame.peak_dbfs for frame in frames) == pytest.approx(-6.0, abs=0.1)
    assert -20 < settled[0].rms_dbfs < -6


def test_detector_gates_quiet_input() -> None:
    detector = PitchDetector(GUITAR, SR)
    quiet = plucked_note("A2", 0.5, amplitude=10 ** (-50 / 20))  # peak -50 dBFS < -45 gate

    frames = _run_detector(detector, quiet)

    assert all(frame.frequency_hz is None and frame.confidence == 0.0 for frame in frames)
    assert all(frame.rms_dbfs < GUITAR.level_gate_dbfs for frame in frames)


def test_detector_bass_range_reaches_low_e1() -> None:
    detector = PitchDetector(BASS, SR)

    frames = _run_detector(detector, plucked_note("E1", 1.0))

    assert detector.window_frames == 2746
    assert abs(cents_offset(frames[30].frequency_hz, 28)) < 10


def test_detector_rejects_a_hop_longer_than_its_window() -> None:
    with pytest.raises(ValueError, match="hop_frames"):
        PitchDetector(AnalysisConfig(instrument="guitar", hop_frames=4800), SR)


def test_detector_ignores_dc_and_rumble() -> None:
    # A DC step leaves a smooth decaying curve after the filter, which YIN alone would call
    # periodic. The level is measured in band, so it is gated instead.
    detector = PitchDetector(GUITAR, SR)
    rumble = 0.3 * np.sin(2 * math.pi * 8.0 * np.arange(SR) / SR)

    frames = _run_detector(detector, np.full(SR, 0.5) + rumble)

    assert all(frame.frequency_hz is None for frame in frames[20:])

import numpy as np
import pytest

from jammate.analysis.notes import midi_to_hz
from jammate.audio.levels import measure_level
from tests.signals import (
    GUITAR_HARMONIC_AMPLITUDES,
    noise,
    onset_times,
    plucked_note,
    riff,
    silence,
)

SR = 48_000


def _dominant_frequency_hz(samples: np.ndarray, sample_rate_hz: int = SR) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / sample_rate_hz)[np.argmax(spectrum)])


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


# --- plucked_note --------------------------------------------------------------------------------


def test_plucked_note_shape_level_and_pitch() -> None:
    tone = plucked_note("E1", 2.0, amplitude=0.5)

    assert tone.dtype == np.float32
    assert tone.shape == (2 * SR,)
    assert np.max(np.abs(tone)) == pytest.approx(0.5)
    # 2 s of signal gives 0.5 Hz FFT bins.
    assert _dominant_frequency_hz(tone) == pytest.approx(midi_to_hz(28), abs=0.5)


def test_plucked_note_accepts_midi_numbers_and_detuning() -> None:
    assert np.array_equal(plucked_note(40, 0.5), plucked_note("E2", 0.5))
    sharp = plucked_note("A1", 2.0, cents=50)

    assert _dominant_frequency_hz(sharp) == pytest.approx(55.0 * 2 ** (50 / 1200), abs=0.5)


def test_plucked_note_decays_and_starts_and_ends_without_a_click() -> None:
    tone = plucked_note("A1", 2.0)
    tenth = len(tone) // 10

    assert _rms(tone[-tenth:]) < 0.25 * _rms(tone[:tenth])
    assert tone[0] == 0.0
    assert abs(tone[-1]) < 1e-6
    # The 2 ms attack is short: the note is near full level within 5 ms.
    assert np.max(np.abs(tone[: SR // 200])) > 0.3


def test_plucked_note_weak_fundamental_puts_the_peak_an_octave_up() -> None:
    tone = plucked_note("E1", 2.0, harmonic_amplitudes=(0.2, 1.0, 0.4))

    assert _dominant_frequency_hz(tone) == pytest.approx(2 * midi_to_hz(28), abs=0.5)


def test_guitar_timbre_peaks_an_octave_up_at_the_attack() -> None:
    tone = plucked_note("E2", 2.0, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    attack = tone[: SR // 10]  # the first 100 ms, which pitch detection sees first
    spectrum = np.abs(np.fft.rfft(attack * np.hanning(len(attack))))
    frequencies_hz = np.fft.rfftfreq(len(attack), 1 / SR)

    assert _dominant_frequency_hz(attack) == pytest.approx(2 * midi_to_hz(40), abs=10.0)
    # The fundamental is still strong (the true pitch is E2, not E3) and wins over the whole note.
    fundamental_bin = np.argmin(np.abs(frequencies_hz - midi_to_hz(40)))
    assert spectrum[fundamental_bin] > 0.5 * np.max(spectrum)
    assert _dominant_frequency_hz(tone) == pytest.approx(midi_to_hz(40), abs=0.5)


def test_plucked_note_skips_harmonics_above_nyquist() -> None:
    # At 8 kHz sampling, harmonics of A4 from the 10th (4.4 kHz) up would alias.
    tone = plucked_note("A4", 1.0, harmonic_amplitudes=(1.0,) * 20, sample_rate_hz=8_000)

    spectrum = np.abs(np.fft.rfft(tone))
    frequencies_hz = np.fft.rfftfreq(len(tone), 1 / 8_000)
    assert np.all(np.isfinite(tone))
    # Only the harmonics 440 … 3960 Hz exist; nothing aliased below the fundamental.
    assert np.max(spectrum[frequencies_hz < 400]) < 0.01 * np.max(spectrum)


def test_plucked_note_handles_empty_and_tiny_durations() -> None:
    assert plucked_note("E1", 0.0).shape == (0,)
    assert plucked_note("E1", 1 / SR).shape == (1,)


def test_plucked_note_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        plucked_note("E1", 1.0, amplitude=0.0)
    with pytest.raises(ValueError):
        plucked_note("E1", 1.0, decay_seconds=0.0)
    with pytest.raises(ValueError):
        plucked_note("H1", 1.0)


# --- riff ----------------------------------------------------------------------------------------


def test_riff_places_onsets_on_exact_beats() -> None:
    rendered = riff(["E1", "G1", "A1", "E1"], bpm=120)

    assert rendered.onset_frames == (0, 24_000, 48_000, 72_000)
    assert rendered.midi_notes == (28, 31, 33, 28)
    assert rendered.end_frames == (21_600, 45_600, 69_600, 93_600)  # gate 0.9
    assert rendered.samples.shape == (96_000,)
    assert rendered.samples.dtype == np.float32
    assert rendered.bpm == 120
    assert rendered.sample_rate_hz == SR


def test_riff_has_no_rounding_drift_at_fractional_frames_per_beat() -> None:
    # 92 BPM = 31,304.35 frames per beat: onsets are rounded per beat, not accumulated.
    rendered = riff(["E1"], bpm=92, repeats=400)

    assert rendered.onset_frames[1] == 31_304
    assert rendered.onset_frames[3] == 93_913
    assert rendered.onset_frames[-1] == round(399 * 60 * SR / 92)


def test_riff_audio_matches_its_ground_truth() -> None:
    rendered = riff(["E1", None, "A1"], bpm=100, gate=0.5, lead_in_seconds=0.25, tail_seconds=0.5)
    samples = rendered.samples

    assert rendered.onset_frames == (12_000, 69_600)  # the rest has no onset
    assert len(samples) == 12_000 + 3 * 28_800 + 24_000
    assert np.all(samples[:12_000] == 0.0)  # lead-in
    for onset_frame, end_frame in zip(rendered.onset_frames, rendered.end_frames, strict=True):
        assert np.max(np.abs(samples[onset_frame : onset_frame + 480])) > 0.1
        assert np.all(samples[end_frame : end_frame + 1000] == 0.0)  # gated off
    assert np.all(samples[rendered.end_frames[0] : rendered.onset_frames[1]] == 0.0)  # rest
    assert np.all(samples[rendered.end_frames[-1] :] == 0.0)  # tail
    assert np.max(np.abs(samples)) == pytest.approx(0.5)


def test_riff_step_lengths_and_repeats() -> None:
    rendered = riff([("E1", 1.5), ("G1", 0.5), (None, 1.0), "A1"], bpm=120, repeats=2)

    assert rendered.onset_frames == (0, 36_000, 72_000, 96_000, 132_000, 168_000)
    assert rendered.midi_notes == (28, 31, 33) * 2
    assert len(rendered.samples) == 2 * 4 * 24_000


def test_riff_each_note_has_its_own_pitch() -> None:
    rendered = riff(["E1", "A1", "D2", "G2"], bpm=60, gate=1.0)

    for onset_frame, end_frame, midi_note in zip(
        rendered.onset_frames, rendered.end_frames, rendered.midi_notes, strict=True
    ):
        note = rendered.samples[onset_frame:end_frame]
        assert _dominant_frequency_hz(note) == pytest.approx(midi_to_hz(midi_note), abs=1.0)


def test_riff_with_guitar_timbre() -> None:
    guitar = riff(["E2", "G2", "A2"], bpm=120, harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES)
    bass = riff(["E2", "G2", "A2"], bpm=120)

    assert guitar.onset_frames == bass.onset_frames
    assert guitar.midi_notes == (40, 43, 45)
    assert not np.array_equal(guitar.samples, bass.samples)
    first_attack = guitar.samples[guitar.onset_frames[0] : guitar.onset_frames[0] + SR // 10]
    assert _dominant_frequency_hz(first_attack) == pytest.approx(2 * midi_to_hz(40), abs=10.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"bpm": 0}, {"bpm": 120, "repeats": 0}, {"bpm": 120, "gate": 0.0}, {"bpm": 120, "gate": 1.1}],
)
def test_riff_rejects_invalid_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        riff(["E1"], **kwargs)  # type: ignore[arg-type]


def test_riff_rejects_zero_length_steps() -> None:
    with pytest.raises(ValueError):
        riff([("E1", 0.0)], bpm=120)


# --- silence and noise ---------------------------------------------------------------------------


def test_silence() -> None:
    samples = silence(0.5)

    assert samples.dtype == np.float32
    assert samples.shape == (24_000,)
    assert not np.any(samples)


def test_noise_has_the_requested_level_and_is_deterministic() -> None:
    floor = noise(1.0)
    loud = noise(1.0, rms_dbfs=-20.0)

    assert floor.dtype == np.float32
    assert floor.shape == (SR,)
    assert measure_level(floor).rms_dbfs == pytest.approx(-65.0, abs=0.01)
    assert measure_level(loud).rms_dbfs == pytest.approx(-20.0, abs=0.01)
    assert np.array_equal(noise(1.0, seed=3), noise(1.0, seed=3))
    assert not np.array_equal(noise(1.0, seed=3), noise(1.0, seed=4))
    assert noise(0.0).shape == (0,)


# --- onset_times ---------------------------------------------------------------------------------


def test_onset_times_follow_the_pattern_on_its_grid() -> None:
    times = onset_times(120, "x..x..x.", bars=2, start_seconds=1.0)

    # 8th notes at 120 BPM are 0.25 s apart; a bar of 8 of them lasts 2 s.
    assert times == pytest.approx([1.0, 1.75, 2.5, 3.0, 3.75, 4.5])
    assert onset_times(100, "xxxx", bars=1, steps_per_beat=1) == pytest.approx([0, 0.6, 1.2, 1.8])


def test_onset_times_jitter_and_misses_are_deterministic() -> None:
    exact = onset_times(92, "x" * 8, bars=10)
    played = onset_times(92, "x" * 8, bars=10, jitter_seconds=0.02, miss_ratio=0.25, seed=5)

    assert np.array_equal(
        played, onset_times(92, "x" * 8, bars=10, jitter_seconds=0.02, miss_ratio=0.25, seed=5)
    )
    assert 0.6 * len(exact) < len(played) < 0.9 * len(exact)
    assert np.all(np.diff(played) > 0)
    nearest = np.abs(played[:, np.newaxis] - exact).min(axis=1)
    assert 0.005 < float(np.std(nearest)) < 0.03


@pytest.mark.parametrize("pattern", ["", "x-x-", "X..."])
def test_onset_times_rejects_invalid_patterns(pattern: str) -> None:
    with pytest.raises(ValueError, match="pattern"):
        onset_times(100, pattern, bars=1)

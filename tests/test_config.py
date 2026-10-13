import pytest
from pydantic import ValidationError

from jammate.config import AnalysisConfig, AudioConfig, JamMateConfig


def test_audio_defaults_match_expected_system_settings() -> None:
    audio = JamMateConfig().audio

    assert audio.sample_rate_hz == 48_000
    assert audio.block_size_frames == 128
    assert audio.fallback_block_size_frames == 256
    assert audio.input_channel_index == 0
    assert audio.monitor_gain == 1.0
    assert audio.input_device_name is None
    assert audio.output_device_name is None


def test_config_is_frozen() -> None:
    audio = AudioConfig()

    with pytest.raises(ValidationError):
        audio.block_size_frames = 256  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AudioConfig.model_validate({"sample_rate": 44_100})


def test_fallback_block_size_must_not_be_smaller() -> None:
    with pytest.raises(ValidationError, match="fallback_block_size_frames"):
        AudioConfig(block_size_frames=512, fallback_block_size_frames=256)


def test_input_ring_capacity_is_derived_from_seconds() -> None:
    assert AudioConfig().input_ring_capacity_frames == 96_000
    assert AudioConfig(input_ring_seconds=0.5).input_ring_capacity_frames == 24_000


def test_input_ring_must_hold_a_fallback_block() -> None:
    with pytest.raises(ValidationError, match="input ring"):
        AudioConfig(input_ring_seconds=0.001)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_rate_hz": 0},
        {"block_size_frames": 0},
        {"input_channel_index": -1},
        {"monitor_gain": -0.1},
        {"monitor_gain": 4.5},
    ],
)
def test_out_of_range_values_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AudioConfig.model_validate(overrides)


def test_analysis_defaults_select_the_bass_range() -> None:
    analysis = JamMateConfig().analysis

    assert analysis.instrument == "bass"
    assert analysis.pitch_range_hz == (35.0, 400.0)
    assert analysis.hop_frames == 480


def test_instrument_selects_the_pitch_range_and_overrides_win() -> None:
    assert AnalysisConfig(instrument="guitar").pitch_range_hz == (60.0, 1400.0)
    assert AnalysisConfig(instrument="guitar", pitch_min_hz=70.0).pitch_range_hz == (70.0, 1400.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"instrument": "ukulele"},
        {"pitch_min_hz": 500.0},  # above the bass maximum
        {"hysteresis_cents": 30.0},  # not above the tuning tolerance
        {"tuning_tolerance_cents": 50.0},
        {"yin_threshold": 1.0},
        {"level_gate_dbfs": 3.0},
    ],
)
def test_invalid_analysis_values_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(overrides)


def test_pitch_range_must_fit_the_sample_rate() -> None:
    with pytest.raises(ValidationError, match="quarter of the sample rate"):
        JamMateConfig(
            audio=AudioConfig(sample_rate_hz=4_000),
            analysis=AnalysisConfig(instrument="guitar"),
        )


def test_analysis_sections_have_defaults() -> None:
    analysis = AnalysisConfig()

    assert analysis.onset.refractory_seconds == 0.08
    assert analysis.intensity.report_step == 0.05
    assert analysis.playing.hold_seconds == 1.0
    assert (analysis.tempo.min_bpm, analysis.tempo.max_bpm) == (50.0, 200.0)
    assert analysis.tempo.preferred_bpm == 100.0


@pytest.mark.parametrize(
    "tempo",
    [
        {"min_bpm": 120.0, "max_bpm": 200.0},  # less than an octave
        {"preferred_bpm": 40.0},
        {"preferred_bpm": 250.0},
        {"min_onsets": 2},
        {"agreement_percent": 0.0},
        {"adaptation": 1.5},
    ],
)
def test_invalid_tempo_settings_are_rejected(tempo: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate({"tempo": tempo})


@pytest.mark.parametrize(
    "intensity",
    [
        {"quiet_dbfs": -6.0, "loud_dbfs": -50.0},
        {"attack_quiet_dbfs": 0.0, "attack_loud_dbfs": -40.0},
        {"level_weight": 0.0, "attack_weight": 0.0, "density_weight": 0.0},
    ],
)
def test_invalid_intensity_mappings_are_rejected(intensity: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate({"intensity": intensity})


def test_musical_state_section_keeps_a_bounded_note_history() -> None:
    assert JamMateConfig().state.recent_note_count == 8
    for count in (0, 65):
        with pytest.raises(ValidationError):
            JamMateConfig.model_validate({"state": {"recent_note_count": count}})

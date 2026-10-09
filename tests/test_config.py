import pytest
from pydantic import ValidationError

from jammate.config import AudioConfig, JamMateConfig


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

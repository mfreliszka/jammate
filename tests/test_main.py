import pytest

from jammate.main import build_config, build_parser, main


def test_help_lists_all_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--headless", "--list-devices", "--input", "--output", "--block-size"):
        assert option in help_text


def test_defaults_produce_default_config() -> None:
    config = build_config(build_parser().parse_args([]))

    assert config.audio.block_size_frames == 128
    assert config.audio.input_device_name is None


def test_device_names_are_passed_to_config() -> None:
    args = build_parser().parse_args(["--input", "Scarlett", "--output", "MacBook"])

    config = build_config(args)

    assert config.audio.input_device_name == "Scarlett"
    assert config.audio.output_device_name == "MacBook"


@pytest.mark.parametrize(
    ("block_size", "expected_fallback"),
    [(64, 256), (256, 256), (512, 512)],
)
def test_block_size_override_never_falls_back_to_smaller(
    block_size: int, expected_fallback: int
) -> None:
    args = build_parser().parse_args(["--block-size", str(block_size)])

    audio = build_config(args).audio

    assert audio.block_size_frames == block_size
    assert audio.fallback_block_size_frames == expected_fallback


def test_invalid_block_size_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--block-size", "0"])

    assert exit_info.value.code == 2

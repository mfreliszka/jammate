import pytest

from jammate import main as main_module
from jammate.audio.devices import DeviceInventory
from jammate.main import build_config, build_parser, main
from tests.fakes import make_device


@pytest.fixture
def fake_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = DeviceInventory(
        devices=(
            make_device(0, "MacBook Pro Microphone", inputs=1, outputs=0),
            make_device(1, "MacBook Pro Speakers", inputs=0, outputs=2),
            make_device(2, "Scarlett 2i2 USB", inputs=2, outputs=2),
        ),
        default_input_index=0,
        default_output_index=1,
    )
    monkeypatch.setattr(main_module, "query_inventory", lambda: inventory)
    monkeypatch.setattr(main_module, "supports_settings", lambda *args, **kwargs: True)


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


@pytest.mark.usefixtures("fake_devices")
def test_list_devices_prints_table_and_default_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--list-devices"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Scarlett 2i2 USB" in out
    assert "Selected input: MacBook Pro Microphone" in out
    assert "Selected output: MacBook Pro Speakers" in out


@pytest.mark.usefixtures("fake_devices")
def test_list_devices_resolves_requested_names(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--list-devices", "--input", "scarlett", "--output", "scarlett"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Selected input: Scarlett 2i2 USB" in out
    assert "Selected output: Scarlett 2i2 USB" in out


@pytest.mark.usefixtures("fake_devices")
def test_list_devices_reports_unknown_device(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--list-devices", "--input", "Behringer"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Selected input: error: No input device matches 'Behringer'" in captured.err
    assert "Selected output: MacBook Pro Speakers" in captured.out


def test_input_channel_is_one_based_on_the_command_line() -> None:
    args = build_parser().parse_args(["--input-channel", "2"])

    assert build_config(args).audio.input_channel_index == 1


def test_input_channel_zero_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--input-channel", "0"])

    assert exit_info.value.code == 2


def test_monitor_gain_override() -> None:
    args = build_parser().parse_args(["--monitor-gain", "0"])

    assert build_config(args).audio.monitor_gain == 0.0


def test_monitor_gain_out_of_range_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--monitor-gain", "5"])

    assert exit_info.value.code == 2


def test_headless_runs_the_headless_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr(main_module, "run_headless", lambda config: received.append(config) or 0)

    exit_code = main(["--headless", "--input", "usb", "--output", "usb", "--input-channel", "2"])

    assert exit_code == 0
    assert received[0].audio.input_device_name == "usb"
    assert received[0].audio.input_channel_index == 1

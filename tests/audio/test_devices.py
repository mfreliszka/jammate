from typing import Any

import pytest
import sounddevice as sd

from jammate.audio import devices
from jammate.audio.devices import (
    AmbiguousDeviceError,
    AudioDevice,
    DeviceInventory,
    DeviceKind,
    DeviceNotFoundError,
    format_device_list,
    inventory_from_portaudio,
    resolve_device,
    supports_settings,
)

HOST_APIS = [{"name": "Core Audio"}]


def _device_info(
    index: int, name: str, inputs: int, outputs: int, sample_rate_hz: float = 48_000.0
) -> dict[str, Any]:
    """Mimic one entry of `sounddevice.query_devices()`."""
    return {
        "name": name,
        "index": index,
        "hostapi": 0,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_low_input_latency": 0.005,
        "default_low_output_latency": 0.004,
        "default_high_input_latency": 0.1,
        "default_high_output_latency": 0.1,
        "default_samplerate": sample_rate_hz,
    }


DEVICE_INFOS = [
    _device_info(0, "MacBook Pro Microphone", inputs=1, outputs=0),
    _device_info(1, "MacBook Pro Speakers", inputs=0, outputs=2),
    _device_info(2, "Scarlett 2i2 USB", inputs=2, outputs=2),
    _device_info(3, "Scarlett 2i2 USB Loopback", inputs=2, outputs=0),
    _device_info(4, "Studio Display Speakers", inputs=0, outputs=2, sample_rate_hz=44_100.0),
]


@pytest.fixture
def inventory() -> DeviceInventory:
    return inventory_from_portaudio(DEVICE_INFOS, HOST_APIS, default_indices=(0, 1))


def test_inventory_maps_portaudio_fields(inventory: DeviceInventory) -> None:
    scarlett = inventory.devices[2]

    assert scarlett == AudioDevice(
        index=2,
        name="Scarlett 2i2 USB",
        host_api="Core Audio",
        max_input_channels=2,
        max_output_channels=2,
        default_sample_rate_hz=48_000.0,
        default_low_input_latency_seconds=0.005,
        default_low_output_latency_seconds=0.004,
    )


def test_inventory_splits_inputs_and_outputs(inventory: DeviceInventory) -> None:
    assert [device.index for device in inventory.inputs] == [0, 2, 3]
    assert [device.index for device in inventory.outputs] == [1, 2, 4]


def test_negative_default_index_means_no_default() -> None:
    inventory = inventory_from_portaudio(DEVICE_INFOS, HOST_APIS, default_indices=(-1, -1))

    assert inventory.default_input_index is None
    assert inventory.default("input") is None
    assert inventory.default("output") is None


def test_default_pointing_at_wrong_kind_is_ignored() -> None:
    inventory = inventory_from_portaudio(DEVICE_INFOS, HOST_APIS, default_indices=(1, 0))

    assert inventory.default("input") is None
    assert inventory.default("output") is None


def test_no_query_selects_system_default(inventory: DeviceInventory) -> None:
    assert resolve_device(inventory, "input", None).name == "MacBook Pro Microphone"
    assert resolve_device(inventory, "output", None).name == "MacBook Pro Speakers"


def test_no_query_without_default_raises() -> None:
    inventory = inventory_from_portaudio(DEVICE_INFOS, HOST_APIS, default_indices=(-1, -1))

    with pytest.raises(DeviceNotFoundError, match="No default input device"):
        resolve_device(inventory, "input", None)


def test_unique_substring_matches_case_insensitively(inventory: DeviceInventory) -> None:
    assert resolve_device(inventory, "output", "scarlett").index == 2
    assert resolve_device(inventory, "input", "  LOOPBACK ").index == 3


def test_exact_name_beats_longer_substring_matches(inventory: DeviceInventory) -> None:
    assert resolve_device(inventory, "input", "scarlett 2i2 usb").index == 2


def test_ambiguous_substring_lists_all_matches(inventory: DeviceInventory) -> None:
    with pytest.raises(AmbiguousDeviceError) as error_info:
        resolve_device(inventory, "input", "Scarlett")

    message = str(error_info.value)
    assert "'Scarlett 2i2 USB'" in message
    assert "'Scarlett 2i2 USB Loopback'" in message


def test_device_of_wrong_kind_is_not_selected(inventory: DeviceInventory) -> None:
    with pytest.raises(DeviceNotFoundError, match="Available: 'MacBook Pro Microphone'"):
        resolve_device(inventory, "input", "Speakers")


def test_numeric_query_is_not_treated_as_index(inventory: DeviceInventory) -> None:
    with pytest.raises(DeviceNotFoundError):
        resolve_device(inventory, "input", "0")


def test_format_lists_devices_and_marks_defaults(inventory: DeviceInventory) -> None:
    def is_rate_supported(device: AudioDevice, kind: DeviceKind) -> bool:
        return device.default_sample_rate_hz == 48_000.0

    text = format_device_list(inventory, 48_000, is_rate_supported)
    lines = text.splitlines()

    assert lines[0] == "Input devices:"
    assert lines[1].startswith("  * MacBook Pro Microphone")
    assert "Output devices:" in lines
    studio_line = next(line for line in lines if "Studio Display Speakers" in line)
    assert "48000 Hz: NO" in studio_line
    scarlett_input_line = next(line for line in lines if "Scarlett 2i2 USB  " in line)
    assert " 2 ch" in scarlett_input_line
    assert "48000 Hz: yes" in scarlett_input_line


def test_format_handles_missing_device_kind() -> None:
    inventory = inventory_from_portaudio(DEVICE_INFOS[:1], HOST_APIS, default_indices=(0, -1))

    text = format_device_list(inventory, 48_000, lambda device, kind: True)

    assert "Output devices:\n    (none)" in text


@pytest.mark.parametrize(
    ("kind", "checker_name"),
    [("input", "check_input_settings"), ("output", "check_output_settings")],
)
def test_supports_settings_queries_portaudio(
    monkeypatch: pytest.MonkeyPatch,
    inventory: DeviceInventory,
    kind: DeviceKind,
    checker_name: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(devices.sd, checker_name, lambda **kwargs: calls.append(kwargs))

    assert supports_settings(inventory.devices[2], kind, 48_000, channels=2)
    assert calls == [{"device": 2, "channels": 2, "samplerate": 48_000, "dtype": "float32"}]


def test_supports_settings_returns_false_when_portaudio_rejects(
    monkeypatch: pytest.MonkeyPatch, inventory: DeviceInventory
) -> None:
    def reject(**kwargs: Any) -> None:
        raise sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(devices.sd, "check_input_settings", reject)

    assert not supports_settings(inventory.devices[0], "input", 48_000)

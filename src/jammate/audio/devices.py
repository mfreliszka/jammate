"""Audio device discovery and selection.

Everything here queries PortAudio or formats the result. It runs on the main/GUI thread before a
stream opens, never from the audio callback. Devices are selected by name or a unique part of it,
never by a hard-coded index, because PortAudio indices change when devices are plugged in.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import sounddevice as sd

type DeviceKind = Literal["input", "output"]


class DeviceResolutionError(LookupError):
    """A requested device could not be selected."""


class DeviceNotFoundError(DeviceResolutionError):
    """No device of the requested kind matches the query."""


class AmbiguousDeviceError(DeviceResolutionError):
    """Several devices of the requested kind match the query."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate_hz: float
    default_low_input_latency_seconds: float
    default_low_output_latency_seconds: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def max_channels(self, kind: DeviceKind) -> int:
        return self.max_input_channels if kind == "input" else self.max_output_channels

    def default_low_latency_seconds(self, kind: DeviceKind) -> float:
        if kind == "input":
            return self.default_low_input_latency_seconds
        return self.default_low_output_latency_seconds


@dataclass(frozen=True)
class DeviceInventory:
    """Snapshot of the devices PortAudio reports, plus the system default indices."""

    devices: tuple[AudioDevice, ...]
    default_input_index: int | None
    default_output_index: int | None

    @property
    def inputs(self) -> tuple[AudioDevice, ...]:
        return tuple(device for device in self.devices if device.is_input)

    @property
    def outputs(self) -> tuple[AudioDevice, ...]:
        return tuple(device for device in self.devices if device.is_output)

    def candidates(self, kind: DeviceKind) -> tuple[AudioDevice, ...]:
        return self.inputs if kind == "input" else self.outputs

    def default(self, kind: DeviceKind) -> AudioDevice | None:
        default_index = self.default_input_index if kind == "input" else self.default_output_index
        for device in self.candidates(kind):
            if device.index == default_index:
                return device
        return None


def inventory_from_portaudio(
    device_infos: Sequence[Mapping[str, Any]],
    host_api_infos: Sequence[Mapping[str, Any]],
    default_indices: tuple[int, int],
) -> DeviceInventory:
    """Convert raw `sounddevice` query results into a `DeviceInventory`.

    PortAudio reports a missing default device as a negative index.
    """
    devices = tuple(
        AudioDevice(
            index=int(info["index"]),
            name=str(info["name"]),
            host_api=str(host_api_infos[int(info["hostapi"])]["name"]),
            max_input_channels=int(info["max_input_channels"]),
            max_output_channels=int(info["max_output_channels"]),
            default_sample_rate_hz=float(info["default_samplerate"]),
            default_low_input_latency_seconds=float(info["default_low_input_latency"]),
            default_low_output_latency_seconds=float(info["default_low_output_latency"]),
        )
        for info in device_infos
    )
    default_input_index, default_output_index = default_indices
    return DeviceInventory(
        devices=devices,
        default_input_index=default_input_index if default_input_index >= 0 else None,
        default_output_index=default_output_index if default_output_index >= 0 else None,
    )


def query_inventory() -> DeviceInventory:
    """Query PortAudio for the current devices.

    PortAudio enumerates devices when it initializes, so a device plugged in after start-up
    appears only after PortAudio is re-initialized.
    """
    default_input_index, default_output_index = sd.default.device
    return inventory_from_portaudio(
        device_infos=list(sd.query_devices()),
        host_api_infos=list(sd.query_hostapis()),
        default_indices=(int(default_input_index), int(default_output_index)),
    )


def refresh_inventory() -> DeviceInventory:
    """Re-initialize PortAudio and return a fresh `DeviceInventory` snapshot.

    Must only be called while no audio stream is active. Re-initializing PortAudio allows newly
    plugged-in or unplugged USB audio interfaces to appear in both input and output lists without
    restarting JamMate.
    """
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if callable(terminate) and callable(initialize):
        terminate()
        initialize()
    return query_inventory()


def resolve_device(inventory: DeviceInventory, kind: DeviceKind, query: str | None) -> AudioDevice:
    """Select a device of `kind` by name.

    `None` selects the system default. Otherwise a case-insensitive exact name match wins, then a
    unique case-insensitive substring match. A numeric query is treated as text, not an index.
    """
    candidates = inventory.candidates(kind)

    if query is None:
        default = inventory.default(kind)
        if default is None:
            raise DeviceNotFoundError(f"No default {kind} device. Available: {_names(candidates)}.")
        return default

    needle = query.strip().casefold()
    exact = [device for device in candidates if device.name.casefold() == needle]
    matches = exact or [device for device in candidates if needle in device.name.casefold()]

    if not matches:
        raise DeviceNotFoundError(
            f"No {kind} device matches {query!r}. Available: {_names(candidates)}."
        )
    if len(matches) > 1:
        raise AmbiguousDeviceError(
            f"{query!r} matches several {kind} devices: {_names(matches)}. "
            "Use a longer part of the name."
        )
    return matches[0]


def supports_settings(
    device: AudioDevice, kind: DeviceKind, sample_rate_hz: int, channels: int = 1
) -> bool:
    """Ask PortAudio whether `device` accepts float32 audio at `sample_rate_hz`."""
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(device=device.index, channels=channels, samplerate=sample_rate_hz, dtype="float32")
    except (sd.PortAudioError, ValueError):
        return False
    return True


def format_device_list(
    inventory: DeviceInventory,
    sample_rate_hz: int,
    is_rate_supported: Callable[[AudioDevice, DeviceKind], bool],
) -> str:
    """Render the inventory as a human-readable table for the terminal."""
    lines: list[str] = []
    sections: tuple[tuple[str, DeviceKind], ...] = (
        ("Input devices", "input"),
        ("Output devices", "output"),
    )
    for title, kind in sections:
        devices = inventory.candidates(kind)
        default = inventory.default(kind)
        lines.append(f"{title}:")
        if not devices:
            lines.append("    (none)")
            continue
        name_width = max(len(device.name) for device in devices)
        for device in devices:
            marker = "*" if device == default else " "
            latency_ms = device.default_low_latency_seconds(kind) * 1000
            supported = "yes" if is_rate_supported(device, kind) else "NO"
            lines.append(
                f"  {marker} {device.name:<{name_width}}  {device.max_channels(kind):>2} ch"
                f"  default {device.default_sample_rate_hz:>6.0f} Hz"
                f"  low latency {latency_ms:5.1f} ms"
                f"  {sample_rate_hz} Hz: {supported}"
            )
    lines.append("")
    lines.append("* = system default. Select with --input/--output NAME (or a unique part of it).")
    return "\n".join(lines)


def _names(devices: Sequence[AudioDevice]) -> str:
    return ", ".join(repr(device.name) for device in devices) or "none"

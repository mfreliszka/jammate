"""Shared test doubles for audio hardware (no PortAudio stream is ever opened in tests)."""

from typing import Any, ClassVar

import sounddevice as sd

from jammate.audio.devices import AudioDevice


def make_device(index: int, name: str, inputs: int, outputs: int) -> AudioDevice:
    return AudioDevice(
        index=index,
        name=name,
        host_api="Core Audio",
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_sample_rate_hz=48_000.0,
        default_low_input_latency_seconds=0.005,
        default_low_output_latency_seconds=0.004,
    )


# The developer's interface appears as two same-named devices: input-only and output-only.
INPUT_DEVICE = make_device(1, "USB Audio CODEC", inputs=2, outputs=0)
OUTPUT_DEVICE = make_device(0, "USB Audio CODEC", inputs=0, outputs=2)


class FakeStream:
    """Stands in for `sd.Stream`; records how it was opened and its lifecycle."""

    instances: ClassVar[list["FakeStream"]] = []
    reject_block_sizes: ClassVar[frozenset[int]] = frozenset()
    actual_sample_rate_hz: ClassVar[float | None] = None

    def __init__(self, **kwargs: Any) -> None:
        if kwargs["blocksize"] in self.reject_block_sizes:
            raise sd.PortAudioError("Invalid buffer size")
        self.kwargs = kwargs
        self.samplerate = self.actual_sample_rate_hz or kwargs["samplerate"]
        self.blocksize = kwargs["blocksize"]
        self.latency = (0.016, 0.007)
        self.active = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.reject_block_sizes = frozenset()
        cls.actual_sample_rate_hz = None

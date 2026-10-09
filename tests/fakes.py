"""Shared test doubles: audio hardware (no PortAudio stream is ever opened in tests) and a clock."""

from typing import Any, ClassVar

import sounddevice as sd

from jammate.audio.devices import AudioDevice
from jammate.music.events import SampleFrame, frames_to_seconds, seconds_to_frames


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


class FakeClock:
    """Deterministic clock for tests (§66): advance it explicitly instead of sleeping.

    Time is kept as a whole number of sample frames, so `frame_index` and `seconds` always agree;
    the resolution is one frame (about 21 µs at 48 kHz). Call the instance where code expects
    `time.monotonic`, and pass `clock.advance` where it expects `time.sleep`.
    """

    def __init__(self, sample_rate_hz: int = 48_000, start_frame: SampleFrame = 0) -> None:
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        self.sample_rate_hz = sample_rate_hz
        self._frame_index = start_frame

    @property
    def frame_index(self) -> SampleFrame:
        return self._frame_index

    @property
    def seconds(self) -> float:
        return frames_to_seconds(self._frame_index, self.sample_rate_hz)

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        """Move forward by `seconds`, rounded to the nearest frame."""
        self.advance_frames(seconds_to_frames(seconds, self.sample_rate_hz))

    def advance_frames(self, frames: int) -> None:
        if frames < 0:
            raise ValueError("a monotonic clock never goes backwards")
        self._frame_index += frames

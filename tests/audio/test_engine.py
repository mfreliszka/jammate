import numpy as np
import pytest

from jammate.audio import engine as engine_module
from jammate.audio.devices import DeviceInventory, DeviceNotFoundError
from jammate.audio.engine import AudioEngine, AudioEngineError, BlockProcessor
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AudioConfig
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, FakeStream, make_device

BLOCK = 128


def _two_channel_block(frames: int = BLOCK) -> np.ndarray:
    left = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    return np.stack([left, -left * 0.25], axis=1)


# --- BlockProcessor -----------------------------------------------------------------------------


def test_monitor_copies_selected_channel_to_both_outputs_with_gain() -> None:
    processor = BlockProcessor(RingBuffer(1024), input_channel_index=1, monitor_gain=2.0)
    indata = _two_channel_block()
    outdata = np.full((BLOCK, 2), 9.0, dtype=np.float32)

    processor.process(indata, outdata)

    expected = indata[:, 1] * 2.0
    np.testing.assert_allclose(outdata[:, 0], expected)
    np.testing.assert_allclose(outdata[:, 1], expected)


def test_selected_channel_is_written_to_input_ring() -> None:
    ring = RingBuffer(1024)
    processor = BlockProcessor(ring, input_channel_index=0, monitor_gain=1.0)
    indata = _two_channel_block()

    processor.process(indata, np.zeros((BLOCK, 2), dtype=np.float32))
    out = np.zeros((BLOCK, 1), dtype=np.float32)
    ring.read(out)

    np.testing.assert_array_equal(out[:, 0], indata[:, 0])


def test_zero_gain_outputs_silence_but_still_feeds_ring() -> None:
    ring = RingBuffer(1024)
    processor = BlockProcessor(ring, input_channel_index=0, monitor_gain=0.0)
    outdata = np.full((BLOCK, 2), 1.0, dtype=np.float32)

    processor.process(_two_channel_block(), outdata)

    assert not outdata.any()
    assert ring.frames_written == BLOCK


def test_counters_track_frames_blocks_and_xruns() -> None:
    processor = BlockProcessor(RingBuffer(1024), input_channel_index=0, monitor_gain=1.0)
    outdata = np.zeros((BLOCK, 2), dtype=np.float32)

    processor.process(_two_channel_block(), outdata)
    processor.process(_two_channel_block(), outdata, input_overflow=True)
    processor.process(_two_channel_block(), outdata, output_underflow=True)

    assert processor.frames_processed == 3 * BLOCK
    assert processor.blocks_processed == 3
    assert processor.input_overflows == 1
    assert processor.output_underflows == 1


def test_xruns_during_settling_are_counted_separately() -> None:
    processor = BlockProcessor(RingBuffer(1024), input_channel_index=0, monitor_gain=1.0)
    outdata = np.zeros((BLOCK, 2), dtype=np.float32)
    processor.begin_settling(2)

    processor.process(_two_channel_block(), outdata, input_overflow=True)
    processor.process(_two_channel_block(), outdata, output_underflow=True)
    processor.process(_two_channel_block(), outdata, input_overflow=True)

    assert processor.startup_xruns == 2
    assert processor.input_overflows == 1
    assert processor.output_underflows == 0


def test_exception_in_block_outputs_silence_and_is_counted() -> None:
    class BrokenRing(RingBuffer):
        def write(self, frames: np.ndarray) -> None:
            raise RuntimeError("boom")

    processor = BlockProcessor(BrokenRing(1024), input_channel_index=0, monitor_gain=1.0)
    outdata = np.full((BLOCK, 2), 1.0, dtype=np.float32)

    processor.process(_two_channel_block(), outdata)

    assert not outdata.any()
    assert processor.errors == 1
    assert isinstance(processor.last_error, RuntimeError)
    assert processor.frames_processed == BLOCK


def test_timing_covers_most_recent_window() -> None:
    processor = BlockProcessor(
        RingBuffer(1024), input_channel_index=0, monitor_gain=1.0, timing_window_blocks=4
    )
    assert processor.timing().blocks_measured == 0

    outdata = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(6):
        processor.process(_two_channel_block(), outdata)
    timing = processor.timing()

    assert timing.blocks_measured == 4
    assert 0.0 < timing.mean_seconds <= timing.max_seconds


# --- AudioEngine (fake PortAudio stream) ---------------------------------------------------------


def test_engine_rejects_input_channel_the_device_lacks() -> None:
    with pytest.raises(ValueError, match="Input channel 3 requested"):
        AudioEngine(AudioConfig(input_channel_index=2), INPUT_DEVICE, OUTPUT_DEVICE)


def test_engine_rejects_output_device_without_outputs() -> None:
    with pytest.raises(ValueError, match="no output channels"):
        AudioEngine(AudioConfig(), INPUT_DEVICE, INPUT_DEVICE)


def test_start_opens_duplex_stream_with_configured_settings(fake_stream: type[FakeStream]) -> None:
    engine = AudioEngine(AudioConfig(input_channel_index=1), INPUT_DEVICE, OUTPUT_DEVICE)

    info = engine.start()

    stream = fake_stream.instances[0]
    assert stream.kwargs["device"] == (1, 0)
    assert stream.kwargs["channels"] == (2, 2)
    assert stream.kwargs["samplerate"] == 48_000
    assert stream.kwargs["blocksize"] == 128
    assert stream.kwargs["dtype"] == "float32"
    assert engine.is_running
    assert info.block_size_frames == 128
    assert info.input_latency_seconds == pytest.approx(0.016)
    assert info.block_period_seconds == pytest.approx(128 / 48_000)


def test_start_falls_back_to_larger_block_size(fake_stream: type[FakeStream]) -> None:
    fake_stream.reject_block_sizes = frozenset({128})
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)

    assert engine.start().block_size_frames == 256


def test_start_fails_when_no_block_size_works(fake_stream: type[FakeStream]) -> None:
    fake_stream.reject_block_sizes = frozenset({128, 256})
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)

    with pytest.raises(AudioEngineError, match=r"block sizes \[128, 256\]"):
        engine.start()


def test_start_refuses_a_different_sample_rate(fake_stream: type[FakeStream]) -> None:
    fake_stream.actual_sample_rate_hz = 44_100.0
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)

    with pytest.raises(AudioEngineError, match="44100 Hz"):
        engine.start()
    assert fake_stream.instances[0].closed


def test_stop_close_and_restart_lifecycle(fake_stream: type[FakeStream]) -> None:
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)
    engine.start()

    engine.stop()
    assert not engine.is_running
    engine.start()
    assert len(fake_stream.instances) == 1  # resumed, not reopened

    engine.close()
    engine.close()  # idempotent
    assert fake_stream.instances[0].closed
    assert not engine.is_running


def test_context_manager_closes_stream(fake_stream: type[FakeStream]) -> None:
    with AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE) as engine:
        engine.start()

    assert fake_stream.instances[0].closed


def test_set_monitor_gain_validates_range() -> None:
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)

    engine.set_monitor_gain(0.5)
    assert engine.processor.monitor_gain == 0.5
    with pytest.raises(ValueError):
        engine.set_monitor_gain(5.0)


class OverflowStatus:
    """Mimics `sd.CallbackFlags` with only `input_overflow` set."""

    input_overflow = True
    output_underflow = False

    def __bool__(self) -> bool:
        return True


def test_callback_forwards_status_flags(fake_stream: type[FakeStream]) -> None:
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)
    outdata = np.zeros((BLOCK, 2), dtype=np.float32)

    engine._callback(_two_channel_block()[:, :1], outdata, BLOCK, None, OverflowStatus())

    stats = engine.stats()
    assert stats.input_overflows == 1
    assert stats.output_underflows == 0
    assert stats.frames_processed == BLOCK


def test_each_start_opens_a_settling_window(fake_stream: type[FakeStream]) -> None:
    engine = AudioEngine(AudioConfig(), INPUT_DEVICE, OUTPUT_DEVICE)
    indata = _two_channel_block()[:, :1]
    outdata = np.zeros((BLOCK, 2), dtype=np.float32)
    settle_blocks = 19  # ceil(50 ms / (128 / 48 kHz))

    engine.start()
    engine._callback(indata, outdata, BLOCK, None, OverflowStatus())
    for _ in range(settle_blocks - 1):
        engine._callback(indata, outdata, BLOCK, None, None)
    engine._callback(indata, outdata, BLOCK, None, OverflowStatus())
    engine.stop()
    engine.start()
    engine._callback(indata, outdata, BLOCK, None, OverflowStatus())

    stats = engine.stats()
    assert stats.startup_xruns == 2
    assert stats.input_overflows == 1


def test_from_config_resolves_devices_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = DeviceInventory(
        devices=(make_device(5, "MacBook Pro Microphone", 1, 0), OUTPUT_DEVICE, INPUT_DEVICE),
        default_input_index=5,
        default_output_index=0,
    )
    monkeypatch.setattr(engine_module, "query_inventory", lambda: inventory)

    engine = AudioEngine.from_config(AudioConfig(input_device_name="usb"))

    assert engine._input_device == INPUT_DEVICE
    assert engine._output_device == OUTPUT_DEVICE


def test_from_config_propagates_resolution_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = DeviceInventory((OUTPUT_DEVICE, INPUT_DEVICE), None, None)
    monkeypatch.setattr(engine_module, "query_inventory", lambda: inventory)

    with pytest.raises(DeviceNotFoundError):
        AudioEngine.from_config(AudioConfig(input_device_name="Behringer"))

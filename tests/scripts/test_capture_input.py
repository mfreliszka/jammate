import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from jammate.audio.ring_buffer import RingBuffer

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capture_input.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("capture_input", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["capture_input"] = module
    spec.loader.exec_module(module)
    return module


capture_input = _load_script()


class ListSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def write(self, frames: np.ndarray) -> None:
        self.chunks.append(frames.copy())


def test_drain_moves_every_unread_frame_in_order() -> None:
    ring = RingBuffer(8)
    buffer = np.zeros((3, 1), dtype=np.float32)  # smaller than the backlog: needs several reads
    for start in range(0, 6, 2):
        ring.write(np.arange(start, start + 2, dtype=np.float32).reshape(-1, 1))
    sink = ListSink()

    moved = capture_input.drain(ring, buffer, sink)

    assert moved == 6
    np.testing.assert_array_equal(np.concatenate(sink.chunks)[:, 0], np.arange(6))
    assert capture_input.drain(ring, buffer, sink) == 0


def test_level_tracking_sink_remembers_peak() -> None:
    inner = ListSink()
    sink = capture_input.LevelTrackingSink(inner)

    sink.write(np.array([[0.1], [-0.7]], dtype=np.float32))
    sink.write(np.array([[0.3]], dtype=np.float32))

    assert sink.peak == pytest.approx(0.7)
    assert len(inner.chunks) == 2


def test_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "riff.wav"
    existing.write_bytes(b"keep me")

    assert capture_input.main([str(existing)]) == 1
    assert existing.read_bytes() == b"keep me"

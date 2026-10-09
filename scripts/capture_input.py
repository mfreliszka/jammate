"""Record the raw bass input to a WAV file. Developer tool for analysis test fixtures.

The bass is monitored while recording, like in the headless runner. Recording uses the same
engine and input ring buffer as the app. The main thread drains the ring to disk, so the audio
callback never touches the file.

Example:
    uv run python scripts/capture_input.py tests/data/riff_e_92bpm.wav --input usb --output usb \\
        --seconds 20
"""

import argparse
import logging
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Protocol

import numpy as np
import soundfile as sf

from jammate.audio.devices import DeviceResolutionError
from jammate.audio.engine import AudioEngine, AudioEngineError
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AudioConfig

logger = logging.getLogger("capture_input")

POLL_SECONDS = 0.1


class FrameSink(Protocol):
    def write(self, frames: np.ndarray) -> None: ...


class LevelTrackingSink:
    """Forwards frames to a sound file and remembers the absolute peak."""

    def __init__(self, sink: FrameSink) -> None:
        self._sink = sink
        self.peak = 0.0

    def write(self, frames: np.ndarray) -> None:
        if frames.size:
            self.peak = max(self.peak, float(np.max(np.abs(frames))))
        self._sink.write(frames)


def drain(ring: RingBuffer, buffer: np.ndarray, sink: FrameSink) -> int:
    """Move every unread frame from `ring` to `sink`. Returns the number of frames moved."""
    moved = 0
    while True:
        result = ring.read(buffer)
        if result.frame_count == 0:
            return moved
        sink.write(buffer[: result.frame_count])
        moved += result.frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record raw instrument input to a WAV file.")
    parser.add_argument("path", type=Path, help="WAV file to create")
    parser.add_argument("--input", metavar="NAME", help="input device name or unique part")
    parser.add_argument("--output", metavar="NAME", help="output device name or unique part")
    parser.add_argument(
        "--input-channel", type=int, default=1, metavar="N", help="1 = first input (default: 1)"
    )
    parser.add_argument(
        "--monitor-gain", type=float, default=1.0, metavar="GAIN", help="0 to 4 (default: 1.0)"
    )
    parser.add_argument(
        "--seconds", type=float, metavar="S", help="stop after S seconds (default: Ctrl+C)"
    )
    parser.add_argument("--subtype", default="PCM_24", help="soundfile subtype (default: PCM_24)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing file")
    return parser


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    path: Path = args.path
    if path.exists() and not args.overwrite:
        logger.error("%s already exists (use --overwrite to replace it).", path)
        return 1
    if args.input_channel < 1:
        logger.error("--input-channel must be 1 or more.")
        return 1

    audio = AudioConfig(
        input_device_name=args.input,
        output_device_name=args.output,
        input_channel_index=args.input_channel - 1,
        monitor_gain=args.monitor_gain,
    )
    try:
        engine = AudioEngine.from_config(audio)
    except (DeviceResolutionError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    ring = engine.input_ring
    buffer = np.zeros((ring.capacity_frames, ring.channels), dtype=np.float32)
    frames_recorded = 0
    failed = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        with sf.SoundFile(
            path, "w", samplerate=audio.sample_rate_hz, channels=1, subtype=args.subtype
        ) as wav:
            sink = LevelTrackingSink(wav)
            try:
                engine.start()
                deadline = None if args.seconds is None else time.monotonic() + args.seconds
                print(f"Recording to {path}. Press Ctrl+C to stop.", flush=True)
                while deadline is None or time.monotonic() < deadline:
                    time.sleep(POLL_SECONDS)
                    frames_recorded += drain(ring, buffer, sink)
            except KeyboardInterrupt:
                pass
            except AudioEngineError as exc:
                logger.error("%s", exc)
                failed = True
            finally:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                engine.close()
                frames_recorded += drain(ring, buffer, sink)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    if failed:
        path.unlink(missing_ok=True)
        return 1

    seconds = frames_recorded / audio.sample_rate_hz
    peak_dbfs = 20 * np.log10(sink.peak) if sink.peak > 0 else float("-inf")
    logger.info("Recorded %.1f s to %s, peak %.1f dBFS.", seconds, path, peak_dbfs)
    if ring.dropped_frames:
        logger.warning(
            "%d frames were lost (%d ring overruns): the file has gaps.",
            ring.dropped_frames,
            ring.overrun_count,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

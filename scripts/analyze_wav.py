"""Run the analysis pipeline over a WAV file and print the event timeline (offline tuning, §67).

The file goes through the same path as live input: 128-frame blocks into a ring buffer, read by
an `AnalysisWorker` (without its thread). Each event shows when the note began and when the
worker decided, so the difference is the decision latency.

Examples:
    uv run python scripts/analyze_wav.py tests/data/guitar_open_strings.wav --instrument guitar
    uv run python scripts/analyze_wav.py riff.wav --instrument guitar --hops --level-gate -50
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import ValidationError

from jammate.analysis.notes import cents_offset, hz_to_midi, midi_to_name
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisStats, EmittedEvent, analyze_offline
from jammate.config import AnalysisConfig, AudioConfig
from jammate.music.events import PitchEvent, SilenceEvent, frames_to_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the analysis event timeline of a WAV.")
    parser.add_argument("path", type=Path, help="mono or multichannel WAV (channel 1 is used)")
    parser.add_argument("--instrument", choices=["bass", "guitar"], default="bass")
    parser.add_argument("--channel", type=int, default=1, help="channel to analyze (default: 1)")
    parser.add_argument("--block-size", type=int, default=128, help="frames per block")
    parser.add_argument("--hops", action="store_true", help="also print every hop's estimate")
    parser.add_argument("--level-gate", type=float, metavar="DBFS", help="override the gate")
    parser.add_argument("--yin-threshold", type=float, help="override the YIN threshold")
    return parser


def format_event(emitted: EmittedEvent, sample_rate_hz: int) -> str:
    event = emitted.event
    decided_seconds = frames_to_seconds(emitted.emitted_frame, sample_rate_hz)
    latency_ms = 1000 * frames_to_seconds(emitted.emitted_frame - event.frame_index, sample_rate_hz)
    when = f"{event.timestamp_seconds:8.3f} s"
    decided = f"(decided {decided_seconds:7.3f} s, +{latency_ms:3.0f} ms)"
    if isinstance(event, PitchEvent):
        cents = cents_offset(event.frequency_hz, event.midi_note)
        return (
            f"{when}  {midi_to_name(event.midi_note):4s} {event.frequency_hz:7.2f} Hz "
            f"{cents:+4.0f} c  conf {event.confidence:.2f}  {decided}"
        )
    if isinstance(event, SilenceEvent):
        return f"{when}  --   silence                          {decided}"
    return f"{when}  {event!r}"


def format_hop(frame: PitchFrame, sample_rate_hz: int) -> str:
    when = f"{frames_to_seconds(frame.frame_index, sample_rate_hz):8.3f} s"
    level = f"rms {frame.rms_dbfs:6.1f} dBFS"
    if frame.frequency_hz is None:
        return f"{when}    hop  {level}  unvoiced  conf {frame.confidence:.2f}"
    name = midi_to_name(hz_to_midi(frame.frequency_hz))
    return (
        f"{when}    hop  {level}  {frame.frequency_hz:7.2f} Hz {name:4s}"
        f"  conf {frame.confidence:.2f}"
    )


def format_summary(
    events: list[EmittedEvent], stats: AnalysisStats, duration_seconds: float, wall_seconds: float
) -> str:
    notes = [e.event for e in events if isinstance(e.event, PitchEvent)]
    counts = Counter(midi_to_name(note.midi_note) for note in notes)
    silences = sum(isinstance(e.event, SilenceEvent) for e in events)
    note_list = ", ".join(f"{name} x{count}" for name, count in counts.most_common())
    return (
        f"{len(notes)} notes ({note_list or 'none'}), {silences} silences. "
        f"{stats.hops_processed} hops, mean {stats.hop_mean_seconds * 1e6:.0f} us, "
        f"max {stats.hop_max_seconds * 1e6:.0f} us. "
        f"{duration_seconds:.1f} s of audio in {wall_seconds:.2f} s."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, object] = {"instrument": args.instrument}
    if args.level_gate is not None:
        overrides["level_gate_dbfs"] = args.level_gate
    if args.yin_threshold is not None:
        overrides["yin_threshold"] = args.yin_threshold
    try:
        config = AnalysisConfig.model_validate(overrides)
    except ValidationError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    samples, sample_rate_hz = sf.read(args.path, dtype="float32", always_2d=True)
    expected_rate_hz = AudioConfig().sample_rate_hz
    if sample_rate_hz != expected_rate_hz:
        print(
            f"error: {args.path} is {sample_rate_hz} Hz; JamMate expects {expected_rate_hz} Hz.",
            file=sys.stderr,
        )
        return 1
    if not 1 <= args.channel <= samples.shape[1]:
        print(f"error: {args.path} has {samples.shape[1]} channel(s).", file=sys.stderr)
        return 1
    mono = np.ascontiguousarray(samples[:, args.channel - 1])
    duration_seconds = len(mono) / sample_rate_hz
    min_hz, max_hz = config.pitch_range_hz
    print(
        f"{args.path}: {duration_seconds:.2f} s, {args.instrument} {min_hz:.0f}-{max_hz:.0f} Hz, "
        f"hop {config.hop_frames}, gate {config.level_gate_dbfs:.0f} dBFS"
    )

    lines: list[tuple[int, int, str]] = []  # (frame, order, text): hops sort before events
    hop_printer = None
    if args.hops:

        def hop_printer(frame: PitchFrame) -> None:
            lines.append((frame.frame_index, 0, format_hop(frame, sample_rate_hz)))

    started = time.perf_counter()
    events, stats = analyze_offline(
        mono, config, sample_rate_hz, block_frames=args.block_size, on_pitch_frame=hop_printer
    )
    wall_seconds = time.perf_counter() - started
    lines.extend((e.event.frame_index, 1, format_event(e, sample_rate_hz)) for e in events)
    for _, _, text in sorted(lines, key=lambda line: (line[0], line[1])):
        print(text)
    print(format_summary(events, stats, duration_seconds, wall_seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

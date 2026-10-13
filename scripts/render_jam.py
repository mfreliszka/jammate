"""Render a riff WAV through analysis, the drummer, and optional AI accompaniment (Steps 3.5, 4.6).

Feeds a mono or multichannel WAV through `BlockProcessor`, `AnalysisWorker`, `BeatClock`,
`MusicalStateManager`, `DrumEngine`, `DrumSampler`, and optionally `AccompanimentEngine` in
128-frame blocks, writing the mixed stereo output (`monitor + drums + AI`) to a WAV file and
printing a summary of tempo lock, drummer entry, AI entry, fills, and output levels.

Examples:
    uv run python scripts/render_jam.py tests/data/guitar_riff_e2_g2_a2_92bpm.wav out_92.wav \
        --instrument guitar
    uv run python scripts/render_jam.py tests/data/guitar_riff_e2_g2_a2_120bpm.wav out_120.wav \
        --instrument guitar --style stoner_rock --ai
    uv run python scripts/render_jam.py tests/data/guitar_open_strings.wav out_manual.wav \
        --instrument guitar --bpm 100
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import ValidationError

from jammate.ai.base import AccompanimentEngine, GuardedAccompaniment, NoteTracker
from jammate.ai.mrt import create_accompaniment_engine, shutdown_mlx_runtime
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import AnalysisWorker
from jammate.audio.engine import BlockProcessor
from jammate.audio.levels import measure_level
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import (
    AccompanimentConfig,
    AnalysisConfig,
    AudioConfig,
    DrumConfig,
    JamMateConfig,
    StyleFamily,
    TempoConfig,
)
from jammate.drums.engine import DrumEngine, DrummerState
from jammate.drums.sampler import DrumSampler
from jammate.music.beat_clock import BeatClock
from jammate.music.events import (
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    TempoUpdate,
    frames_to_seconds,
)
from jammate.music.state import MusicalStateManager
from jammate.music.styles import AiEntryController


@dataclass(frozen=True, slots=True)
class RenderSummary:
    """Offline render metrics for a single WAV file."""

    duration_seconds: float
    wall_seconds: float
    notes_detected: int
    onsets_detected: int
    tempo_locked: bool
    locked_bpm: float | None
    tempo_confidence: float
    first_lock_seconds: float | None
    drummer_entered: bool
    entry_bar: int | None
    entry_seconds: float | None
    events_scheduled: int
    fill_bars: tuple[int, ...]
    tiers_visited: tuple[str, ...]
    states_visited: tuple[str, ...]
    peak_dbfs: float
    rms_dbfs: float
    stolen_voices: int
    dropped_events: int
    max_boundary_jump: float
    ai_enabled: bool = False
    ai_entered: bool = False
    ai_entry_bar: int | None = None
    ai_frames_generated: int = 0
    ai_ms_per_frame: float = 0.0
    ai_underruns: int = 0


def render_jam_offline(
    mono: np.ndarray,
    config: JamMateConfig | None = None,
    *,
    accompaniment_engine: AccompanimentEngine | None = None,
) -> tuple[np.ndarray, RenderSummary]:
    """Run `mono` (1D float32 at `config.audio.sample_rate_hz`) through analysis + drummer + AI."""
    cfg = config or JamMateConfig()
    sample_rate_hz = cfg.audio.sample_rate_hz
    block_frames = cfg.audio.block_size_frames
    manual_bpm = cfg.analysis.tempo.manual_bpm

    total_frames = len(mono)
    padded_frames = ((total_frames + block_frames - 1) // block_frames) * block_frames
    if padded_frames > total_frames:
        padded = np.zeros(padded_frames, dtype=np.float32)
        padded[:total_frames] = mono
    else:
        padded = np.ascontiguousarray(mono, dtype=np.float32)

    input_ring = RingBuffer(cfg.audio.input_ring_capacity_frames, channels=1)
    drum_sampler = DrumSampler.from_config(cfg.drums, sample_rate_hz=sample_rate_hz)
    raw_ai = accompaniment_engine or create_accompaniment_engine(cfg.ai, sample_rate_hz)
    ai = GuardedAccompaniment(raw_ai)
    note_tracker = NoteTracker(ai)
    ai_entry = AiEntryController(cfg.ai, beats_per_bar=cfg.beat_clock.beats_per_bar)

    processor = BlockProcessor(
        input_ring,
        input_channel_index=0,
        monitor_gain=cfg.audio.monitor_gain,
        monitor_muted=cfg.audio.monitor_muted,
        drums_gain=cfg.audio.drums_gain,
        drums_muted=cfg.audio.drums_muted,
        drum_sampler=drum_sampler,
        ai_gain=cfg.audio.ai_gain,
        ai_muted=cfg.audio.ai_muted,
        ai_ring=None,
        soft_clip_threshold=cfg.audio.soft_clip_threshold,
    )
    state_manager = MusicalStateManager(cfg.state, manual_bpm=manual_bpm)
    beat_clock = BeatClock(
        cfg.beat_clock,
        sample_rate_hz=sample_rate_hz,
        manual_bpm=manual_bpm,
    )
    drums = DrumEngine(
        cfg.drums,
        sample_rate_hz=sample_rate_hz,
        event_ring=drum_sampler.event_ring,
        beat_clock=beat_clock,
        state_provider=lambda: state_manager.snapshot,
        frame_provider=lambda: processor.frames_processed,
        manual_bpm_active=manual_bpm is not None,
    )

    notes_detected = 0
    onsets_detected = 0
    first_lock_seconds: float | None = 0.0 if manual_bpm is not None else None
    ai_entered = False

    def sync_ai_entry() -> None:
        nonlocal ai_entered
        ai_entry.update(
            state_manager.snapshot,
            drums.drummer_state,
            drums.controller.entry_bar,
        )
        if ai_entry.is_active:
            ai_entered = True
        set_gate = getattr(ai.inner, "set_gate_gain", None)
        if callable(set_gate):
            set_gate(ai_entry.gate_gain)
        processor.ai_ring = ai.output_ring if (cfg.ai.enabled and ai_entry.is_active) else None

    def on_event(event: MusicalEvent) -> None:
        nonlocal notes_detected, onsets_detected, first_lock_seconds
        if isinstance(event, PitchEvent):
            notes_detected += 1
        elif isinstance(event, OnsetEvent):
            onsets_detected += 1
        elif isinstance(event, TempoUpdate) and event.locked and first_lock_seconds is None:
            first_lock_seconds = event.timestamp_seconds
        beat_clock.handle_event(event)
        state_manager.apply(event)
        if beat_clock.is_running or state_manager.snapshot.beat is not None:
            state_manager.update_beat(event.frame_index, beat_clock.position_at(event.frame_index))
        note_tracker.handle_event(event)
        sync_ai_entry()

    def on_hop(frame: PitchFrame) -> None:
        if beat_clock.is_running or state_manager.snapshot.beat is not None:
            state_manager.update_beat(frame.frame_index, beat_clock.position_at(frame.frame_index))
        sync_ai_entry()

    worker = AnalysisWorker(
        input_ring,
        cfg.analysis,
        sample_rate_hz=sample_rate_hz,
        on_event=on_event,
    )
    worker.set_hop_listener(on_hop)

    rendered = np.zeros((padded_frames, 2), dtype=np.float32)
    in_block = np.zeros((block_frames, 1), dtype=np.float32)
    out_block = np.zeros((block_frames, 2), dtype=np.float32)

    states_visited: list[str] = [drums.drummer_state.value]
    tiers_visited: list[str] = []
    fill_bars_seen: list[int] = []
    entry_seconds: float | None = 0.0 if manual_bpm is not None else None
    max_boundary_jump = 0.0
    prev_last_sample: np.ndarray | None = None
    generate_chunk = getattr(ai.inner, "generate_chunk", None)

    started = time.perf_counter()
    # Pre-schedule at frame 0 when manual_bpm is active so bar 1 beat 1 is ready before block 0.
    drums.schedule_available(current_frame=0)
    sync_ai_entry()

    for start in range(0, padded_frames, block_frames):
        end = start + block_frames
        if cfg.ai.enabled and callable(generate_chunk) and processor.ai_ring is not None:
            while processor.ai_ring.available_frames() < block_frames:
                generate_chunk()

        in_block[:, 0] = padded[start:end]
        processor.process(in_block, out_block)
        if prev_last_sample is not None:
            jump = float(np.max(np.abs(out_block[0] - prev_last_sample)))
            if jump > max_boundary_jump:
                max_boundary_jump = jump
        prev_last_sample = out_block[-1].copy()
        rendered[start:end] = out_block

        worker.process_available()
        drums.schedule_available(current_frame=processor.frames_processed)
        sync_ai_entry()

        current_state = drums.drummer_state.value
        if current_state != states_visited[-1]:
            states_visited.append(current_state)
            if (
                drums.drummer_state == DrummerState.PLAYING
                and entry_seconds is None
                and drums.controller.entry_bar is not None
            ):
                entry_seconds = frames_to_seconds(processor.frames_processed, sample_rate_hz)

        if drums.drummer_state in (DrummerState.PLAYING, DrummerState.HOLDING):
            tier = drums.tier
            if not tiers_visited or tiers_visited[-1] != tier:
                tiers_visited.append(tier)
            bar = state_manager.snapshot.bar
            if bar is not None:
                pattern = drums.controller.pattern_for_bar(bar)
                if (
                    pattern is not None
                    and pattern.name.endswith("_fill")
                    and (not fill_bars_seen or fill_bars_seen[-1] != bar)
                ):
                    fill_bars_seen.append(bar)

    note_tracker.all_notes_off()
    wall_seconds = time.perf_counter() - started
    output = rendered[:total_frames]
    final_state = state_manager.snapshot
    ai_stats = ai.stats()

    # If the drummer armed entry_bar and scheduled its hits within the look-ahead window right
    # before the recording ended, record the entry time from the beat clock.
    drummer_entered = (
        DrummerState.PLAYING.value in states_visited or drums.scheduler.events_scheduled > 0
    )
    if (
        drummer_entered
        and entry_seconds is None
        and drums.controller.entry_bar is not None
        and beat_clock.snapshot is not None
    ):
        entry_beat = (drums.controller.entry_bar - 1) * cfg.beat_clock.beats_per_bar
        entry_frame = round(
            beat_clock.snapshot.anchor_frame
            + entry_beat * (60.0 * sample_rate_hz / beat_clock.snapshot.bpm)
        )
        entry_seconds = frames_to_seconds(entry_frame, sample_rate_hz)

    level = measure_level(output)
    summary = RenderSummary(
        duration_seconds=total_frames / sample_rate_hz,
        wall_seconds=wall_seconds,
        notes_detected=notes_detected,
        onsets_detected=onsets_detected,
        tempo_locked=final_state.tempo_locked,
        locked_bpm=final_state.bpm if final_state.tempo_locked else None,
        tempo_confidence=final_state.tempo_confidence,
        first_lock_seconds=first_lock_seconds,
        drummer_entered=drummer_entered,
        entry_bar=drums.controller.entry_bar,
        entry_seconds=entry_seconds,
        events_scheduled=drums.scheduler.events_scheduled,
        fill_bars=tuple(fill_bars_seen),
        tiers_visited=tuple(tiers_visited),
        states_visited=tuple(states_visited),
        peak_dbfs=level.peak_dbfs,
        rms_dbfs=level.rms_dbfs,
        stolen_voices=drum_sampler.stolen_voices,
        dropped_events=drum_sampler.event_ring.dropped_events,
        max_boundary_jump=max_boundary_jump,
        ai_enabled=cfg.ai.enabled,
        ai_entered=ai_entered,
        ai_entry_bar=ai_entry.entry_bar,
        ai_frames_generated=ai_stats.frames_generated,
        ai_ms_per_frame=ai_stats.ms_per_frame,
        ai_underruns=processor.ai_underruns if cfg.ai.enabled and ai_entered else 0,
    )
    return output, summary


def format_render_summary(summary: RenderSummary) -> str:
    bpm_str = f"{summary.locked_bpm:.1f} BPM" if summary.locked_bpm is not None else "unlocked"
    lock_at = (
        f" (locked at {summary.first_lock_seconds:.2f} s)"
        if summary.first_lock_seconds is not None
        else ""
    )
    if summary.drummer_entered:
        at_str = f" at {summary.entry_seconds:.2f} s" if summary.entry_seconds is not None else ""
        entry_str = f"entered bar {summary.entry_bar}{at_str}"
    else:
        entry_str = "did not enter"
    states = " -> ".join(summary.states_visited)
    tiers = ", ".join(summary.tiers_visited) or "none"
    fills = ", ".join(str(b) for b in summary.fill_bars) or "none"
    lines = [
        f"Duration: {summary.duration_seconds:.2f} s (rendered in {summary.wall_seconds:.2f} s)",
        f"Analysis: {summary.notes_detected} notes, {summary.onsets_detected} onsets, "
        f"tempo {bpm_str}{lock_at} (conf {summary.tempo_confidence:.2f})",
        f"Drummer:  {entry_str}, {summary.events_scheduled} events scheduled, "
        f"states [{states}], tiers [{tiers}], fill bars [{fills}]",
    ]
    if summary.ai_enabled:
        ai_entry_str = (
            f"entered bar {summary.ai_entry_bar}" if summary.ai_entered else "did not enter"
        )
        lines.append(
            f"AI:       {ai_entry_str}, {summary.ai_frames_generated} frames "
            f"({summary.ai_ms_per_frame:.1f} ms/frame), underruns {summary.ai_underruns}"
        )
    lines.append(
        f"Output:   peak {summary.peak_dbfs:.1f} dBFS, rms {summary.rms_dbfs:.1f} dBFS, "
        f"max boundary jump {summary.max_boundary_jump:.4f}, "
        f"dropped {summary.dropped_events}, stolen {summary.stolen_voices}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a riff WAV through analysis, drummer, and optional AI to a stereo WAV."
    )
    parser.add_argument("input_path", type=Path, help="Input mono or multichannel WAV (48 kHz)")
    parser.add_argument("output_path", type=Path, help="Output stereo WAV path")
    parser.add_argument("--instrument", choices=["bass", "guitar"], default="bass")
    parser.add_argument(
        "--style",
        choices=["rock", "stoner_rock", "blues", "funk", "metal"],
        default="rock",
        help="Style preset for drums and AI (default: rock)",
    )
    parser.add_argument(
        "--channel", type=int, default=1, help="Input channel (1-based, default: 1)"
    )
    parser.add_argument("--block-size", type=int, default=128, help="Frames per block (128 or 256)")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override (50-200)")
    parser.add_argument("--monitor-gain", type=float, default=1.0, help="Monitor bus gain (0-4)")
    parser.add_argument("--drums-gain", type=float, default=0.8, help="Drums bus gain (0-4)")
    parser.add_argument("--ai", action="store_true", help="Enable MRT2 AI accompaniment")
    parser.add_argument("--ai-gain", type=float, default=0.7, help="AI bus gain (0-4)")
    parser.add_argument(
        "--follow-improvise",
        type=float,
        default=0.5,
        help="Follow (0.0) <-> Improvise (1.0) control",
    )
    parser.add_argument(
        "--timing-offset",
        type=float,
        default=0.0,
        help="Drum timing offset in seconds (-0.1 to 0.1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        style: StyleFamily = args.style
        config = JamMateConfig(
            audio=AudioConfig(
                block_size_frames=args.block_size,
                monitor_gain=args.monitor_gain,
                drums_gain=args.drums_gain,
                ai_gain=args.ai_gain,
                ai_muted=not args.ai,
            ),
            analysis=AnalysisConfig(
                instrument=args.instrument,
                tempo=TempoConfig(manual_bpm=args.bpm),
            ),
            drums=DrumConfig(
                style=style,
                timing_offset_seconds=args.timing_offset,
            ),
            ai=AccompanimentConfig(
                enabled=args.ai,
                style=style,
                follow_improvise=args.follow_improvise,
            ),
        )
    except ValidationError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    if not args.input_path.exists():
        print(f"error: input file not found: {args.input_path}", file=sys.stderr)
        return 1

    samples, sample_rate_hz = sf.read(args.input_path, dtype="float32", always_2d=True)
    if sample_rate_hz != config.audio.sample_rate_hz:
        print(
            f"error: {args.input_path} is {sample_rate_hz} Hz; "
            f"JamMate expects {config.audio.sample_rate_hz} Hz.",
            file=sys.stderr,
        )
        return 1
    if not 1 <= args.channel <= samples.shape[1]:
        print(f"error: {args.input_path} has {samples.shape[1]} channel(s).", file=sys.stderr)
        return 1

    mono = np.ascontiguousarray(samples[:, args.channel - 1], dtype=np.float32)
    try:
        rendered, summary = render_jam_offline(mono, config)
    finally:
        if args.ai:
            shutdown_mlx_runtime()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_path, rendered, sample_rate_hz, subtype="FLOAT")
    print(f"Wrote {args.output_path} ({len(rendered)} stereo frames @ {sample_rate_hz} Hz)")
    print(format_render_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

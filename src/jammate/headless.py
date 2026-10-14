"""Headless runner: runs a `JamSession` from the terminal (until the GUI).

The session's analysis worker updates the Musical State and hands each event on to this runner
through a bounded deque. The main thread prints each note change, playing-state change and tempo
lock or unlock as it happens. About twice per second it prints a status line: the musical part
comes from the Musical State snapshot, the rest from the engine's and the worker's diagnostics.
The session owns startup and shutdown (§83, §84); this runner adds Ctrl+C and SIGTERM handling.
"""

import logging
import signal
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import TextIO

from jammate.analysis.notes import cents_offset, midi_to_name
from jammate.analysis.worker import AnalysisStats
from jammate.audio.devices import DeviceResolutionError
from jammate.audio.engine import AudioEngine, AudioEngineError, EngineStats
from jammate.audio.levels import SILENCE_FLOOR_DBFS
from jammate.config import JamMateConfig
from jammate.music.events import (
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingStateUpdate,
    SilenceEvent,
    TempoUpdate,
)
from jammate.music.state import MusicalState
from jammate.session import EngineFactory, JamSession, WorkerFactory, default_worker_factory

logger = logging.getLogger(__name__)

STATUS_INTERVAL_SECONDS = 0.5
EVENT_POLL_SECONDS = 0.05  # how often new events are printed
MAX_PENDING_EVENTS = 256  # bound on events waiting for the main thread


@dataclass(frozen=True)
class StatusSnapshot:
    elapsed_seconds: float
    ring_fill_ratio: float
    block_period_seconds: float
    clipped: bool  # the input clipped since the previous status line
    state: MusicalState
    stats: EngineStats
    analysis: AnalysisStats
    session_riffs: int = 0


def format_elapsed(elapsed_seconds: float) -> str:
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_stream_time(timestamp_seconds: float) -> str:
    minutes, seconds = divmod(timestamp_seconds, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def format_note(state: MusicalState) -> str:
    return "--" if state.current_note is None else midi_to_name(state.current_note)


def format_tempo(state: MusicalState) -> str:
    if state.bpm is None:
        return "tempo   --"
    lock = "locked" if state.tempo_locked else "search"
    text = f"tempo {state.bpm:5.1f} {lock} {state.tempo_confidence:.2f}"
    if state.bar is not None and state.beat is not None:
        text += f"  bar {state.bar}:{state.beat}"
    return text


def format_state_summary(
    state: MusicalState,
    tempo_locks: int,
    *,
    session_riffs: int = 0,
) -> str:
    """The Musical State in one log line, for the end of a run."""
    recent = " ".join(midi_to_name(note) for note in state.recent_notes) or "none"
    if state.bpm is None:
        tempo = "no tempo estimate"
    else:
        lock = "locked at" if state.tempo_locked else "not locked, last estimate"
        tempo = (
            f"tempo {lock} {state.bpm:.1f} BPM (confidence {state.tempo_confidence:.2f}, "
            f"tempo locks: {tempo_locks})"
        )
    summary = (
        f"Musical state: note {format_note(state)}, recent notes {recent}, {tempo}, "
        f"intensity {state.intensity:.2f}, {state.playing_state}."
    )
    if state.key_label is not None:
        summary += f" Key: {state.key_label} (confidence {state.harmonic_confidence:.2f})."
    if state.riff_id is not None or session_riffs > 0:
        riff_label = state.riff_id or "none"
        summary += f" Riff: {riff_label} (x{state.riff_iteration}, memory: {session_riffs} riffs)."
    return summary


def format_event(event: MusicalEvent) -> str | None:
    """One timeline line per note change, playing-state change or tempo update. Onsets (several
    per second) and intensity are shown in the status line instead. The runner prints a tempo
    update only when it locks or unlocks."""
    when = format_stream_time(event.timestamp_seconds)
    if isinstance(event, PitchEvent):
        cents = cents_offset(event.frequency_hz, event.midi_note)
        return (
            f"  {when}  {midi_to_name(event.midi_note):4s} {event.frequency_hz:7.2f} Hz "
            f"{cents:+4.0f} c  conf {event.confidence:.2f}"
        )
    if isinstance(event, SilenceEvent):
        return f"  {when}  --   silence"
    if isinstance(event, PlayingStateUpdate):
        return f"  {when}  [{event.state}]"
    if isinstance(event, TempoUpdate):
        if event.locked and event.bpm is not None:
            return f"  {when}  [tempo locked {event.bpm:.1f} BPM]"
        about = "" if event.bpm is None else f", now about {event.bpm:.0f} BPM"
        return f"  {when}  [tempo unlocked{about}]"
    return None


def format_status(snapshot: StatusSnapshot) -> str:
    state = snapshot.state
    stats = snapshot.stats
    analysis = snapshot.analysis
    timing = stats.callback_timing
    load_percent = 100.0 * timing.max_seconds / snapshot.block_period_seconds
    rms_dbfs = SILENCE_FLOOR_DBFS if analysis.latest is None else analysis.latest.rms_dbfs
    clip = " CLIP" if snapshot.clipped else ""
    key_str = (
        f"  key {state.key_label} {state.harmonic_confidence:.2f}"
        if state.key_label is not None
        else ""
    )
    riff_str = (
        f"  riff {state.riff_id} x{state.riff_iteration} (mem {snapshot.session_riffs})"
        if state.riff_id is not None
        else ""
    )
    line = (
        f"{format_elapsed(snapshot.elapsed_seconds)}"
        f"  note {format_note(state):4s} {state.playing_state:10s}"
        f" int {state.intensity:.2f}"
        f"  {format_tempo(state)}"
        f"{key_str}"
        f"{riff_str}"
        f"  level rms {rms_dbfs:6.1f} peak {analysis.recent_peak_dbfs:6.1f} dBFS{clip}"
        f"  xruns in {stats.input_overflows} out {stats.output_underflows}"
        f" (start-up {stats.startup_xruns})"
        f"  ring {100.0 * snapshot.ring_fill_ratio:3.0f}%"
        f"  callback {timing.mean_seconds * 1e6:3.0f}/{timing.max_seconds * 1e6:4.0f} us"
        f" ({load_percent:.0f}%)"
        f"  analysis {analysis.hop_mean_seconds * 1e6:3.0f}"
        f"/{analysis.hop_max_seconds * 1e6:4.0f} us"
    )
    if analysis.dropped_frames:
        line += f"  DROPPED {analysis.dropped_frames}"
    if stats.callback_errors:
        line += f"  CALLBACK ERRORS {stats.callback_errors}"
    if analysis.errors:
        line += f"  ANALYSIS ERRORS {analysis.errors}"
    return line


class _RunSummary:
    """Whole-run figures. The callback timing window (~5 s) is longer than the status interval,
    so polling every interval sees every block's duration."""

    def __init__(self) -> None:
        self.max_callback_seconds = 0.0
        self.notes = 0
        self.onsets = 0
        self.user_was_playing = False
        self.tempo_locked = False
        self.tempo_locks = 0

    def update(self, stats: EngineStats) -> None:
        self.max_callback_seconds = max(
            self.max_callback_seconds, stats.callback_timing.max_seconds
        )


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def run_headless(
    config: JamMateConfig,
    *,
    engine_factory: EngineFactory = AudioEngine.from_config,
    worker_factory: WorkerFactory = default_worker_factory,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    output: TextIO = sys.stdout,
    status_interval_seconds: float = STATUS_INTERVAL_SECONDS,
) -> int:
    """Run a jam session until Ctrl+C or SIGTERM, then shut it down cleanly.

    Returns an exit code.
    """
    # Deque appends and pops are atomic, so the worker thread can hand events to this thread.
    pending_events: deque[MusicalEvent] = deque(maxlen=MAX_PENDING_EVENTS)
    try:
        session = JamSession(
            config,
            on_event=pending_events.append,
            engine_factory=engine_factory,
            worker_factory=worker_factory,
        )
    except (DeviceResolutionError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    engine = session.engine
    worker = session.worker

    live_line = output.isatty()
    summary = _RunSummary()
    exit_code = 0
    started_seconds = clock()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        info = session.start()
        started_seconds = clock()
        next_status_seconds = started_seconds + status_interval_seconds
        reported_callback_errors = 0
        reported_analysis_errors = 0
        clipped_hops = 0
        status_line = ""
        output.write("Listening. Press Ctrl+C to stop.\n")
        output.flush()

        while True:
            sleep(EVENT_POLL_SECONDS)
            if not session.is_running:
                logger.error("The audio stream stopped unexpectedly (interface disconnected?).")
                exit_code = 1
                break

            event_lines = _drain_events(pending_events, summary)
            if event_lines:
                for event_line in event_lines:
                    output.write(f"\r{event_line}\x1b[K\n" if live_line else f"{event_line}\n")
                if live_line and status_line:
                    output.write(status_line)
                output.flush()

            now_seconds = clock()
            if now_seconds < next_status_seconds:
                continue
            next_status_seconds = now_seconds + status_interval_seconds

            stats = engine.stats()
            analysis = worker.stats()
            summary.update(stats)
            if stats.callback_errors > reported_callback_errors:
                reported_callback_errors = stats.callback_errors
                logger.error("Audio callback error: %r", engine.processor.last_error)
            if analysis.errors > reported_analysis_errors:
                reported_analysis_errors = analysis.errors
                logger.error("Analysis error: %r", worker.last_error)

            ring = engine.input_ring
            line = format_status(
                StatusSnapshot(
                    elapsed_seconds=now_seconds - started_seconds,
                    ring_fill_ratio=ring.available_frames() / ring.capacity_frames,
                    block_period_seconds=info.block_period_seconds,
                    clipped=analysis.clipped_hops > clipped_hops,
                    state=session.state,
                    stats=stats,
                    analysis=analysis,
                    session_riffs=len(session.riff_detector.memory),
                )
            )
            clipped_hops = analysis.clipped_hops
            status_line = f"\r{line}\x1b[K" if live_line else f"{line}\n"
            output.write(status_line)
            output.flush()
    except KeyboardInterrupt:
        pass
    except AudioEngineError as exc:
        logger.error("%s", exc)
        exit_code = 1
    finally:
        # A second Ctrl+C must not interrupt the shutdown, which the session orders (§84).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if live_line:
                output.write("\n")
            session.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)

    _log_summary(session, clock() - started_seconds, summary)
    return exit_code


def _drain_events(pending_events: deque[MusicalEvent], summary: _RunSummary) -> list[str]:
    lines = []
    while pending_events:
        event = pending_events.popleft()
        if isinstance(event, PitchEvent):
            summary.notes += 1
        elif isinstance(event, OnsetEvent):
            summary.onsets += 1
        elif isinstance(event, PlayingStateUpdate):
            # PLAYING <-> SUSTAINING happens at every pause between attacks; both are playing.
            if event.user_is_playing and summary.user_was_playing:
                continue
            summary.user_was_playing = event.user_is_playing
        elif isinstance(event, TempoUpdate):
            # Tempo updates come with every small change; show only locks and unlocks.
            if event.locked == summary.tempo_locked:
                continue
            summary.tempo_locked = event.locked
            if event.locked:
                summary.tempo_locks += 1
        line = format_event(event)
        if line is not None:
            lines.append(line)
    return lines


def _log_summary(session: JamSession, elapsed_seconds: float, summary: _RunSummary) -> None:
    info = session.engine.stream_info
    if info is None:
        return
    stats = session.engine.stats()
    analysis = session.worker.stats()
    logger.info(
        "Session: %s, %d blocks of %d frames at %d Hz. Xruns: %d input overflows, "
        "%d output underflows (%d at start-up). Callback max %.0f us (%.0f%% of the block "
        "period). Input peak %.1f dBFS. Callback errors: %d.",
        format_elapsed(elapsed_seconds),
        stats.blocks_processed,
        info.block_size_frames,
        info.sample_rate_hz,
        stats.input_overflows,
        stats.output_underflows,
        stats.startup_xruns,
        summary.max_callback_seconds * 1e6,
        100.0 * summary.max_callback_seconds / info.block_period_seconds,
        analysis.max_peak_dbfs,
        stats.callback_errors,
    )
    logger.info(
        "Analysis: %d hops, mean %.0f us, max %.0f us. %d notes, %d onsets. Dropped frames: %d "
        "(%d skips). Analysis errors: %d.",
        analysis.hops_processed,
        analysis.hop_mean_seconds * 1e6,
        analysis.hop_max_seconds * 1e6,
        summary.notes,
        summary.onsets,
        analysis.dropped_frames,
        analysis.skips,
        analysis.errors,
    )
    logger.info(
        "%s",
        format_state_summary(
            session.state,
            summary.tempo_locks,
            session_riffs=len(session.riff_detector.memory),
        ),
    )

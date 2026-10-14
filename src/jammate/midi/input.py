"""MIDI keyboard input worker, device discovery, and chord root analysis (Step 6.3, §55, §150).

Listens for `note_on` / `note_off` events on a selected MIDI input port off the audio thread,
stamps each event with the current audio stream `frame_index`, tracks active polyphonic MIDI notes,
and emits `PitchEvent` (`NoteObservation`: lowest/triad root note + full active pitch set for
`NoteTracker` pianoroll conditioning), `OnsetEvent` (`OnsetObservation`: scaled from MIDI velocity
`1..127`), `SilenceEvent`, `PlayingStateUpdate`, `IntensityUpdate`, `TempoUpdate`, and `KeyUpdate`.

Realtime safety:
- Never runs on or touches the PortAudio callback thread.
- `stop()` and disconnect error handling immediately release all active notes so zero stuck notes
  remain in `MusicalState` or `NoteTracker` (§69, §84, §137).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import threading
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import mido

from jammate.analysis.key import KeyEstimate, KeyEstimator, classify_register
from jammate.analysis.notes import midi_to_hz
from jammate.analysis.tempo import TempoTracker
from jammate.config import AnalysisConfig, Instrument
from jammate.music.events import (
    HIGHEST_MIDI_NOTE,
    LOWEST_MIDI_NOTE,
    IntensityUpdate,
    KeyUpdate,
    MidiNote,
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SampleFrame,
    SilenceEvent,
    frames_to_seconds,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.002
STOP_TIMEOUT_SECONDS = 2.0
ALL_SOUND_OFF_CC = 120
ALL_NOTES_OFF_CC = 123

type MidiChordType = Literal["power", "major", "minor"]
type NoteObservation = PitchEvent
type OnsetObservation = OnsetEvent


class MidiDeviceResolutionError(ValueError):
    """Raised when a requested MIDI input device name/substring cannot be resolved."""


class MidiStatus(StrEnum):
    """Lifecycle and health state of the MIDI input subsystem (§55, §138, §150)."""

    DISABLED = "disabled"
    LISTENING = "listening"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MidiChordAnalysis:
    """Extracted root, active chord notes, and chord quality from held MIDI notes (§150)."""

    root_midi: MidiNote
    chord_notes: tuple[MidiNote, ...]
    chord_type: MidiChordType | None


@dataclass(frozen=True, slots=True)
class MidiStats:
    """Diagnostic snapshot of the MIDI input worker (§138, §150)."""

    status: MidiStatus = MidiStatus.DISABLED
    port_name: str | None = None
    active_notes: tuple[MidiNote, ...] = ()
    messages_processed: int = 0
    notes_triggered: int = 0
    last_velocity: int = 0
    last_error: str | None = None


@runtime_checkable
class MidiPort(Protocol):
    """Minimal non-blocking MIDI input port interface (`mido` ports and `FakeMidiPort`)."""

    def iter_pending(self) -> Iterable[Any]: ...

    def close(self) -> None: ...


class FakeMidiPort:
    """Hardware-free in-memory MIDI port for deterministic unit tests (Step 6.3, §67)."""

    def __init__(self, name: str = "Fake MIDI Keyboard") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._queue: collections.deque[mido.Message] = collections.deque()
        self._closed = False
        self._disconnected = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def send(self, message: mido.Message) -> None:
        """Queue a raw `mido.Message` for the next `iter_pending()` pass."""
        with self._lock:
            self._queue.append(message)

    def note_on(self, note: int, velocity: int = 100, *, channel: int = 0) -> None:
        """Queue a `note_on` message (`velocity=0` acts as `note_off` per the MIDI spec)."""
        self.send(mido.Message("note_on", note=note, velocity=velocity, channel=channel))

    def note_off(self, note: int, velocity: int = 0, *, channel: int = 0) -> None:
        """Queue a `note_off` message."""
        self.send(mido.Message("note_off", note=note, velocity=velocity, channel=channel))

    def control_change(self, control: int, value: int = 0, *, channel: int = 0) -> None:
        """Queue a `control_change` message (e.g. CC 123 All Notes Off)."""
        self.send(mido.Message("control_change", control=control, value=value, channel=channel))

    def disconnect(self) -> None:
        """Simulate unplugging the USB MIDI controller mid-session."""
        with self._lock:
            self._disconnected = True

    def iter_pending(self) -> list[mido.Message]:
        with self._lock:
            if self._disconnected:
                raise OSError(f"MIDI port {self.name!r} disconnected")
            if self._closed:
                return []
            messages = list(self._queue)
            self._queue.clear()
            return messages

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queue.clear()


def list_midi_inputs(
    backend_names_fn: Callable[[], Iterable[str]] | None = None,
) -> list[str]:
    """Return available MIDI input port names (deduplicated, preserving order)."""
    provider = backend_names_fn or mido.get_input_names
    try:
        raw_names = list(provider())
    except Exception as exc:
        logger.warning("Failed to enumerate MIDI input ports: %s", exc)
        return []
    seen: set[str] = set()
    result: list[str] = []
    for name in raw_names:
        cleaned = name.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def resolve_midi_input(
    query: str | None,
    available: Sequence[str] | None = None,
) -> str | None:
    """Resolve `query` (exact name or unique case-insensitive substring) against `available`."""
    if query is None:
        return None
    needle = query.strip()
    if not needle or needle.lower() == "none":
        return None
    ports = list(available) if available is not None else list_midi_inputs()
    if not ports:
        raise MidiDeviceResolutionError(
            f"No MIDI input devices are available (requested {query!r})."
        )
    for port in ports:
        if port == needle:
            return port
    needle_lower = needle.lower()
    for port in ports:
        if port.lower() == needle_lower:
            return port
    matches = [port for port in ports if needle_lower in port.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available_str = ", ".join(repr(p) for p in ports)
        raise MidiDeviceResolutionError(
            f"No MIDI input device matches {query!r}. Available: {available_str}"
        )
    matches_str = ", ".join(repr(p) for p in matches)
    raise MidiDeviceResolutionError(
        f"MIDI input query {query!r} is ambiguous; matches: {matches_str}"
    )


def format_midi_device_list(
    devices: Sequence[str],
    selected: str | None = None,
) -> str:
    """Format a human-readable list of MIDI input ports for `--list-midi`."""
    if not devices:
        return "MIDI Input Devices:\n  (none found)"
    lines = ["MIDI Input Devices:"]
    for idx, name in enumerate(devices, start=1):
        marker = "*" if selected is not None and name == selected else " "
        lines.append(f" {marker} [{idx}] {name}")
    return "\n".join(lines)


def open_midi_input_port(port_name: str) -> MidiPort:
    """Open a `mido` input port by exact resolved name."""
    return mido.open_input(port_name)  # type: ignore[return-value]


def velocity_to_strength(velocity: int) -> float:
    """Map MIDI velocity `1..127` to linear onset strength `0.0..1.0` (§55, §150)."""
    if velocity <= 0:
        return 0.0
    return min(1.0, max(0.0, velocity / 127.0))


def analyze_midi_notes(active_notes: Collection[MidiNote]) -> MidiChordAnalysis | None:
    """Extract the root MIDI note, chord notes tuple, and chord quality from held MIDI notes (§150).

    - Single held note: `root_midi = note`, `chord_notes = ()`, `chord_type = None`.
    - Multiple held notes:
      - Checks whether the lowest bass note forms a major triad (`0, 4, 7`), minor triad
        (`0, 3, 7`), or power chord (`0, 7`).
      - If not, checks whether the held pitch classes form a first- or second-inversion major or
        minor triad and extracts the triad root pitch class.
      - Otherwise defaults `root_midi` to the lowest active note.
    """
    valid = sorted({n for n in active_notes if LOWEST_MIDI_NOTE <= n <= HIGHEST_MIDI_NOTE})
    if not valid:
        return None
    if len(valid) == 1:
        return MidiChordAnalysis(root_midi=valid[0], chord_notes=(), chord_type=None)

    notes_tuple = tuple(valid)
    lowest = valid[0]
    lowest_pc = lowest % 12
    pcs = {n % 12 for n in valid}

    # 1. Root-position major or minor triad anchored on the lowest note
    if (lowest_pc + 4) % 12 in pcs and (lowest_pc + 7) % 12 in pcs:
        return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="major")
    if (lowest_pc + 3) % 12 in pcs and (lowest_pc + 7) % 12 in pcs:
        return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="minor")

    # 2. Power chord (root + fifth, optional octave doublings)
    if (lowest_pc + 7) % 12 in pcs and pcs <= {lowest_pc, (lowest_pc + 7) % 12}:
        return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="power")

    # 3. Inverted major or minor triad (e.g. E3-G3-C4 = C major)
    for note in valid:
        pc = note % 12
        if (pc + 4) % 12 in pcs and (pc + 7) % 12 in pcs:
            return MidiChordAnalysis(root_midi=note, chord_notes=notes_tuple, chord_type="major")
        if (pc + 3) % 12 in pcs and (pc + 7) % 12 in pcs:
            return MidiChordAnalysis(root_midi=note, chord_notes=notes_tuple, chord_type="minor")

    # 4. Dyads (major third, minor third, fourth, etc.): lowest note is root
    if (lowest_pc + 4) % 12 in pcs:
        return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="major")
    if (lowest_pc + 3) % 12 in pcs:
        return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="minor")
    return MidiChordAnalysis(root_midi=lowest, chord_notes=notes_tuple, chord_type="power")


class MidiInputWorker:
    """Polls a MIDI input port off the audio thread and emits `MusicalEvent`s (§55, §150)."""

    def __init__(
        self,
        config: AnalysisConfig,
        sample_rate_hz: int = 48_000,
        *,
        port_name: str | None = None,
        port: MidiPort | None = None,
        frame_source: Callable[[], SampleFrame] = lambda: 0,
        on_event: Callable[[MusicalEvent], None] | None = None,
        port_factory: Callable[[str], MidiPort] = open_midi_input_port,
    ) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._port_name = port_name.strip() if port_name and port_name.strip() else None
        if self._port_name is not None and self._port_name.lower() == "none":
            self._port_name = None
        self._port: MidiPort | None = port
        if self._port is not None and self._port_name is None:
            self._port_name = getattr(port, "name", "MIDI Input")
        self._owns_port = port is None
        self._frame_source = frame_source
        self._on_event = on_event or (lambda event: None)
        self._port_factory = port_factory

        self._lock = threading.Lock()
        self._active_notes: dict[MidiNote, int] = {}
        self._current_root: MidiNote | None = None
        self._current_chord_notes: tuple[MidiNote, ...] = ()
        self._current_chord_type: MidiChordType | None = None
        self._playing_state: PlayingState = PlayingState.STOPPED
        self._intensity: float = 0.0
        self._reported_intensity: float = 0.0

        self._tempo = TempoTracker(config, sample_rate_hz)
        self._key = KeyEstimator(config, sample_rate_hz)
        self._last_key_reported: tuple[str | None, str | None, float, str] = (
            None,
            None,
            0.0,
            "low",
        )

        self._status: MidiStatus = (
            MidiStatus.LISTENING if self._port is not None else MidiStatus.DISABLED
        )
        self._messages_processed: int = 0
        self._notes_triggered: int = 0
        self._last_velocity: int = 0
        self._last_error: str | None = None

        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

    @property
    def port_name(self) -> str | None:
        return self._port_name

    @property
    def status(self) -> MidiStatus:
        return self._status

    @property
    def active_notes(self) -> tuple[MidiNote, ...]:
        with self._lock:
            return tuple(sorted(self._active_notes))

    @property
    def current_root(self) -> MidiNote | None:
        return self._current_root

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def key_estimate(self) -> KeyEstimate:
        return self._key.estimate

    def set_instrument(self, instrument: Instrument) -> None:
        """Update the active instrument (`guitar` or `bass`) for register classification."""
        if instrument == self._config.instrument:
            return
        self._config = self._config.model_copy(update={"instrument": instrument})
        self._key = KeyEstimator(self._config, self._sample_rate_hz)

    def set_manual_bpm(self, manual_bpm: float | None) -> None:
        """Set or clear a manual BPM override on the MIDI tempo tracker."""
        self._tempo.set_manual_bpm(manual_bpm)

    def set_manual_key(self, manual_key: str | None) -> KeyEstimate:
        """Set or clear a manual key override on the MIDI key estimator."""
        return self._key.set_manual_key(manual_key)

    def reset_context(self) -> None:
        """Clear accumulated MIDI notes, tempo history, and key histograms (Step 6.1, Step 6.3)."""
        frame = max(0, int(self._frame_source()))
        self._release_all_notes(frame)
        self._tempo.reset()
        self._key.reset()
        self._last_key_reported = (None, None, 0.0, "low")

    def stats(self) -> MidiStats:
        with self._lock:
            active = tuple(sorted(self._active_notes))
        return MidiStats(
            status=self._status,
            port_name=self._port_name,
            active_notes=active,
            messages_processed=self._messages_processed,
            notes_triggered=self._notes_triggered,
            last_velocity=self._last_velocity,
            last_error=self._last_error,
        )

    def start(self) -> None:
        """Open the configured MIDI input port (if any) and start the listener thread."""
        if self._thread is not None:
            raise RuntimeError("MidiInputWorker can only be started once")
        if self._port is None and self._port_name is None:
            self._status = MidiStatus.DISABLED
            return
        if self._port is None and self._port_name is not None:
            try:
                resolved = resolve_midi_input(self._port_name)
                if resolved is None:
                    self._status = MidiStatus.DISABLED
                    return
                self._port_name = resolved
                self._port = self._port_factory(resolved)
                self._owns_port = True
            except Exception as exc:
                self._status = MidiStatus.ERROR
                self._last_error = str(exc)
                logger.warning("MIDI input %r unavailable: %s", self._port_name, exc)
                return

        self._status = MidiStatus.LISTENING
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, name="jammate-midi")
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener thread, emit `SilenceEvent` for any held notes, and close the port."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                logger.error(
                    "The MIDI input worker did not stop within %.1f s.", STOP_TIMEOUT_SECONDS
                )
            self._thread = None
        frame = max(0, int(self._frame_source()))
        self._release_all_notes(frame)
        if self._port is not None and self._owns_port:
            with contextlib.suppress(Exception):
                self._port.close()
        if self._status == MidiStatus.LISTENING:
            self._status = MidiStatus.DISABLED

    def process_available(self, frame_index: SampleFrame | None = None) -> int:
        """Process all pending messages from the MIDI port without sleeping (for tests & worker)."""
        port = self._port
        if port is None or self._status in (
            MidiStatus.DISABLED,
            MidiStatus.DISCONNECTED,
            MidiStatus.ERROR,
        ):
            return 0
        frame = max(0, int(self._frame_source() if frame_index is None else frame_index))
        try:
            pending = list(port.iter_pending())
        except Exception as exc:
            self._status = MidiStatus.DISCONNECTED
            self._last_error = str(exc)
            logger.warning("MIDI input port %r disconnected: %s", self._port_name, exc)
            self._release_all_notes(frame)
            return 0

        processed = 0
        for msg in pending:
            self.process_message(msg, frame_index=frame)
            processed += 1
        return processed

    def process_message(self, msg: Any, *, frame_index: SampleFrame | None = None) -> None:
        """Handle a single MIDI message (`note_on`, `note_off`, or CC `120`/`123`)."""
        frame = max(0, int(self._frame_source() if frame_index is None else frame_index))
        msg_type = getattr(msg, "type", None)
        self._messages_processed += 1

        if msg_type == "note_on":
            note = int(getattr(msg, "note", -1))
            velocity = int(getattr(msg, "velocity", 0))
            if not LOWEST_MIDI_NOTE <= note <= HIGHEST_MIDI_NOTE:
                return
            if velocity > 0:
                self._handle_note_on(note, velocity, frame)
            else:
                self._handle_note_off(note, frame)
        elif msg_type == "note_off":
            note = int(getattr(msg, "note", -1))
            if LOWEST_MIDI_NOTE <= note <= HIGHEST_MIDI_NOTE:
                self._handle_note_off(note, frame)
        elif msg_type == "control_change":
            control = int(getattr(msg, "control", -1))
            if control in (ALL_SOUND_OFF_CC, ALL_NOTES_OFF_CC):
                self._release_all_notes(frame)

    def _handle_note_on(self, note: MidiNote, velocity: int, frame: SampleFrame) -> None:
        clamped_vel = min(127, max(1, velocity))
        with self._lock:
            self._active_notes[note] = clamped_vel
            active_snapshot = dict(self._active_notes)
        self._notes_triggered += 1
        self._last_velocity = clamped_vel

        ts = frames_to_seconds(frame, self._sample_rate_hz)
        strength = velocity_to_strength(clamped_vel)
        onset = OnsetEvent(frame_index=frame, timestamp_seconds=ts, strength=strength)

        analysis = analyze_midi_notes(active_snapshot.keys())
        assert analysis is not None
        pitch_event: PitchEvent | None = None
        if (
            analysis.root_midi != self._current_root
            or analysis.chord_notes != self._current_chord_notes
            or analysis.chord_type != self._current_chord_type
        ):
            self._current_root = analysis.root_midi
            self._current_chord_notes = analysis.chord_notes
            self._current_chord_type = analysis.chord_type
            pitch_event = PitchEvent(
                frame_index=frame,
                timestamp_seconds=ts,
                midi_note=analysis.root_midi,
                frequency_hz=midi_to_hz(analysis.root_midi),
                confidence=1.0,
                chord_notes=analysis.chord_notes,
                chord_type=analysis.chord_type,
            )

        playing_event: PlayingStateUpdate | None = None
        if self._playing_state != PlayingState.PLAYING:
            self._playing_state = PlayingState.PLAYING
            playing_event = PlayingStateUpdate(
                frame_index=frame,
                timestamp_seconds=ts,
                state=PlayingState.PLAYING,
            )

        mean_vel = sum(active_snapshot.values()) / len(active_snapshot) / 127.0
        poly_boost = min(1.0, (len(active_snapshot) - 1) * 0.12)
        target_intensity = min(1.0, max(0.05, 0.85 * mean_vel + poly_boost))
        self._intensity = target_intensity
        intensity_event: IntensityUpdate | None = None
        if abs(self._intensity - self._reported_intensity) >= self._config.intensity.report_step:
            self._reported_intensity = self._intensity
            intensity_event = IntensityUpdate(
                frame_index=frame,
                timestamp_seconds=ts,
                intensity=self._intensity,
            )

        tempo_event = self._tempo.update(onset, frame)
        self._key.observe_note(
            analysis.root_midi,
            duration_seconds=0.25,
            velocity=strength,
            chord_notes=analysis.chord_notes,
            chord_type=analysis.chord_type,
        )
        key_event = self._build_key_update(frame, ts, analysis.root_midi)

        for event in (onset, pitch_event, playing_event, intensity_event, tempo_event, key_event):
            if event is not None:
                self._on_event(event)

    def _handle_note_off(self, note: MidiNote, frame: SampleFrame) -> None:
        with self._lock:
            if note not in self._active_notes:
                return
            del self._active_notes[note]
            remaining = dict(self._active_notes)

        if not remaining:
            self._release_all_notes(frame)
            return

        analysis = analyze_midi_notes(remaining.keys())
        assert analysis is not None
        if (
            analysis.root_midi != self._current_root
            or analysis.chord_notes != self._current_chord_notes
            or analysis.chord_type != self._current_chord_type
        ):
            self._current_root = analysis.root_midi
            self._current_chord_notes = analysis.chord_notes
            self._current_chord_type = analysis.chord_type
            ts = frames_to_seconds(frame, self._sample_rate_hz)
            self._on_event(
                PitchEvent(
                    frame_index=frame,
                    timestamp_seconds=ts,
                    midi_note=analysis.root_midi,
                    frequency_hz=midi_to_hz(analysis.root_midi),
                    confidence=1.0,
                    chord_notes=analysis.chord_notes,
                    chord_type=analysis.chord_type,
                )
            )

    def _release_all_notes(self, frame: SampleFrame) -> None:
        with self._lock:
            had_notes = bool(self._active_notes) or self._current_root is not None
            self._active_notes.clear()
        was_playing = self._playing_state.user_is_playing
        self._current_root = None
        self._current_chord_notes = ()
        self._current_chord_type = None
        if not had_notes and not was_playing:
            return

        ts = frames_to_seconds(frame, self._sample_rate_hz)
        if had_notes:
            self._on_event(SilenceEvent(frame_index=frame, timestamp_seconds=ts))
        if self._playing_state != PlayingState.STOPPED:
            self._playing_state = PlayingState.STOPPED
            self._on_event(
                PlayingStateUpdate(
                    frame_index=frame,
                    timestamp_seconds=ts,
                    state=PlayingState.STOPPED,
                )
            )
        if self._reported_intensity > 0.0:
            self._intensity = 0.0
            self._reported_intensity = 0.0
            self._on_event(
                IntensityUpdate(
                    frame_index=frame,
                    timestamp_seconds=ts,
                    intensity=0.0,
                )
            )

    def _build_key_update(
        self,
        frame: SampleFrame,
        timestamp_seconds: float,
        root_midi: MidiNote,
    ) -> KeyUpdate | None:
        est = self._key.estimate
        register = classify_register(root_midi, self._config.instrument)
        prev_key, prev_mode, prev_conf, prev_reg = self._last_key_reported
        key_changed = (est.key_name != prev_key) or (est.mode != prev_mode)
        conf_changed = abs(est.confidence - prev_conf) >= self._config.key.report_confidence_step
        reg_changed = register != prev_reg
        if est.key_name is None and prev_key is None and not reg_changed:
            return None
        if key_changed or conf_changed or reg_changed:
            self._last_key_reported = (est.key_name, est.mode, est.confidence, register)
            return KeyUpdate(
                frame_index=frame,
                timestamp_seconds=timestamp_seconds,
                key_name=est.key_name,
                mode=est.mode,
                confidence=est.confidence,
                bass_register=register,
            )
        return None

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            try:
                self.process_available()
            except Exception as exc:
                self._status = MidiStatus.ERROR
                self._last_error = str(exc)
                logger.exception("MIDI input worker error")
            self._stop_requested.wait(POLL_SECONDS)

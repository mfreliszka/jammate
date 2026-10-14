"""The Musical State: what JamMate currently believes is happening musically (§34-§39).

Analysis reports events. `MusicalStateManager` folds them into one immutable `MusicalState`
snapshot. Drums, AI and the UI read snapshots: they never see raw audio, and they never keep
their own copy of the tempo or the note (§38).

Threads (§39):
- One thread writes. The analysis worker calls `apply` for each event, through `JamSession`. The
  manager refuses a second writer thread, because two writers would overwrite each other's
  updates.
- Any thread reads `snapshot`. Each update builds a new frozen `MusicalState` and then swaps the
  reference, a single assignment that is atomic under the GIL. A reader gets the old state or the
  new one, never a mix, and never waits. There are no locks, so the state can never hold up
  another thread. The audio callback doesn't read it at all.
"""

import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace

from jammate.config import MusicalStateConfig
from jammate.music.beat_clock import BeatClock, BeatPosition
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SampleFrame,
    SilenceEvent,
    TempoUpdate,
)


@dataclass(frozen=True, kw_only=True)
class MusicalState:
    """One immutable snapshot, following the §36 schema. `MusicalState()` is the state before any
    playing.

    - `frame_index`: the stream position (D5) the snapshot is current as of, which is the latest
      frame of the events applied so far.
    - `recent_notes`: the last notes, oldest first. A note enters when it starts. A note picked
      again without a gap appears once, because the stabilizer reports it once.
    - `bpm`, `tempo_confidence`, `tempo_locked`: the tempo tracker's latest report (§28, §29).
    - `beat`, `bar`, `beat_phase`: the BeatClock's position (Step 3.1), or None while unlocked.
    - `playing_state`: the finer state of §70. `user_is_playing` derives from it.
    """

    frame_index: SampleFrame = 0
    current_note: MidiNote | None = None
    chord_notes: tuple[MidiNote, ...] = ()
    chord_type: str | None = None
    recent_notes: tuple[MidiNote, ...] = ()
    bpm: float | None = None
    tempo_confidence: float = 0.0
    tempo_locked: bool = False
    beat: int | None = None
    bar: int | None = None
    beat_phase: float | None = None
    intensity: float = 0.0
    playing_state: PlayingState = PlayingState.STOPPED

    @property
    def user_is_playing(self) -> bool:
        return self.playing_state.user_is_playing


class MusicalStateManager:
    """The single owner of the Musical State (§38). See the module docstring for threading."""

    def __init__(self, config: MusicalStateConfig, *, manual_bpm: float | None = None) -> None:
        self._recent_note_count = config.recent_note_count
        if manual_bpm is not None:
            self._snapshot = MusicalState(
                bpm=float(manual_bpm),
                tempo_confidence=1.0,
                tempo_locked=True,
                bar=1,
                beat=1,
                beat_phase=0.0,
            )
        else:
            self._snapshot = MusicalState()
        self._writer_thread_id: int | None = None

    @property
    def snapshot(self) -> MusicalState:
        """The current state. Reading it never blocks, and the object never changes."""
        return self._snapshot

    def apply(self, event: MusicalEvent) -> MusicalState:
        """Fold one analysis event into the state and publish the result, which is also
        returned. Only the writer thread may call this: the first thread that does."""
        self._check_writer()
        state = self._snapshot
        frame_index = max(state.frame_index, event.frame_index)
        match event:
            case PitchEvent(midi_note=note, chord_notes=chord_notes, chord_type=chord_type):
                recent = (*state.recent_notes, note)[-self._recent_note_count :]
                state = replace(
                    state,
                    frame_index=frame_index,
                    current_note=note,
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                    recent_notes=recent,
                )
            case SilenceEvent():
                state = replace(
                    state,
                    frame_index=frame_index,
                    current_note=None,
                    chord_notes=(),
                    chord_type=None,
                )
            case TempoUpdate(bpm=bpm, confidence=confidence, locked=locked):
                state = replace(
                    state,
                    frame_index=frame_index,
                    bpm=bpm,
                    tempo_confidence=confidence,
                    tempo_locked=locked,
                )
            case IntensityUpdate(intensity=intensity):
                state = replace(state, frame_index=frame_index, intensity=intensity)
            case PlayingStateUpdate(state=playing_state):
                state = replace(state, frame_index=frame_index, playing_state=playing_state)
            case OnsetEvent():
                return state  # no field of the §36 schema; tempo already reflects onsets
        self._snapshot = state
        return state

    def update_beat(self, frame_index: SampleFrame, position: BeatPosition | None) -> MusicalState:
        """Publish the BeatClock's `(bar, beat, beat_phase)` at `frame_index` (or clear them when
        `position` is None). Must be called from the single writer thread (§38)."""
        self._check_writer()
        state = self._snapshot
        if position is None:
            if state.beat is None and state.bar is None and state.beat_phase is None:
                return state
            state = replace(
                state,
                frame_index=max(state.frame_index, frame_index),
                beat=None,
                bar=None,
                beat_phase=None,
            )
        else:
            state = replace(
                state,
                frame_index=max(state.frame_index, frame_index),
                beat=position.beat,
                bar=position.bar,
                beat_phase=position.beat_phase,
            )
        self._snapshot = state
        return state

    def _check_writer(self) -> None:
        thread_id = threading.get_ident()
        if self._writer_thread_id is None:
            self._writer_thread_id = thread_id
        elif thread_id != self._writer_thread_id:
            raise RuntimeError(
                "MusicalStateManager has a single writer thread (§38); "
                f"{threading.current_thread().name!r} is not it."
            )


def replay(
    events: Iterable[MusicalEvent],
    config: MusicalStateConfig | None = None,
    *,
    beat_clock: BeatClock | None = None,
) -> MusicalState:
    """The state after applying `events` in order: for offline tools and tests."""
    manager = MusicalStateManager(config or MusicalStateConfig())
    for event in events:
        manager.apply(event)
        if beat_clock is not None:
            beat_clock.handle_event(event)
            manager.update_beat(event.frame_index, beat_clock.position_at(event.frame_index))
    return manager.snapshot

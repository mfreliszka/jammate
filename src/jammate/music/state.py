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
from typing import Literal

from jammate.analysis.key import BassRegister, KeyMode, classify_register, parse_key_string
from jammate.config import Instrument, MusicalStateConfig
from jammate.music.beat_clock import BeatClock, BeatPosition
from jammate.music.events import (
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
    TempoUpdate,
)

type SectionType = Literal["listening", "intro", "groove", "variation", "fill", "breakdown"]


@dataclass(frozen=True, kw_only=True)
class MusicalState:
    """One immutable snapshot, following the §36 / §40 / §144 schema. `MusicalState()` is the
    state before any playing.

    - `frame_index`: the stream position (D5) the snapshot is current as of, which is the latest
      frame of the events applied so far.
    - `recent_notes`: the last notes, oldest first. A note enters when it starts. A note picked
      again without a gap appears once, because the stabilizer reports it once.
    - `bpm`, `tempo_confidence`, `tempo_locked`: the tempo tracker's latest report (§28, §29).
    - `beat`, `bar`, `beat_phase`: the BeatClock's position (Step 3.1), or None while unlocked.
    - `playing_state`: the finer state of §70. `user_is_playing` derives from it.
    - `estimated_key`, `estimated_mode`, `harmonic_confidence`, `bass_register`: tonal-center and
      register estimates (Step 5.2, §40, §90, §145).
    - `riff_id`, `riff_iteration`, `riff_changed`, `section`: riff recognition and arrangement
      section state (Steps 5.3 & 5.5, §40, §41, §144, §146).
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
    estimated_key: str | None = None
    estimated_mode: KeyMode | None = None
    harmonic_confidence: float = 0.0
    bass_register: BassRegister = "low"
    riff_id: str | None = None
    riff_iteration: int = 0
    riff_changed: bool = False
    section: SectionType = "listening"

    @property
    def user_is_playing(self) -> bool:
        return self.playing_state.user_is_playing

    @property
    def key_label(self) -> str | None:
        if self.estimated_key is None:
            return None
        if self.estimated_mode is None:
            return self.estimated_key
        return f"{self.estimated_key} {self.estimated_mode}"


class MusicalStateManager:
    """The single owner of the Musical State (§38). See the module docstring for threading."""

    def __init__(
        self,
        config: MusicalStateConfig,
        *,
        manual_bpm: float | None = None,
        manual_key: str | None = None,
    ) -> None:
        self._recent_note_count = config.recent_note_count
        self._instrument: Instrument = config.instrument
        self._manual_bpm: float | None = float(manual_bpm) if manual_bpm is not None else None
        self._manual_key: tuple[str, KeyMode] | None = None
        if manual_key is not None:
            _, key_name, mode = parse_key_string(manual_key)
            self._manual_key = (key_name, mode)
        self._snapshot = self._build_initial_state()
        self._writer_thread_id: int | None = None

    def _build_initial_state(
        self,
        *,
        frame_index: SampleFrame = 0,
        position: BeatPosition | None = None,
    ) -> MusicalState:
        bpm = self._manual_bpm
        tempo_locked = bpm is not None
        tempo_conf = 1.0 if tempo_locked else 0.0
        bar = position.bar if position is not None else (1 if tempo_locked else None)
        beat = position.beat if position is not None else (1 if tempo_locked else None)
        beat_phase = (
            position.beat_phase if position is not None else (0.0 if tempo_locked else None)
        )
        est_key = self._manual_key[0] if self._manual_key is not None else None
        est_mode = self._manual_key[1] if self._manual_key is not None else None
        harm_conf = 1.0 if self._manual_key is not None else 0.0
        return MusicalState(
            frame_index=frame_index,
            bpm=bpm,
            tempo_confidence=tempo_conf,
            tempo_locked=tempo_locked,
            bar=bar,
            beat=beat,
            beat_phase=beat_phase,
            estimated_key=est_key,
            estimated_mode=est_mode,
            harmonic_confidence=harm_conf,
        )

    def set_instrument(self, instrument: Instrument) -> None:
        """Update the instrument used for `bass_register` classification."""
        self._instrument = instrument

    def set_manual_bpm(
        self,
        manual_bpm: float | None,
        *,
        position: BeatPosition | None = None,
    ) -> MusicalState:
        """Set or clear a manual BPM override and publish a new snapshot (Step 6.1, §31, §149)."""
        self._manual_bpm = float(manual_bpm) if manual_bpm is not None else None
        state = self._snapshot
        if self._manual_bpm is not None:
            bar = position.bar if position is not None else (state.bar or 1)
            beat = position.beat if position is not None else (state.beat or 1)
            phase = (
                position.beat_phase
                if position is not None
                else (0.0 if state.beat_phase is None else state.beat_phase)
            )
            state = replace(
                state,
                bpm=self._manual_bpm,
                tempo_confidence=1.0,
                tempo_locked=True,
                bar=bar,
                beat=beat,
                beat_phase=phase,
            )
        else:
            state = replace(
                state,
                bpm=None,
                tempo_confidence=0.0,
                tempo_locked=False,
                bar=position.bar if position is not None else None,
                beat=position.beat if position is not None else None,
                beat_phase=position.beat_phase if position is not None else None,
            )
        self._snapshot = state
        return state

    def set_manual_key(self, manual_key: str | None) -> MusicalState:
        """Set or clear a manual key override and publish a new snapshot (Step 6.1, §149)."""
        state = self._snapshot
        if manual_key is None:
            self._manual_key = None
            state = replace(
                state,
                estimated_key=None,
                estimated_mode=None,
                harmonic_confidence=0.0,
            )
        else:
            _, key_name, mode = parse_key_string(manual_key)
            self._manual_key = (key_name, mode)
            state = replace(
                state,
                estimated_key=key_name,
                estimated_mode=mode,
                harmonic_confidence=1.0,
            )
        self._snapshot = state
        return state

    def reset(self, *, position: BeatPosition | None = None) -> MusicalState:
        """Reset musical context while preserving frame_index and manual overrides (Step 6.1)."""
        state = self._build_initial_state(
            frame_index=self._snapshot.frame_index,
            position=position,
        )
        self._snapshot = state
        return state

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
                reg = classify_register(note, self._instrument)
                state = replace(
                    state,
                    frame_index=frame_index,
                    current_note=note,
                    chord_notes=chord_notes,
                    chord_type=chord_type,
                    recent_notes=recent,
                    bass_register=reg,
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
                if self._manual_bpm is not None:
                    state = replace(
                        state,
                        frame_index=frame_index,
                        bpm=self._manual_bpm,
                        tempo_confidence=1.0,
                        tempo_locked=True,
                    )
                else:
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
            case KeyUpdate(
                key_name=key_name,
                mode=mode,
                confidence=confidence,
                bass_register=bass_register,
            ):
                if bass_register in ("low", "mid", "high"):
                    typed_reg: BassRegister = bass_register
                else:
                    typed_reg = state.bass_register
                if self._manual_key is not None:
                    state = replace(
                        state,
                        frame_index=frame_index,
                        estimated_key=self._manual_key[0],
                        estimated_mode=self._manual_key[1],
                        harmonic_confidence=1.0,
                        bass_register=typed_reg,
                    )
                else:
                    typed_mode: KeyMode | None = mode if mode in ("minor", "major") else None
                    state = replace(
                        state,
                        frame_index=frame_index,
                        estimated_key=key_name,
                        estimated_mode=typed_mode,
                        harmonic_confidence=confidence,
                        bass_register=typed_reg,
                    )
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

    def update_riff(
        self,
        frame_index: SampleFrame,
        *,
        riff_id: str | None,
        riff_iteration: int,
        riff_changed: bool,
        section: SectionType | None = None,
    ) -> MusicalState:
        """Publish the RiffDetector's `(riff_id, riff_iteration, riff_changed)` at `frame_index`
        (Step 5.3, §144, §146). Must be called from the single writer thread (§38)."""
        self._check_writer()
        state = self._snapshot
        next_section = section if section is not None else state.section
        state = replace(
            state,
            frame_index=max(state.frame_index, frame_index),
            riff_id=riff_id,
            riff_iteration=riff_iteration,
            riff_changed=riff_changed,
            section=next_section,
        )
        self._snapshot = state
        return state

    def update_section(self, frame_index: SampleFrame, section: SectionType) -> MusicalState:
        """Publish the active arrangement `section` at `frame_index` (Steps 5.3 & 5.5, §144)."""
        self._check_writer()
        state = self._snapshot
        if state.section == section and state.frame_index >= frame_index:
            return state
        state = replace(
            state,
            frame_index=max(state.frame_index, frame_index),
            section=section,
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

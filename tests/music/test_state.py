import dataclasses
import threading

import pytest

from jammate.analysis.notes import midi_to_hz
from jammate.config import MusicalStateConfig
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SampleFrame,
    SilenceEvent,
    TempoUpdate,
)
from jammate.music.state import MusicalState, MusicalStateManager, replay

SR = 48_000
CONFIG = MusicalStateConfig()


def _at(frame_index: SampleFrame) -> dict[str, float]:
    return {"frame_index": frame_index, "timestamp_seconds": frame_index / SR}


def _note(midi_note: MidiNote, frame_index: SampleFrame) -> PitchEvent:
    return PitchEvent(
        **_at(frame_index), midi_note=midi_note, frequency_hz=midi_to_hz(midi_note), confidence=0.9
    )


def _tempo(frame_index: SampleFrame, bpm: float, confidence: float, locked: bool) -> TempoUpdate:
    return TempoUpdate(**_at(frame_index), bpm=bpm, confidence=confidence, locked=locked)


def test_the_initial_state_knows_nothing_yet() -> None:
    state = MusicalStateManager(CONFIG).snapshot

    assert state == MusicalState()
    assert state.frame_index == 0
    assert state.current_note is None
    assert state.recent_notes == ()
    assert (state.bpm, state.tempo_confidence, state.tempo_locked) == (None, 0.0, False)
    assert (state.beat, state.bar, state.beat_phase) == (None, None, None)  # until Phase 3
    assert state.intensity == 0.0
    assert state.playing_state == PlayingState.STOPPED
    assert not state.user_is_playing


def test_notes_and_silences_set_the_current_and_recent_notes() -> None:
    manager = MusicalStateManager(CONFIG)

    manager.apply(_note(40, 4_800))
    manager.apply(_note(43, 9_600))
    assert manager.snapshot.current_note == 43
    manager.apply(SilenceEvent(**_at(12_000)))
    state = manager.apply(_note(40, 14_400))  # the same note again after a gap

    assert state.current_note == 40
    assert state.recent_notes == (40, 43, 40)
    assert state.frame_index == 14_400


def test_recent_notes_keep_only_the_last_few() -> None:
    manager = MusicalStateManager(MusicalStateConfig(recent_note_count=3))

    for index, note in enumerate([40, 43, 45, 47, 48]):
        manager.apply(_note(note, 4_800 * (index + 1)))

    assert manager.snapshot.recent_notes == (45, 47, 48)


def test_each_update_changes_only_its_own_fields() -> None:
    manager = MusicalStateManager(CONFIG)
    before = manager.apply(_note(40, 4_800))

    tempo = manager.apply(_tempo(9_600, 92.4, 0.7, locked=True))
    assert tempo == dataclasses.replace(
        before, frame_index=9_600, bpm=92.4, tempo_confidence=0.7, tempo_locked=True
    )
    intensity = manager.apply(IntensityUpdate(**_at(10_080), intensity=0.65))
    assert intensity == dataclasses.replace(tempo, frame_index=10_080, intensity=0.65)
    playing = manager.apply(PlayingStateUpdate(**_at(10_560), state=PlayingState.SUSTAINING))
    assert playing == dataclasses.replace(
        intensity, frame_index=10_560, playing_state=PlayingState.SUSTAINING
    )
    assert playing.user_is_playing


@pytest.mark.parametrize(
    ("playing_state", "user_is_playing"),
    [
        (PlayingState.PLAYING, True),
        (PlayingState.SUSTAINING, True),
        (PlayingState.STOPPED, False),
        (PlayingState.TOO_QUIET, False),
        (PlayingState.NOISE, False),
    ],
)
def test_user_is_playing_follows_the_playing_state(
    playing_state: PlayingState, user_is_playing: bool
) -> None:
    state = replay([PlayingStateUpdate(**_at(480), state=playing_state)])

    assert state.user_is_playing is user_is_playing


def test_an_unlocked_tempo_estimate_is_kept_as_unlocked() -> None:
    state = replay([_tempo(480, 101.5, 0.2, locked=False)])

    assert (state.bpm, state.tempo_confidence, state.tempo_locked) == (101.5, 0.2, False)


def test_onsets_leave_the_state_unchanged() -> None:
    manager = MusicalStateManager(CONFIG)
    before = manager.apply(_note(40, 4_800))

    after = manager.apply(OnsetEvent(**_at(9_600), strength=0.5))

    assert after is before
    assert manager.snapshot is before


def test_frame_index_never_goes_back() -> None:
    # A note is reported about 70 ms after it starts, so its frame can be older than the hop of
    # an intensity update applied just before it.
    manager = MusicalStateManager(CONFIG)
    manager.apply(IntensityUpdate(**_at(9_600), intensity=0.4))

    state = manager.apply(_note(40, 6_240))

    assert state.frame_index == 9_600
    assert state.current_note == 40


def test_snapshots_are_immutable_and_old_ones_stay_valid() -> None:
    manager = MusicalStateManager(CONFIG)
    old = manager.apply(_note(40, 4_800))

    manager.apply(_note(43, 9_600))

    assert old.current_note == 40
    assert old.recent_notes == (40,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        old.current_note = 45  # type: ignore[misc]


def test_only_the_first_writer_thread_may_apply() -> None:
    manager = MusicalStateManager(CONFIG)
    manager.apply(_note(40, 4_800))  # this thread is now the writer
    errors: list[BaseException] = []

    def write_from_another_thread() -> None:
        try:
            manager.apply(_note(43, 9_600))
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=write_from_another_thread, name="intruder")
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert "single writer" in str(errors[0])
    assert manager.snapshot.current_note == 40


def test_readers_on_other_threads_always_see_whole_snapshots() -> None:
    manager = MusicalStateManager(CONFIG)
    update_count = 20_000
    done = threading.Event()

    def write() -> None:
        # The note and the frame always match: note = 40 + (frame / 480) % 12.
        for index in range(1, update_count + 1):
            manager.apply(_note(40 + index % 12, index * 480))
        done.set()

    writer = threading.Thread(target=write, name="writer")
    writer.start()
    torn = 0
    seen = set()
    while not done.is_set():
        state = manager.snapshot
        if state.current_note is not None:
            seen.add(state.frame_index)
            expected_note = 40 + (state.frame_index // 480) % 12
            if state.current_note != expected_note or state.recent_notes[-1] != expected_note:
                torn += 1
    writer.join()

    assert torn == 0
    assert len(seen) > 1  # the reader really ran alongside the writer
    assert manager.snapshot.frame_index == update_count * 480


def test_replay_folds_events_in_order() -> None:
    events = [
        _note(40, 4_800),
        OnsetEvent(**_at(9_000), strength=0.5),
        _note(43, 9_600),
        _tempo(10_080, 120.0, 0.9, locked=True),
        SilenceEvent(**_at(14_400)),
    ]

    state = replay(events, MusicalStateConfig(recent_note_count=4))

    assert state.current_note is None
    assert state.recent_notes == (40, 43)
    assert state.bpm == 120.0 and state.tempo_locked
    assert state.frame_index == 14_400


def test_update_beat_sets_and_clears_beat_bar_and_phase() -> None:
    from jammate.music.beat_clock import BeatPosition

    manager = MusicalStateManager(CONFIG)
    pos = BeatPosition(bar=3, beat=2, beat_phase=0.45, total_beats=9.45)

    state = manager.update_beat(48_000, pos)
    assert (state.bar, state.beat, state.beat_phase) == (3, 2, 0.45)
    assert state.frame_index == 48_000

    cleared = manager.update_beat(96_000, None)
    assert (cleared.bar, cleared.beat, cleared.beat_phase) == (None, None, None)
    assert cleared.frame_index == 96_000
    assert manager.update_beat(96_000, None) is cleared

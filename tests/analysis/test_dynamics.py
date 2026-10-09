import itertools

import pytest

from jammate.analysis.dynamics import IntensityTracker, PlayingStateTracker
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.worker import analyze_offline
from jammate.config import AnalysisConfig, PlayingStateConfig
from jammate.music.events import IntensityUpdate, OnsetEvent, PlayingState, PlayingStateUpdate
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, Riff, riff

SR = 48_000
CONFIG = AnalysisConfig(instrument="guitar")
HOP = CONFIG.hop_frames


def _guitar_riff(amplitude: float, step_beats: float = 0.5) -> Riff:
    return riff(
        ["E2", "G2", "A2", "E2"],
        bpm=120,
        step_beats=step_beats,
        repeats=6,
        amplitude=amplitude,
        tail_seconds=4.0,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )


def _max_intensity(amplitude: float, step_beats: float = 0.5) -> float:
    events, _ = analyze_offline(_guitar_riff(amplitude, step_beats).samples, CONFIG, SR)
    return max(e.event.intensity for e in events if isinstance(e.event, IntensityUpdate))


# --- intensity (end to end) ----------------------------------------------------------------------


def test_intensity_rises_monotonically_with_playing_level() -> None:
    levels = [_max_intensity(amplitude) for amplitude in (0.03, 0.1, 0.3, 0.9)]

    assert levels == sorted(levels)
    assert levels[0] < 0.4 < levels[-1]
    assert all(0.0 <= level <= 1.0 for level in levels)


def test_intensity_rises_with_note_density() -> None:
    assert _max_intensity(0.3, step_beats=0.25) > _max_intensity(0.3, step_beats=0.5) + 0.05


def test_intensity_falls_back_to_zero_after_the_player_stops() -> None:
    events, stats = analyze_offline(_guitar_riff(0.5).samples, CONFIG, SR)
    updates = [e.event for e in events if isinstance(e.event, IntensityUpdate)]

    assert stats.intensity < 0.05
    assert updates[-1].intensity < 0.1
    # Updates only when the value moved at least report_step (0.05).
    steps = [abs(b.intensity - a.intensity) for a, b in itertools.pairwise(updates)]
    assert min(steps) >= 0.05 - 1e-9


def test_intensity_is_zero_while_not_playing() -> None:
    tracker = IntensityTracker(CONFIG, SR)
    frame = PitchFrame(HOP, None, 0.0, -10.0, -5.0, -10.0)  # loud, but not playing

    for _ in range(100):
        assert tracker.update(frame, None, user_is_playing=False) is None
    assert tracker.value == 0.0


# --- playing state (hand-made hops) ---------------------------------------------------------------


class Hops:
    """Feeds hand-made hops to a PlayingStateTracker and collects the states it reports."""

    def __init__(self, config: AnalysisConfig = CONFIG) -> None:
        self.tracker = PlayingStateTracker(config, SR)
        self.frame_index = 0
        self.states: list[PlayingState] = []

    def add(
        self,
        hops: int,
        rms_dbfs: float,
        *,
        note: int | None = None,
        attack: bool = False,
    ) -> PlayingState:
        for hop in range(hops):
            self.frame_index += HOP
            frame = PitchFrame(self.frame_index, None, 0.0, rms_dbfs, rms_dbfs + 6, rms_dbfs)
            onset = None
            if attack and hop == 0:
                onset = OnsetEvent(
                    frame_index=self.frame_index - HOP,
                    timestamp_seconds=(self.frame_index - HOP) / SR,
                    strength=0.5,
                )
            update = self.tracker.update(frame, onset, note)
            if update is not None:
                assert isinstance(update, PlayingStateUpdate)
                self.states.append(update.state)
        return self.tracker.state


def test_starts_stopped_and_stays_stopped_in_silence() -> None:
    hops = Hops()

    assert hops.add(300, -70.0) == PlayingState.STOPPED
    assert hops.states == []


def test_an_attack_is_playing_then_a_ringing_note_is_sustaining() -> None:
    hops = Hops()

    assert hops.add(50, -12.0, note=40, attack=True) == PlayingState.PLAYING  # 0.5 s
    assert hops.add(100, -15.0, note=40) == PlayingState.SUSTAINING  # after the 1 s hold
    assert hops.add(10, -60.0) == PlayingState.STOPPED  # note ended, below the gate
    assert hops.states == [PlayingState.PLAYING, PlayingState.SUSTAINING, PlayingState.STOPPED]


def test_continuous_attacks_stay_playing_through_short_gaps() -> None:
    hops = Hops()

    for _ in range(10):
        hops.add(20, -12.0, note=40, attack=True)
        hops.add(5, -60.0)  # staccato gap

    assert hops.states == [PlayingState.PLAYING]


def test_an_unpitched_chord_ringing_after_its_attack_is_sustaining() -> None:
    hops = Hops()

    hops.add(10, -10.0, attack=True)  # a strum: no pitch
    assert hops.add(120, -20.0) == PlayingState.SUSTAINING  # rings past the hold


def test_attacks_below_the_level_gate_are_too_quiet() -> None:
    hops = Hops()

    assert hops.add(20, -48.0, attack=True) == PlayingState.TOO_QUIET
    assert not PlayingState.TOO_QUIET.user_is_playing


def test_steady_sound_without_attacks_is_noise() -> None:
    hops = Hops()

    assert hops.add(150, -35.0) == PlayingState.STOPPED  # not yet 2 s
    assert hops.add(100, -35.0) == PlayingState.NOISE
    assert hops.add(10, -60.0) == PlayingState.STOPPED


def test_varying_sound_without_attacks_is_not_noise() -> None:
    hops = Hops()

    for _ in range(15):
        hops.add(10, -30.0)
        hops.add(10, -42.0)

    assert hops.states == []  # stayed STOPPED: neither attacked nor steady


def test_gate_hysteresis_stops_flicker_around_the_gate() -> None:
    config = CONFIG.model_copy(update={"playing": PlayingStateConfig(hold_seconds=0.1)})
    hops = Hops(config)

    hops.add(5, -40.0, attack=True)
    for _ in range(20):  # a decaying chord hovering around -45 dBFS
        hops.add(3, -44.0)
        hops.add(3, -46.5)

    assert hops.states == [PlayingState.PLAYING, PlayingState.SUSTAINING]


@pytest.mark.parametrize(
    ("state", "playing"),
    [
        (PlayingState.PLAYING, True),
        (PlayingState.SUSTAINING, True),
        (PlayingState.STOPPED, False),
        (PlayingState.TOO_QUIET, False),
        (PlayingState.NOISE, False),
    ],
)
def test_only_playing_and_sustaining_count_as_playing(state: PlayingState, playing: bool) -> None:
    assert state.user_is_playing is playing
    update = PlayingStateUpdate(frame_index=0, timestamp_seconds=0.0, state=state)
    assert update.user_is_playing is playing

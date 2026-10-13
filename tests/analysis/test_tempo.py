import itertools
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.analysis.tempo import (
    TempoTracker,
    estimate_tempo,
    grid_fit,
    mean_resultants,
)
from jammate.analysis.worker import analyze_offline
from jammate.config import AnalysisConfig, TempoConfig
from jammate.music.events import OnsetEvent, TempoUpdate, frames_to_seconds, seconds_to_frames
from jammate.music.state import replay
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, onset_times, riff

SR = 48_000
CONFIG = AnalysisConfig(instrument="guitar")
TEMPO = CONFIG.tempo
HOP = CONFIG.hop_frames
DATA = Path(__file__).resolve().parents[1] / "data"


def _track(
    times: np.ndarray, config: AnalysisConfig = CONFIG
) -> tuple[TempoTracker, list[TempoUpdate]]:
    """Feed onsets at `times` (seconds) to a tracker, each reported one hop after its attack, as
    the onset detector does. Returns the tracker and its updates."""
    tracker = TempoTracker(config, SR)
    updates = []
    for time_seconds in times:
        frame = seconds_to_frames(time_seconds, SR)
        onset = OnsetEvent(
            frame_index=frame, timestamp_seconds=frames_to_seconds(frame, SR), strength=0.5
        )
        update = tracker.update(onset, frame + HOP)
        if update is not None:
            updates.append(update)
    return tracker, updates


def _first_lock(updates: list[TempoUpdate]) -> TempoUpdate:
    return next(update for update in updates if update.locked)


def _unlocks(updates: list[TempoUpdate]) -> int:
    return sum(a.locked and not b.locked for a, b in itertools.pairwise(updates))


# --- grid alignment and fit ----------------------------------------------------------------------


def test_alignment_is_one_on_the_grid_and_its_angle_is_the_phase() -> None:
    onsets = 0.1 + 0.5 * np.arange(12)  # a 0.5 s grid, phase 0.1 s
    periods = np.array([0.5, 0.25, 1.0, 0.75])

    resultants = mean_resultants(onsets, np.ones(12), periods)

    assert abs(resultants[0]) == pytest.approx(1.0)
    assert np.angle(resultants[0]) / (2 * np.pi) * 0.5 == pytest.approx(0.1)
    assert abs(resultants[1]) == pytest.approx(1.0)  # a subdivision fits too
    assert abs(resultants[2]) == pytest.approx(0.0, abs=1e-9)  # every other onset is off-grid
    assert abs(resultants[3]) == pytest.approx(0.0, abs=1e-9)  # 3:2 against the grid


def test_fit_is_high_on_the_grid_and_near_zero_for_random_onsets() -> None:
    weights = np.ones(40)
    periods = np.array([0.5, 0.25])
    on_grid = 0.25 * np.arange(40)
    scattered = np.sort(np.random.default_rng(3).uniform(0.0, 10.0, 40))

    def fit(onsets: np.ndarray) -> np.ndarray:
        return grid_fit(onsets, weights, periods, mean_resultants(onsets, weights, periods), 0.025)

    # 1 minus the share that chance gives (a 50 ms window of a 250 ms grid).
    assert fit(on_grid)[1] == pytest.approx(0.75, abs=0.02)
    # Half the onsets on each of two phases: the mean phase lies between them, near neither.
    assert fit(on_grid)[0] < 0.1
    assert np.all(fit(scattered) < 0.15)


def test_fit_uses_an_absolute_timing_tolerance() -> None:
    # Two of every three onsets 77 ms off a 0.62 s grid: the cosine alignment still counts them
    # (a 3+3+2 riff against three beats per bar), the fit does not.
    beats = 0.615 * np.arange(12)
    onsets = beats + np.tile([0.0, -0.077, 0.077], 4)
    weights = np.ones(12)
    periods = np.array([0.615])
    resultants = mean_resultants(onsets, weights, periods)

    assert abs(resultants[0]) > 0.7
    assert grid_fit(onsets, weights, periods, resultants, 0.025)[0] < 0.35


def test_grids_fine_enough_to_be_hit_by_chance_score_little() -> None:
    onsets = 0.05 * np.arange(40)
    periods = np.array([0.05, 0.5])
    weights = np.ones(40)
    fit = grid_fit(onsets, weights, periods, mean_resultants(onsets, weights, periods), 0.025)

    assert fit[0] < 0.2  # every onset on the grid, but a 50 ms grid is nearly all "near"
    assert fit[1] < 0.05  # only every 10th onset on the grid


# --- estimate_tempo ------------------------------------------------------------------------------


def test_eighth_notes_read_as_the_beat_in_the_reference_octave() -> None:
    eighths = onset_times(92, "x" * 8, bars=6)

    preferred = estimate_tempo(eighths, TEMPO.preferred_bpm, TEMPO)
    near_184 = estimate_tempo(eighths, 184.0, TEMPO)

    assert preferred is not None and near_184 is not None
    assert preferred.bpm == pytest.approx(92.0, abs=0.1)  # not 184 (D9)
    assert near_184.bpm == pytest.approx(184.0, abs=0.2)
    assert preferred.clarity == pytest.approx(1.0, abs=0.01)


def test_the_octave_stays_inside_the_tempo_range() -> None:
    eighths = onset_times(92, "x" * 8, bars=6)

    estimate = estimate_tempo(eighths, 1000.0, TEMPO)

    assert estimate is not None
    assert estimate.bpm == pytest.approx(184.0, abs=0.2)  # the octave 100-200, not 368


@pytest.mark.parametrize(
    ("bpm", "pattern", "steps_per_beat", "jitter_seconds"),
    [
        (92, "xxxxxxxx", 2, 0.02),
        (120, "xxxxxxxx", 2, 0.02),
        (140, "xxxxxxxx", 2, 0.02),
        (100, "xxxx", 1, 0.02),  # quarter notes
        (100, "x" * 16, 4, 0.015),  # 16ths
        (150, "x" * 16, 4, 0.01),  # 16ths at 150 read as 75: outside the preferred octave
        (100, "x.xx.x.x", 2, 0.02),  # syncopated 8ths
        (100, "x..x..x.", 2, 0.02),  # 3+3+2
        (130, "x..x..x.", 2, 0.02),  # 3+3+2, three-beats-per-bar grid inside the octave
        (92, "x..x..x...x.x...", 4, 0.012),  # syncopated 16ths
        (120, "x.xxx.xxx.xxx.xx", 4, 0.012),  # gallop
        (92, "x.x.", 1, 0.02),  # half notes
    ],
)
def test_synthetic_rhythms_give_their_tempo(
    bpm: float, pattern: str, steps_per_beat: int, jitter_seconds: float
) -> None:
    onsets = onset_times(
        bpm, pattern, bars=12, steps_per_beat=steps_per_beat, jitter_seconds=jitter_seconds, seed=1
    )

    estimate = estimate_tempo(onsets, TEMPO.preferred_bpm, TEMPO)

    assert estimate is not None
    expected = bpm if bpm < 141 else bpm / 2
    assert estimate.bpm == pytest.approx(expected, rel=0.01)


def test_missing_notes_and_jitter_barely_move_the_estimate() -> None:
    onsets = onset_times(92, "x" * 8, bars=8, jitter_seconds=0.02, miss_ratio=0.3, seed=2)

    estimate = estimate_tempo(onsets, TEMPO.preferred_bpm, TEMPO)

    assert estimate is not None
    assert estimate.bpm == pytest.approx(92.0, abs=1.0)
    assert estimate.clarity > 0.8


def test_too_few_or_too_sparse_onsets_give_no_estimate() -> None:
    assert estimate_tempo(np.array([0.0, 0.5, 1.0]), 100.0, TEMPO) is None
    assert estimate_tempo(np.arange(8) * 2.5, 100.0, TEMPO) is None  # whole notes at 96 BPM


def test_significance_grows_with_the_onsets_behind_an_estimate() -> None:
    onsets = onset_times(120, "x" * 8, bars=5, jitter_seconds=0.015, seed=3)

    early = estimate_tempo(onsets[:6], 100.0, TEMPO)
    later = estimate_tempo(onsets, 100.0, TEMPO)

    assert early is not None and later is not None
    assert later.significance > 3 * early.significance


# --- TempoTracker: locking -----------------------------------------------------------------------


def test_a_steady_riff_locks_within_a_few_seconds_and_does_not_jump() -> None:
    onsets = onset_times(92, "x" * 8, bars=12, jitter_seconds=0.02, start_seconds=1.0, seed=1)

    tracker, updates = _track(onsets)

    lock = _first_lock(updates)
    assert 1.5 <= lock.timestamp_seconds - onsets[0] <= 4.5
    locked = [update.bpm for update in updates if update.locked]
    assert all(bpm is not None and abs(bpm - 92.0) < 1.0 for bpm in locked)
    assert _unlocks(updates) == 0
    assert tracker.state.locked
    assert tracker.state.confidence > 0.8


def test_estimates_before_the_lock_are_reported_as_unlocked() -> None:
    onsets = onset_times(120, "x" * 8, bars=2, start_seconds=1.0)

    _, updates = _track(onsets[:6])

    assert updates
    assert not any(update.locked for update in updates)


def test_one_unusual_note_does_not_move_the_tempo() -> None:
    steady = onset_times(100, "x" * 8, bars=8, jitter_seconds=0.01, start_seconds=1.0, seed=4)
    stray = np.sort(np.append(steady, steady[40] + 0.11))  # an extra note between two 8ths

    clean, _ = _track(steady)
    disturbed, updates = _track(stray)

    assert disturbed.state.locked and _unlocks(updates) == 0
    assert clean.state.bpm is not None and disturbed.state.bpm is not None
    assert abs(disturbed.state.bpm - clean.state.bpm) < 0.3


def test_half_time_feel_keeps_the_locked_tempo() -> None:
    eighths = onset_times(120, "x" * 8, bars=4, jitter_seconds=0.015, start_seconds=1.0, seed=5)
    half_notes = onset_times(
        120, "x...x...", bars=8, jitter_seconds=0.015, start_seconds=eighths[-1] + 0.25, seed=6
    )

    tracker, updates = _track(np.concatenate([eighths, half_notes]))

    assert _unlocks(updates) == 0
    assert tracker.state.locked
    assert tracker.state.bpm == pytest.approx(120.0, abs=1.0)  # not 60


def test_slow_drift_is_followed_without_unlocking() -> None:
    # 8th notes from 92 to 98 BPM over a minute.
    rng = np.random.default_rng(7)
    times, now = [], 1.0
    while now < 61.0:
        times.append(now + rng.normal(0.0, 0.015))
        now += 60.0 / (92.0 + 6.0 * (now - 1.0) / 60.0) / 2
    tracker, updates = _track(np.array(times))

    assert _unlocks(updates) == 0
    assert tracker.state.locked
    assert tracker.state.bpm == pytest.approx(98.0, abs=1.0)


def test_a_new_tempo_unlocks_then_locks_again() -> None:
    first = onset_times(92, "x" * 8, bars=8, jitter_seconds=0.015, start_seconds=1.0, seed=8)
    change_seconds = first[-1] + 60.0 / 110.0 / 2
    second = onset_times(
        110, "x" * 8, bars=10, jitter_seconds=0.015, start_seconds=change_seconds, seed=9
    )

    tracker, updates = _track(np.concatenate([first, second]))

    after = [update for update in updates if update.timestamp_seconds > change_seconds]
    unlock_index = next(i for i, update in enumerate(after) if not update.locked)
    unlock = after[unlock_index]
    relock = _first_lock(after[unlock_index:])
    # Sustained disagreement only: the old tempo holds for `unlock_seconds` at least.
    assert unlock.timestamp_seconds - change_seconds >= TEMPO.unlock_seconds
    assert relock.timestamp_seconds - change_seconds < 12.0
    assert relock.bpm == pytest.approx(110.0, abs=1.0)
    assert tracker.state.locked and tracker.state.bpm == pytest.approx(110.0, abs=1.0)


def test_silence_keeps_the_lock() -> None:
    before = onset_times(100, "x" * 8, bars=6, jitter_seconds=0.01, start_seconds=1.0, seed=10)
    after = onset_times(
        100, "x" * 8, bars=4, jitter_seconds=0.01, start_seconds=before[-1] + 30.0, seed=11
    )

    tracker, updates = _track(np.concatenate([before, after]))

    assert _unlocks(updates) == 0
    assert tracker.state.locked
    assert tracker.state.bpm == pytest.approx(100.0, abs=1.0)


def test_sparse_onsets_give_no_tempo() -> None:
    tracker, updates = _track(1.0 + 2.5 * np.arange(10))

    assert updates == []
    assert tracker.state.bpm is None
    assert not tracker.state.locked


def test_random_onsets_rarely_lock_and_keep_a_low_confidence() -> None:
    locks, confidences = 0, []
    for seed in range(10):
        times = np.sort(np.random.default_rng(seed).uniform(1.0, 21.0, 80))  # 4 per second
        tracker, updates = _track(times)
        locks += any(update.locked for update in updates)
        confidences.append(tracker.state.confidence)

    assert locks <= 1
    assert float(np.mean(confidences)) < 0.3


def test_updates_come_only_with_a_change() -> None:
    onsets = onset_times(100, "x" * 8, bars=10, jitter_seconds=0.01, start_seconds=1.0, seed=12)

    _, updates = _track(onsets)

    for previous, update in itertools.pairwise(updates):
        assert previous.bpm is not None and update.bpm is not None
        assert (
            update.locked != previous.locked
            or abs(update.bpm - previous.bpm) >= TEMPO.report_bpm_step
            or abs(update.confidence - previous.confidence) >= TEMPO.report_confidence_step
        )
    # Each update is stamped with the hop that produced it, one hop after its onset.
    onset_frames = {seconds_to_frames(t, SR) + HOP for t in onsets}
    assert all(update.frame_index in onset_frames for update in updates)


def test_a_hop_without_an_onset_changes_nothing() -> None:
    tracker = TempoTracker(CONFIG, SR)

    assert tracker.update(None, 48_000) is None
    assert tracker.state.bpm is None


def test_lock_settings_are_configurable() -> None:
    onsets = onset_times(92, "x" * 8, bars=12, jitter_seconds=0.02, start_seconds=1.0, seed=1)
    slow = CONFIG.model_copy(update={"tempo": TempoConfig(lock_seconds=6.0)})

    _, default_updates = _track(onsets)
    _, slow_updates = _track(onsets, slow)

    assert (
        _first_lock(slow_updates).timestamp_seconds
        >= _first_lock(default_updates).timestamp_seconds + 2.0
    )


# --- end to end ----------------------------------------------------------------------------------


def test_a_rendered_riff_locks_through_the_analysis_worker() -> None:
    rendered = riff(
        ["E2", "E2", "G2", "A2"],
        bpm=104,
        step_beats=0.5,
        repeats=8,
        lead_in_seconds=0.5,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    events, _ = analyze_offline(rendered.samples, CONFIG, SR)

    updates = [e.event for e in events if isinstance(e.event, TempoUpdate)]
    assert _first_lock(updates).timestamp_seconds < 0.5 + 4.0
    state = replay(e.event for e in events)
    assert state.tempo_locked
    assert state.bpm == pytest.approx(104.0, abs=0.5)


@pytest.mark.parametrize(
    ("name", "bpm", "first_note_seconds", "max_lock_seconds"),
    [
        ("guitar_riff_e2_g2_a2_92bpm.wav", 91.9, 2.67, 4.0),
        ("guitar_riff_e2_g2_a2_120bpm.wav", 119.2, 1.22, 4.0),
        # Strums: only about 55% detected, and 48 ms timing spread.
        ("guitar_chords_e_g_a_100bpm.wav", 99.8, 1.15, 6.0),
    ],
)
def test_recorded_riffs_lock_at_their_tempo_and_hold_it(
    name: str, bpm: float, first_note_seconds: float, max_lock_seconds: float
) -> None:
    samples, sample_rate_hz = sf.read(DATA / name, dtype="float32")
    events, _ = analyze_offline(samples, CONFIG, sample_rate_hz)
    updates = [e.event for e in events if isinstance(e.event, TempoUpdate)]
    state = replay(e.event for e in events)

    lock = _first_lock(updates)
    assert lock.timestamp_seconds - first_note_seconds <= max_lock_seconds
    locked_bpms = [u.bpm for u in updates if u.locked]
    assert all(b is not None and abs(b - bpm) <= 1.5 for b in locked_bpms)
    assert _unlocks(updates) == 0
    assert state.tempo_locked
    assert state.bpm == pytest.approx(bpm, abs=1.0)


def test_held_notes_give_no_tempo() -> None:
    samples, sample_rate_hz = sf.read(DATA / "guitar_open_strings.wav", dtype="float32")

    events, _ = analyze_offline(samples, CONFIG, sample_rate_hz)

    assert not any(isinstance(e.event, TempoUpdate) for e in events)
    assert replay(e.event for e in events).bpm is None

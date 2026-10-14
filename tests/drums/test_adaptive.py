"""Unit tests for adaptive drummer accent tracking and pattern locking (Step 5.4, §93, §147)."""

import pytest

from jammate.config import BeatClockConfig, DrumConfig
from jammate.drums.adaptive import AccentTracker, adapt_pattern_to_accents
from jammate.drums.engine import DrumEngine
from jammate.drums.patterns import STEPS_PER_BAR, get_fill_pattern, get_pattern
from jammate.drums.sampler import DrumEventRing, DrumPiece
from jammate.music.beat_clock import BeatClock, BeatClockSnapshot
from jammate.music.events import OnsetEvent, PlayingState, frames_to_seconds
from jammate.music.state import MusicalState

SAMPLE_RATE_HZ = 48_000


def _clock_snapshot(bpm: float = 120.0, anchor_frame: float = 0.0) -> BeatClockSnapshot:
    return BeatClockSnapshot(
        anchor_frame=anchor_frame,
        bpm=bpm,
        beats_per_bar=4,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )


def _kick_steps(pattern) -> list[int]:
    return [
        s
        for s, hits in enumerate(pattern.steps)
        if any(hit.piece == DrumPiece.KICK for hit in hits)
    ]


def _hat_velocity(pattern, step: int) -> float | None:
    for hit in pattern.steps[step]:
        if hit.piece == DrumPiece.CLOSED_HAT:
            return hit.velocity
    return None


def test_single_bar_of_accents_stays_below_lock_threshold_until_second_bar() -> None:
    tracker = AccentTracker()
    assert tracker.accent_profile == (0.0,) * STEPS_PER_BAR

    # Bar 1: syncopated accents on steps 0, 3, 6, 10
    bar1 = [0.0] * STEPS_PER_BAR
    for s in (0, 3, 6, 10):
        bar1[s] = 0.85
    tracker.record_bar_onsets(bar1, bar=1)
    assert tracker.bars_tracked == 1
    assert max(tracker.accent_profile) <= 0.50

    # Bar 2: repeating the same syncopated accents reaches >= 0.65 lock threshold
    tracker.record_bar_onsets(bar1, bar=2)
    assert tracker.bars_tracked == 2
    profile = tracker.accent_profile
    for s in (0, 3, 6, 10):
        assert profile[s] == pytest.approx(1.0)
    for s in range(STEPS_PER_BAR):
        if s not in (0, 3, 6, 10):
            assert profile[s] == 0.0


def test_repeated_syncopated_accents_lock_kicks_within_two_bars_and_boost_hats() -> None:
    clock = _clock_snapshot(bpm=120.0)
    frames_per_16th = round(clock.frames_per_beat * 0.25)  # 6,000 frames at 120 BPM
    tracker = AccentTracker()

    # Feed OnsetEvents across 2 bars on syncopated subdivisions (0, 3, 6, 10, 14) with ±8 ms jitter
    syncopated_steps = (0, 3, 6, 10, 14)
    jitter_frames = (120, -240, 180, -150, 90)
    for bar_idx in range(2):
        for step, jitter in zip(syncopated_steps, jitter_frames, strict=True):
            global_step = bar_idx * STEPS_PER_BAR + step
            frame = max(0, global_step * frames_per_16th + jitter)
            onset = OnsetEvent(
                frame_index=frame,
                timestamp_seconds=frames_to_seconds(frame, SAMPLE_RATE_HZ),
                strength=0.82,
            )
            tracker.record_onset(onset, clock)

    config = DrumConfig()
    base = get_pattern("rock", "mid")
    assert _kick_steps(base) == [0, 8]

    adapted = adapt_pattern_to_accents(base, tracker.accent_profile, config)
    kicks = _kick_steps(adapted)
    for step in syncopated_steps:
        assert step in kicks

    # Hi-hat velocities on accented 8th-note steps (0, 6, 10, 14) are boosted vs base pattern.
    for step in (6, 10, 14):
        base_hat = _hat_velocity(base, step)
        adapted_hat = _hat_velocity(adapted, step)
        assert base_hat is not None and adapted_hat is not None
        assert adapted_hat > base_hat


def test_refractory_gap_prevents_machine_gun_adjacent_16th_kicks() -> None:
    tracker = AccentTracker()
    # Player plays rapid 16th notes on steps 0, 2, 3, 4 across 2 bars
    mask = [0.0] * STEPS_PER_BAR
    mask[0] = 0.90
    mask[2] = 0.75
    mask[3] = 0.95  # stronger than step 2 and step 4
    mask[4] = 0.70
    tracker.record_bar_onsets(mask, bar=1)
    tracker.record_bar_onsets(mask, bar=2)

    config = DrumConfig(kick_min_step_gap=2)
    base = get_pattern("rock", "mid")
    adapted = adapt_pattern_to_accents(base, tracker.accent_profile, config)
    kicks = _kick_steps(adapted)

    # Step 3 wins over adjacent steps 2 and 4; no two kicks are closer than 2 steps.
    assert 3 in kicks
    assert 2 not in kicks
    assert 4 not in kicks
    for i in range(len(kicks)):
        for j in range(i + 1, len(kicks)):
            diff = abs(kicks[i] - kicks[j])
            circular_dist = min(diff, STEPS_PER_BAR - diff)
            assert circular_dist >= config.kick_min_step_gap


def test_unaccented_syncopated_base_kicks_are_thinned_to_match_player_pocket() -> None:
    tracker = AccentTracker()
    # rock_high has syncopated kicks on steps 2, 6, 10 alongside 0, 8.
    # Player only accents 0, 3, 8 across 2 bars.
    mask = [0.0] * STEPS_PER_BAR
    mask[0] = 0.90
    mask[3] = 0.90
    mask[8] = 0.90
    tracker.record_bar_onsets(mask, bar=1)
    tracker.record_bar_onsets(mask, bar=2)

    base = get_pattern("rock", "high")
    assert _kick_steps(base) == [0, 2, 6, 8, 10]

    adapted = adapt_pattern_to_accents(base, tracker.accent_profile, DrumConfig())
    assert _kick_steps(adapted) == [0, 3, 8]


def test_fills_and_disabled_config_return_pattern_untouched() -> None:
    tracker = AccentTracker()
    mask = [0.9 if s in (0, 3, 7, 11) else 0.0 for s in range(STEPS_PER_BAR)]
    tracker.record_bar_onsets(mask, bar=1)
    tracker.record_bar_onsets(mask, bar=2)

    base = get_pattern("rock", "mid")
    fill = get_fill_pattern(base)
    assert adapt_pattern_to_accents(fill, tracker.accent_profile, DrumConfig()) == fill

    disabled_cfg = DrumConfig(adaptive_accents_enabled=False)
    assert adapt_pattern_to_accents(base, tracker.accent_profile, disabled_cfg) == base


def test_drum_engine_integrates_accent_tracker_from_onset_events() -> None:
    ring = DrumEventRing(capacity=256)
    clock = BeatClock(BeatClockConfig(), sample_rate_hz=SAMPLE_RATE_HZ, manual_bpm=120.0)
    current_bar = 1
    engine = DrumEngine(
        DrumConfig(lookahead_seconds=0.10, humanize_enabled=False),
        sample_rate_hz=SAMPLE_RATE_HZ,
        event_ring=ring,
        beat_clock=clock,
        state_provider=lambda: MusicalState(
            bar=current_bar,
            beat=1,
            beat_phase=0.0,
            bpm=120.0,
            tempo_confidence=1.0,
            tempo_locked=True,
            intensity=0.55,
            playing_state=PlayingState.PLAYING,
        ),
        frame_provider=lambda: (current_bar - 1) * 96_000,
        manual_bpm_active=True,
    )

    # Feed syncopated accents on steps 0, 3, 6 in bars 1 and 2 (6,000 frames per 16th at 120 BPM).
    for bar_idx in range(2):
        for step in (0, 3, 6):
            frame = (bar_idx * STEPS_PER_BAR + step) * 6_000
            engine.handle_event(
                OnsetEvent(
                    frame_index=frame,
                    timestamp_seconds=frames_to_seconds(frame, SAMPLE_RATE_HZ),
                    strength=0.85,
                )
            )

    current_bar = 3
    pat_bar3 = engine.controller.pattern_for_bar(3)
    assert pat_bar3 is not None
    assert 3 in _kick_steps(pat_bar3)
    assert 6 in _kick_steps(pat_bar3)

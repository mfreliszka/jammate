import pytest

from jammate.config import BeatClockConfig, DrumConfig
from jammate.drums.engine import (
    DrumEngine,
    DrummerController,
    DrummerState,
    DrumStepScheduler,
    step_to_frame,
)
from jammate.drums.patterns import STEPS_PER_BAR, get_pattern
from jammate.drums.sampler import DrumEventRing, DrumPiece
from jammate.music.beat_clock import BeatClock, BeatClockSnapshot
from jammate.music.events import OnsetEvent, PlayingState, TempoUpdate, frames_to_seconds
from jammate.music.state import MusicalState

SAMPLE_RATE_HZ = 48_000


def _state(
    *,
    bar: int | None = 1,
    beat: int | None = 1,
    beat_phase: float | None = 0.0,
    bpm: float | None = 100.0,
    tempo_confidence: float = 0.85,
    tempo_locked: bool = True,
    intensity: float = 0.5,
    playing: bool = True,
) -> MusicalState:
    return MusicalState(
        bar=bar,
        beat=beat,
        beat_phase=beat_phase,
        bpm=bpm,
        tempo_confidence=tempo_confidence,
        tempo_locked=tempo_locked,
        intensity=intensity,
        playing_state=PlayingState.PLAYING if playing else PlayingState.STOPPED,
    )


@pytest.mark.parametrize("bpm", [75.0, 92.0, 120.0, 145.0])
def test_step_to_frame_math_across_bpms(bpm: float) -> None:
    anchor = 1_200.0
    clock = BeatClockSnapshot(
        anchor_frame=anchor,
        bpm=bpm,
        beats_per_bar=4,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    fpb = 60.0 * SAMPLE_RATE_HZ / bpm
    frames_per_16th = 0.25 * fpb

    assert step_to_frame(0, clock) == round(anchor)
    assert step_to_frame(4, clock) == round(anchor + fpb)
    assert step_to_frame(16, clock) == round(anchor + 4.0 * fpb)
    for k in range(32):
        assert step_to_frame(k, clock) == round(anchor + k * frames_per_16th)


def test_overlapping_lookahead_windows_never_duplicate_or_miss_events() -> None:
    ring = DrumEventRing(capacity=512)
    scheduler = DrumStepScheduler(ring)
    clock = BeatClockSnapshot(
        anchor_frame=0.0,
        bpm=120.0,
        beats_per_bar=4,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    pattern = get_pattern("rock", "mid")
    expected_hits_per_bar = sum(len(step) for step in pattern.steps)

    # Advance in 240-frame (5 ms) ticks with a 4,800-frame (100 ms) look-ahead over 2 bars
    # (192,000 frames at 120 BPM). Each step falls inside ~20 overlapping windows.
    two_bars_frames = 2 * 4 * 24_000
    lookahead_frames = 4_800
    scheduled = []
    for frame in range(0, two_bars_frames - lookahead_frames + 1, 240):
        scheduled.extend(
            scheduler.schedule_window(
                current_frame=frame,
                lookahead_frames=lookahead_frames,
                clock=clock,
                pattern_for_bar=lambda _bar: pattern,
            )
        )

    assert len(scheduled) == 2 * expected_hits_per_bar
    bar1_kicks = [e for e in scheduled if e.frame_index == 0 and e.piece == DrumPiece.KICK]
    bar2_kicks = [e for e in scheduled if e.frame_index == 96_000 and e.piece == DrumPiece.KICK]
    assert len(bar1_kicks) == 1
    assert len(bar2_kicks) == 1


def test_beat_clock_phase_corrections_and_mid_bar_tempo_change_never_drop_or_double_steps() -> None:
    ring = DrumEventRing(capacity=512)
    scheduler = DrumStepScheduler(ring)
    clock = BeatClock(BeatClockConfig(), SAMPLE_RATE_HZ)
    clock.start_at(bpm=100.0, anchor_frame=0.0)
    pattern = get_pattern("stoner_rock", "mid")
    expected_hits_per_bar = sum(len(step) for step in pattern.steps)

    lookahead_frames = 4_800
    scheduled = []
    frame = 0
    while (scheduler.last_scheduled_step or -1) < 2 * STEPS_PER_BAR - 1:
        if frame == 36_000:
            clock.on_tempo(
                TempoUpdate(
                    frame_index=frame,
                    timestamp_seconds=frames_to_seconds(frame, SAMPLE_RATE_HZ),
                    bpm=112.0,
                    confidence=0.9,
                    locked=True,
                )
            )
        if frame in (14_000, 28_000, 52_000, 76_000):
            clock.on_onset(
                OnsetEvent(
                    frame_index=frame + 300,
                    timestamp_seconds=frames_to_seconds(frame + 300, SAMPLE_RATE_HZ),
                    strength=0.8,
                )
            )
        snap = clock.snapshot
        assert snap is not None
        batch = scheduler.schedule_window(
            current_frame=frame,
            lookahead_frames=lookahead_frames,
            clock=snap,
            pattern_for_bar=lambda _bar: pattern,
        )
        for ev in batch:
            if scheduled and ev.frame_index < scheduled[-1].frame_index:
                pytest.fail("scheduled event frames must be non-decreasing")
            scheduled.append(ev)
        frame += 480

    assert scheduler.last_scheduled_step is not None
    assert scheduler.last_scheduled_step >= 2 * STEPS_PER_BAR - 1
    assert len(scheduled) >= 2 * expected_hits_per_bar


def test_drum_engine_worker_thread_lifecycle_and_every_8th_bar_fill() -> None:
    ring = DrumEventRing(capacity=256)
    clock = BeatClock(BeatClockConfig(), SAMPLE_RATE_HZ, manual_bpm=120.0)
    current_frame = 0
    engine = DrumEngine(
        DrumConfig(scheduler_interval_seconds=0.002),
        SAMPLE_RATE_HZ,
        ring,
        clock,
        lambda: _state(bpm=120.0, tempo_confidence=1.0, tempo_locked=True),
        lambda: current_frame,
        manual_bpm_active=True,
    )

    assert engine.current_pattern(1).name == "rock_mid"
    assert engine.current_pattern(8).name == "rock_mid_fill"

    engine.start()
    assert engine.is_running
    engine.stop()
    assert not engine.is_running
    assert ring.pending_count() > 0


# --- Step 3.4: DrummerController decision state machine ------------------------------------------


def test_entry_waits_for_confidence_and_joins_on_next_bar_with_crash() -> None:
    controller = DrummerController(DrumConfig(entry_tempo_confidence=0.6))

    # Unlocked or low confidence stays in LISTENING.
    d0 = controller.update(_state(bar=None, beat=None, tempo_locked=False, tempo_confidence=0.3))
    assert d0.state == DrummerState.LISTENING
    assert d0.pattern is None

    d1 = controller.update(_state(bar=2, beat=3, tempo_locked=True, tempo_confidence=0.45))
    assert d1.state == DrummerState.LISTENING

    # Confidence reaches threshold in bar 2 beat 3 -> arms WAITING_FOR_BAR for bar 3.
    d2 = controller.update(_state(bar=2, beat=3, tempo_locked=True, tempo_confidence=0.75))
    assert d2.state == DrummerState.WAITING_FOR_BAR
    assert d2.entry_bar == 3
    assert d2.pattern is None
    assert controller.pattern_for_bar(2) is None
    assert controller.pattern_for_bar(3) is not None  # ready for look-ahead into bar 3

    # Bar 3 beat 1 -> transitions to PLAYING with crash on downbeat.
    d3 = controller.update(_state(bar=3, beat=1, tempo_locked=True, tempo_confidence=0.80))
    assert d3.state == DrummerState.PLAYING
    assert d3.pattern is not None
    assert d3.crash_on_downbeat is True

    # Bar 4 beat 1 -> continues PLAYING without extra entry crash.
    d4 = controller.update(_state(bar=4, beat=1, tempo_locked=True, tempo_confidence=0.80))
    assert d4.state == DrummerState.PLAYING
    assert d4.crash_on_downbeat is False


def test_intensity_tiers_use_hysteresis_and_apply_only_at_bar_boundaries() -> None:
    controller = DrummerController(DrumConfig())
    controller.update(_state(bar=1, beat=2, intensity=0.50))  # arms entry on bar 2 at "mid"
    d_bar2 = controller.update(_state(bar=2, beat=1, intensity=0.50))
    assert d_bar2.state == DrummerState.PLAYING
    assert d_bar2.tier == "mid"

    # Mid-bar jump to high intensity (0.85) in bar 2 beat 3 does NOT change bar 2's tier.
    d_bar2_b3 = controller.update(_state(bar=2, beat=3, intensity=0.85))
    assert d_bar2_b3.tier == "mid"

    # Takes effect on bar 3 boundary!
    d_bar3 = controller.update(_state(bar=3, beat=1, intensity=0.85))
    assert d_bar3.tier == "high"

    # Small dip to 0.72 stays within the high hysteresis band (0.75 - 0.05 = 0.70).
    controller.update(_state(bar=3, beat=4, intensity=0.72))
    d_bar4 = controller.update(_state(bar=4, beat=1, intensity=0.72))
    assert d_bar4.tier == "high"

    # Drop below 0.70 (to 0.55) switches to "mid" on bar 5.
    controller.update(_state(bar=4, beat=3, intensity=0.55))
    d_bar5 = controller.update(_state(bar=5, beat=1, intensity=0.55))
    assert d_bar5.tier == "mid"

    # Small dip to 0.32 stays within mid hysteresis band (0.35 - 0.05 = 0.30), so bar 6 stays mid.
    controller.update(_state(bar=5, beat=3, intensity=0.32))
    d_bar6 = controller.update(_state(bar=6, beat=1, intensity=0.32))
    assert d_bar6.tier == "mid"

    # Drop below 0.30 (to 0.15) switches to "low" on bar 7.
    controller.update(_state(bar=6, beat=4, intensity=0.15))
    d_bar7 = controller.update(_state(bar=7, beat=1, intensity=0.15))
    assert d_bar7.tier == "low"


def test_plays_fill_every_8_bars_from_entry() -> None:
    controller = DrummerController(DrumConfig(fill_every_bars=8))
    controller.update(_state(bar=1, beat=3))  # entry_bar = 2
    for bar in range(2, 18):
        decision = controller.update(_state(bar=bar, beat=1))
        expected_fill = bar in (9, 17)  # 8th and 16th bars of playing (bars 2..9, 10..17)
        assert decision.is_fill is expected_fill
        assert decision.pattern is not None
        assert decision.pattern.name.endswith("_fill") is expected_fill


def test_silence_holds_after_2_bars_stops_after_8_bars_and_reenters_on_resume() -> None:
    controller = DrummerController(DrumConfig(silence_hold_bars=2.0, silence_stop_bars=8.0))
    controller.update(_state(bar=1, beat=2))
    controller.update(_state(bar=2, beat=1, intensity=0.85))
    assert controller.state == DrummerState.PLAYING

    # 1 beat of silence (bar 3 beat 1 -> bar 3 beat 2 = 0.25 bars): keeps PLAYING.
    controller.update(_state(bar=3, beat=1, playing=False, intensity=0.85))
    d_1beat = controller.update(_state(bar=3, beat=2, playing=False, intensity=0.85))
    assert d_1beat.state == DrummerState.PLAYING

    # 2 bars of silence (bar 5 beat 1): drops to HOLDING with "low" tier pattern.
    d_2bars = controller.update(_state(bar=5, beat=1, playing=False, intensity=0.20))
    assert d_2bars.state == DrummerState.HOLDING
    assert d_2bars.tier == "low"
    assert d_2bars.pattern is not None and d_2bars.pattern.tier == "low"

    # Brief resume during HOLDING returns immediately to PLAYING.
    d_resume_hold = controller.update(_state(bar=5, beat=3, playing=True, intensity=0.50))
    assert d_resume_hold.state == DrummerState.PLAYING

    # Now stop playing at bar 6 beat 1 for 8 full bars (until bar 14 beat 1) -> STOPPED.
    controller.update(_state(bar=6, beat=1, playing=False))
    controller.update(_state(bar=8, beat=1, playing=False))
    assert controller.state == DrummerState.HOLDING
    d_8bars = controller.update(_state(bar=14, beat=1, playing=False))
    assert d_8bars.state == DrummerState.STOPPED
    assert d_8bars.pattern is None

    # Player resumes in bar 15 beat 2 -> WAITING_FOR_BAR (entry_bar = 16) -> PLAYING on bar 16.
    d_rearm = controller.update(_state(bar=15, beat=2, playing=True, intensity=0.60))
    assert d_rearm.state == DrummerState.WAITING_FOR_BAR
    assert d_rearm.entry_bar == 16

    d_reenter = controller.update(_state(bar=16, beat=1, playing=True, intensity=0.60))
    assert d_reenter.state == DrummerState.PLAYING
    assert d_reenter.crash_on_downbeat is True
    assert d_reenter.pattern is not None


def test_timing_offset_seconds_shifts_scheduled_frames() -> None:
    ring = DrumEventRing(capacity=128)
    clock = BeatClock(BeatClockConfig(), sample_rate_hz=SAMPLE_RATE_HZ, manual_bpm=120.0)
    engine = DrumEngine(
        DrumConfig(timing_offset_seconds=0.010, lookahead_seconds=0.10),
        sample_rate_hz=SAMPLE_RATE_HZ,
        event_ring=ring,
        beat_clock=clock,
        state_provider=lambda: _state(bar=1, beat=1),
        frame_provider=lambda: 0,
        manual_bpm_active=True,
    )
    events = engine.schedule_available(current_frame=0)
    assert events
    # +10 ms at 48 kHz = +480 frames for step 0.
    assert events[0].frame_index == 480

    engine.set_timing_offset_seconds(-0.005)
    assert engine.timing_offset_seconds == pytest.approx(-0.005)


def test_ten_minute_continuous_drum_run_has_zero_drops_and_zero_drift() -> None:
    """Step 3.5 (§68, §113): 10 simulated minutes of continuous drum scheduling and draining."""
    ring = DrumEventRing(capacity=256)
    clock = BeatClock(BeatClockConfig(), sample_rate_hz=SAMPLE_RATE_HZ, manual_bpm=120.0)
    current_frame = 0
    current_bar = 1

    def state_provider() -> MusicalState:
        return _state(bpm=120.0, bar=current_bar, beat=1, intensity=0.55)

    engine = DrumEngine(
        DrumConfig(lookahead_seconds=0.10),
        sample_rate_hz=SAMPLE_RATE_HZ,
        event_ring=ring,
        beat_clock=clock,
        state_provider=state_provider,
        frame_provider=lambda: current_frame,
        manual_bpm_active=True,
    )

    total_frames = 10 * 60 * SAMPLE_RATE_HZ  # 10 minutes = 28,800,000 frames (300 bars @ 120 BPM)
    step_frames = 2_400  # 50 ms scheduler step (half of the 100 ms look-ahead window)
    frames_per_bar = 4 * 24_000  # 96,000 frames per bar at 120 BPM
    popped_frames: list[int] = []

    for frame in range(0, total_frames, step_frames):
        current_frame = frame
        current_bar = frame // frames_per_bar + 1
        engine.schedule_available(current_frame=frame)
        for ev in ring.pop_due(frame + step_frames):
            popped_frames.append(ev.frame_index)

    assert ring.dropped_events == 0
    assert len(popped_frames) > 3_000
    assert popped_frames == sorted(popped_frames)
    # Bar 300 beat 1 is at exact frame 299 * 96,000 = 28,704,000 (zero drift).
    assert 299 * frames_per_bar in popped_frames

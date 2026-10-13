"""Tests for `jammate.music.beat_clock` (Step 3.1, §31-§33)."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jammate.analysis.worker import analyze_offline
from jammate.config import AnalysisConfig, BeatClockConfig
from jammate.music.beat_clock import (
    BeatClock,
    align_anchor,
    beat_to_frame,
    frame_to_beat,
    frames_per_beat,
    next_bar_frame,
    next_beat_frame,
)
from jammate.music.events import OnsetEvent, SampleFrame, TempoUpdate, frames_to_seconds
from jammate.music.state import replay
from tests.signals import onset_times

SR = 48_000
CONFIG = BeatClockConfig()
DATA = Path(__file__).resolve().parents[1] / "data"


def _onset(frame_index: SampleFrame, strength: float = 0.5) -> OnsetEvent:
    return OnsetEvent(
        frame_index=frame_index,
        timestamp_seconds=frames_to_seconds(frame_index, SR),
        strength=strength,
    )


def _tempo(
    frame_index: SampleFrame, bpm: float = 100.0, locked: bool = True, confidence: float = 0.9
) -> TempoUpdate:
    return TempoUpdate(
        frame_index=frame_index,
        timestamp_seconds=frames_to_seconds(frame_index, SR),
        bpm=bpm,
        confidence=confidence,
        locked=locked,
    )


# --- pure conversions & 10-minute drift check ----------------------------------------------------


@pytest.mark.parametrize("bpm", [75.0, 91.9, 92.0, 100.0, 113.7, 120.0, 145.3])
def test_beat_to_frame_and_frame_to_beat_agree_at_every_boundary(bpm: float) -> None:
    anchor = 1_234.5
    for bar in range(1, 12):
        for beat in range(1, 5):
            frame = beat_to_frame(bar, beat, 0.0, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)
            pos = frame_to_beat(frame, anchor, bpm, SR)
            assert (pos.bar, pos.beat, pos.beat_phase) == (bar, beat, 0.0)

            before = frame_to_beat(frame - 1, anchor, bpm, SR)
            expected_prev_beat = 4 if beat == 1 else beat - 1
            expected_prev_bar = bar - 1 if beat == 1 else bar
            assert (before.bar, before.beat) == (expected_prev_bar, expected_prev_beat)
            assert 0.99 < before.beat_phase < 1.0


@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5, 0.75])
def test_sub_beat_phase_round_trips_within_one_frame(phase: float) -> None:
    anchor = 4_800.0
    bpm = 92.0
    frame = beat_to_frame(3, 2, phase, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)
    pos = frame_to_beat(frame, anchor, bpm, SR)

    assert (pos.bar, pos.beat) == (3, 2)
    assert pos.beat_phase == pytest.approx(phase, abs=1.5 / frames_per_beat(bpm, SR))


def test_next_beat_and_next_bar_frames_advance_strictly() -> None:
    anchor = 10_000.0
    bpm = 92.0
    bar1_beat1 = beat_to_frame(1, 1, 0.0, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)
    bar1_beat2 = beat_to_frame(1, 2, 0.0, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)
    bar2_beat1 = beat_to_frame(2, 1, 0.0, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)

    assert next_beat_frame(bar1_beat1 - 1, anchor, bpm, SR) == bar1_beat1
    assert next_beat_frame(bar1_beat1, anchor, bpm, SR) == bar1_beat2
    assert next_bar_frame(bar1_beat1, anchor, bpm, SR) == bar2_beat1
    assert next_bar_frame(bar2_beat1 - 1, anchor, bpm, SR) == bar2_beat1


def test_no_drift_over_ten_simulated_minutes() -> None:
    # At 92 BPM, frames_per_beat is 31304.347826... (fractional). Over 10 minutes (920 beats =
    # 230 bars), naive per-beat integer rounding would drift by 320 frames (~6.7 ms).
    bpm = 92.0
    anchor = 0.0
    ten_minutes_frames = 10 * 60 * SR  # 28,800,000 frames = exactly 920 beats at 92 BPM
    bar_231_beat_1 = beat_to_frame(231, 1, 0.0, anchor_frame=anchor, bpm=bpm, sample_rate_hz=SR)

    assert bar_231_beat_1 == ten_minutes_frames
    pos = frame_to_beat(ten_minutes_frames, anchor, bpm, SR)
    assert (pos.bar, pos.beat, pos.beat_phase) == (231, 1, 0.0)


def test_invalid_parameters_raise_value_error() -> None:
    with pytest.raises(ValueError, match="bpm"):
        frames_per_beat(0.0, SR)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        frames_per_beat(100.0, 0)
    with pytest.raises(ValueError, match="beat must be"):
        beat_to_frame(1, 5, anchor_frame=0.0, bpm=100.0, sample_rate_hz=SR)
    with pytest.raises(ValueError, match="beat_phase"):
        beat_to_frame(1, 1, 1.0, anchor_frame=0.0, bpm=100.0, sample_rate_hz=SR)


# --- start-up alignment & downbeat selection -----------------------------------------------------


def test_align_anchor_picks_the_strongest_accented_beat_as_downbeat() -> None:
    bpm = 100.0
    fpb = frames_per_beat(bpm, SR)
    start_frame = 24_000
    # 4 bars of quarter notes where beat index 2 (the 3rd quarter note played) is accented in
    # every bar: that position is the true downbeat of the riff.
    onsets = [
        _onset(round(start_frame + i * fpb), strength=0.9 if i % 4 == 2 else 0.3) for i in range(16)
    ]

    anchor = align_anchor(onsets, bpm, SR)
    pos = frame_to_beat(round(start_frame + 2 * fpb), anchor, bpm, SR)

    assert pos.beat == 1
    assert pos.beat_phase == pytest.approx(0.0, abs=1e-3)


def test_align_anchor_picks_the_most_repeated_position_in_riff_cycle() -> None:
    bpm = 120.0
    fpb = frames_per_beat(bpm, SR)
    # Syncopated 4-bar phrase where only beat 2 of the raw grid is played in all 4 bars.
    played_beats = [0, 1, 2, 5, 8, 9, 13, 14]  # offset 1 mod 4 appears 4 times (1, 5, 9, 13)
    onsets = [_onset(round(10_000 + b * fpb), strength=0.5) for b in played_beats]

    anchor = align_anchor(onsets, bpm, SR)
    pos = frame_to_beat(round(10_000 + 1 * fpb), anchor, bpm, SR)

    assert pos.beat == 1
    assert pos.beat_phase == pytest.approx(0.0, abs=1e-3)


def test_align_anchor_refines_phase_across_jittered_eighth_notes() -> None:
    bpm = 92.0
    lead_in_seconds = 1.5
    times = onset_times(
        bpm,
        "xxxxxxxx",
        bars=4,
        steps_per_beat=2,
        jitter_seconds=0.012,
        start_seconds=lead_in_seconds,
        seed=7,
    )
    onsets = [_onset(round(t * SR), strength=0.6) for t in times]

    anchor = align_anchor(onsets, bpm, SR)
    true_anchor = lead_in_seconds * SR
    bar_frames = 4 * frames_per_beat(bpm, SR)
    phase_error_seconds = (
        ((anchor - true_anchor + 0.5 * bar_frames) % bar_frames) - 0.5 * bar_frames
    ) / SR

    assert abs(phase_error_seconds) < 0.01


# --- PLL phase correction, tempo smoothing & persistent-error re-anchor --------------------------


def test_pll_correction_converges_from_initial_phase_offset() -> None:
    bpm = 100.0
    fpb = frames_per_beat(bpm, SR)
    clock = BeatClock(CONFIG, SR)
    # Start the clock 75 ms (3,600 frames = 0.125 beats) early relative to the player's beats.
    true_anchor = 48_000.0
    clock.start_at(bpm, anchor_frame=true_anchor - 3_600.0)

    errors_ms: list[float] = []
    for beat_idx in range(1, 20):
        onset_frame = round(true_anchor + beat_idx * fpb)
        clock.on_onset(_onset(onset_frame))
        assert clock.anchor_frame is not None
        err_frames = ((clock.anchor_frame - true_anchor + 0.5 * fpb) % fpb) - 0.5 * fpb
        errors_ms.append(abs(err_frames) / SR * 1000.0)

    assert errors_ms[0] < 75.0
    assert errors_ms[-1] < 2.0  # converged to within 2 ms
    assert clock.reanchors == 0


def test_eighth_note_offbeats_do_not_pull_clock_off_the_beat() -> None:
    bpm = 110.0
    fpb = frames_per_beat(bpm, SR)
    clock = BeatClock(CONFIG, SR)
    clock.start_at(bpm, anchor_frame=0.0)

    # Feed 4 bars of straight 8th notes (on-beat and off-beat alternating).
    for step in range(32):
        clock.on_onset(_onset(round(step * 0.5 * fpb)))

    pos_on_beat = clock.position_at(round(16 * fpb))
    assert pos_on_beat is not None
    assert (pos_on_beat.bar, pos_on_beat.beat) == (5, 1)
    assert pos_on_beat.beat_phase == pytest.approx(0.0, abs=1e-3)
    assert clock.reanchors == 0


def test_estimator_bpm_noise_produces_no_phase_or_beat_jumps() -> None:
    clock = BeatClock(CONFIG, SR)
    base_bpm = 92.0
    clock.start_at(base_bpm, anchor_frame=0.0)
    rng = np.random.default_rng(42)

    previous_total_beats = 0.0
    frame = 0
    for _ in range(200):
        frame += 12_000  # 0.25 s steps
        pos_before = clock.position_at(frame)
        assert pos_before is not None
        noisy_bpm = base_bpm + float(rng.uniform(-2.0, 2.0))
        clock.on_tempo(_tempo(frame, bpm=noisy_bpm, locked=True))
        pos_after = clock.position_at(frame)
        assert pos_after is not None

        # At the exact update frame, neither bar, beat nor continuous total_beats may jump.
        assert (pos_after.bar, pos_after.beat) == (pos_before.bar, pos_before.beat)
        assert pos_after.total_beats == pytest.approx(pos_before.total_beats, abs=1e-4)
        assert pos_after.total_beats > previous_total_beats
        previous_total_beats = pos_after.total_beats
        assert 90.0 <= (clock.bpm or 0.0) <= 94.0


def test_reanchors_only_after_persistent_large_phase_error() -> None:
    bpm = 100.0
    fpb = frames_per_beat(bpm, SR)
    clock = BeatClock(CONFIG, SR)
    clock.start_at(bpm, anchor_frame=0.0)

    # One wild onset 0.45 beats off does not trigger a re-anchor.
    clock.on_onset(_onset(round(1.45 * fpb)))
    assert clock.reanchors == 0
    # Back on the beat: resets the large-error counter.
    clock.on_onset(_onset(round(2.0 * fpb)))
    assert clock.reanchors == 0

    # Now the player shifts by 0.45 beats and stays there for 8 beats (> 1.5 s and >= 6 onsets).
    for beat_idx in range(3, 11):
        clock.on_onset(_onset(round((beat_idx + 0.45) * fpb)))

    assert clock.reanchors == 1
    pos = clock.position_at(round(11.45 * fpb))
    assert pos is not None
    assert pos.beat_phase < 0.05 or pos.beat_phase > 0.95


def test_unlocking_tempo_stops_the_clock() -> None:
    clock = BeatClock(CONFIG, SR)
    clock.on_onset(_onset(0))
    clock.on_tempo(_tempo(48_000, bpm=100.0, locked=True))
    assert clock.is_running

    clock.on_tempo(_tempo(96_000, bpm=100.0, locked=False))
    assert not clock.is_running
    assert clock.position_at(96_000) is None


# --- end-to-end on recorded guitar riffs ---------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected_bpm"),
    [
        ("guitar_riff_e2_g2_a2_92bpm.wav", 92.0),
        ("guitar_riff_e2_g2_a2_120bpm.wav", 120.0),
    ],
)
def test_beat_counter_stays_steady_on_recorded_riffs(filename: str, expected_bpm: float) -> None:
    samples, sample_rate_hz = sf.read(DATA / filename, dtype="float32")
    emitted, _ = analyze_offline(samples, AnalysisConfig(instrument="guitar"), sample_rate_hz)

    clock = BeatClock(CONFIG, sample_rate_hz)
    total_beats_history: list[float] = []
    bars_and_beats: list[tuple[int, int]] = []
    event_idx = 0
    for hop_frame in range(480, len(samples) + 1, 480):
        while event_idx < len(emitted) and emitted[event_idx].emitted_frame <= hop_frame:
            clock.handle_event(emitted[event_idx].event)
            event_idx += 1
        pos = clock.position_at(hop_frame)
        if pos is not None:
            total_beats_history.append(pos.total_beats)
            if not bars_and_beats or bars_and_beats[-1] != (pos.bar, pos.beat):
                bars_and_beats.append((pos.bar, pos.beat))

    assert clock.is_running
    assert clock.bpm == pytest.approx(expected_bpm, abs=1.5)
    assert clock.reanchors == 0
    # Continuous beat progress is strictly increasing (no backward jumps).
    diffs = np.diff(total_beats_history)
    assert np.all(diffs > 0)
    # Every beat transition steps forward by exactly 1 beat in 4/4.
    for (bar1, beat1), (bar2, beat2) in itertools.pairwise(bars_and_beats):
        idx1 = (bar1 - 1) * 4 + (beat1 - 1)
        idx2 = (bar2 - 1) * 4 + (beat2 - 1)
        assert idx2 == idx1 + 1

    final_state = replay((e.event for e in emitted), beat_clock=BeatClock(CONFIG, sample_rate_hz))
    assert final_state.bar is not None and final_state.bar >= 4
    assert final_state.beat in (1, 2, 3, 4)
    assert final_state.beat_phase is not None and 0.0 <= final_state.beat_phase < 1.0

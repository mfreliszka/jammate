"""Unit tests for deterministic drum humanization (Step 5.4, §95, §147)."""

import pytest

from jammate.config import DrumConfig
from jammate.drums.humanize import (
    Humanizer,
    deterministic_unit_float,
    humanize_step,
    humanize_timing_offset_frames,
    humanize_velocity,
    metric_velocity_scale,
)
from jammate.drums.patterns import STEPS_PER_BAR, StepHit, get_pattern
from jammate.drums.sampler import DrumPiece

SAMPLE_RATE_HZ = 48_000


def test_beat_1_downbeat_always_has_zero_microtiming_offset() -> None:
    for bar_idx in range(64):
        for piece in DrumPiece:
            offset = humanize_timing_offset_frames(
                bar_idx,
                0,
                piece,
                max_timing_ms=4.0,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
            assert offset == 0


def test_non_downbeat_microtiming_stays_within_configured_ms_bounds() -> None:
    max_ms = 4.0
    max_frames = round((max_ms / 1000.0) * SAMPLE_RATE_HZ)  # 192 frames at 48 kHz
    offsets: list[int] = []
    for bar_idx in range(16):
        for step_idx in range(1, STEPS_PER_BAR):
            offset = humanize_timing_offset_frames(
                bar_idx,
                step_idx,
                DrumPiece.CLOSED_HAT,
                max_timing_ms=max_ms,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
            assert -max_frames <= offset <= max_frames
            offsets.append(offset)

    # Offsets span both positive and negative microtiming values and are not all zero.
    assert any(o < 0 for o in offsets)
    assert any(o > 0 for o in offsets)


def test_humanization_is_pure_and_100_percent_deterministic_for_identical_seeds() -> None:
    pattern = get_pattern("rock", "mid")
    humanizer_a = Humanizer(DrumConfig(humanize_enabled=True), sample_rate_hz=SAMPLE_RATE_HZ)
    humanizer_b = Humanizer(DrumConfig(humanize_enabled=True), sample_rate_hz=SAMPLE_RATE_HZ)

    for bar_idx in range(8):
        for step_idx in range(STEPS_PER_BAR):
            hits = pattern.steps[step_idx]
            res_a = humanizer_a.humanize_step(
                hits, bar_index=bar_idx, step_index=step_idx, tier="mid"
            )
            res_b = humanizer_b.humanize_step(
                hits, bar_index=bar_idx, step_index=step_idx, tier="mid"
            )
            assert res_a == res_b

    # Different bars produce different microtiming/velocity variations on non-downbeat steps.
    step2_bar0 = humanizer_a.humanize_step(pattern.steps[2], bar_index=0, step_index=2, tier="mid")
    step2_bar1 = humanizer_a.humanize_step(pattern.steps[2], bar_index=1, step_index=2, tier="mid")
    assert step2_bar0 != step2_bar1


def test_metric_velocity_shaping_hierarchy_downbeat_gt_backbeat_gt_8th_gt_16th() -> None:
    downbeat = metric_velocity_scale(0)
    backbeat = metric_velocity_scale(4)
    eighth = metric_velocity_scale(2)
    sixteenth = metric_velocity_scale(1)
    assert downbeat > backbeat > eighth > sixteenth

    # Averaged across many bars, humanized velocities preserve the metric hierarchy.
    base_vel = 0.80
    mean_downbeat = (
        sum(humanize_velocity(base_vel, b, 0, DrumPiece.CLOSED_HAT) for b in range(64)) / 64
    )
    mean_backbeat = (
        sum(humanize_velocity(base_vel, b, 4, DrumPiece.CLOSED_HAT) for b in range(64)) / 64
    )
    mean_eighth = (
        sum(humanize_velocity(base_vel, b, 2, DrumPiece.CLOSED_HAT) for b in range(64)) / 64
    )
    mean_sixteenth = (
        sum(humanize_velocity(base_vel, b, 1, DrumPiece.CLOSED_HAT) for b in range(64)) / 64
    )
    assert mean_downbeat > mean_backbeat > mean_eighth > mean_sixteenth


def test_ghost_note_snares_stay_in_022_to_032_velocity_range_at_mid_and_high_tiers() -> None:
    ghost_velocities: list[float] = []
    for bar_idx in range(32):
        for step_idx in (7, 9, 15):
            humanized = humanize_step(
                (StepHit(DrumPiece.CLOSED_HAT, 0.70),),
                bar_index=bar_idx,
                step_index=step_idx,
                tier="high",
            )
            for hit in humanized:
                if hit.piece == DrumPiece.SNARE:
                    ghost_velocities.append(hit.velocity)
                    assert 0.22 <= hit.velocity <= 0.32

    assert len(ghost_velocities) >= 10

    # Low intensity tier never adds synthesized ghost snares.
    for bar_idx in range(16):
        for step_idx in (7, 9, 15):
            humanized_low = humanize_step(
                (StepHit(DrumPiece.CLOSED_HAT, 0.65),),
                bar_index=bar_idx,
                step_index=step_idx,
                tier="low",
            )
            assert all(hit.piece != DrumPiece.SNARE for hit in humanized_low)


def test_disabled_humanizer_passes_hits_through_with_zero_offset() -> None:
    humanizer = Humanizer(DrumConfig(humanize_enabled=False), sample_rate_hz=SAMPLE_RATE_HZ)
    hits = (StepHit(DrumPiece.KICK, 0.85), StepHit(DrumPiece.CLOSED_HAT, 0.75))
    out = humanizer.humanize_step(hits, bar_index=3, step_index=6, tier="high")
    assert len(out) == 2
    assert out[0].piece == DrumPiece.KICK
    assert out[0].velocity == pytest.approx(0.85)
    assert out[0].timing_offset_frames == 0
    assert out[1].piece == DrumPiece.CLOSED_HAT
    assert out[1].velocity == pytest.approx(0.75)
    assert out[1].timing_offset_frames == 0
    assert -1.0 <= deterministic_unit_float(3, 6, DrumPiece.KICK) <= 1.0

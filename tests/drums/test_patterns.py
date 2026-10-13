import pytest

from jammate.drums.patterns import (
    DEFAULT_BLUES_SWING,
    FILL_PATTERN,
    INTENSITY_TIERS,
    STEPS_PER_BAR,
    STYLE_FAMILIES,
    get_fill_pattern,
    get_pattern,
    step_beat_offset,
)
from jammate.drums.sampler import DrumPiece


def test_every_family_and_tier_has_valid_16_step_pattern() -> None:
    for family in STYLE_FAMILIES:
        for tier in INTENSITY_TIERS:
            pattern = get_pattern(family, tier)
            assert pattern.family == family
            assert pattern.tier == tier
            assert len(pattern.steps) == STEPS_PER_BAR
            total_hits = sum(len(step) for step in pattern.steps)
            assert total_hits >= 4
            for step in pattern.steps:
                for hit in step:
                    assert 0.0 < hit.velocity <= 1.0


def test_higher_intensity_tiers_increase_energy_or_density() -> None:
    for family in STYLE_FAMILIES:
        low = get_pattern(family, "low")
        mid = get_pattern(family, "mid")
        high = get_pattern(family, "high")
        low_energy = sum(h.velocity for s in low.steps for h in s)
        mid_energy = sum(h.velocity for s in mid.steps for h in s)
        high_energy = sum(h.velocity for s in high.steps for h in s)
        assert low_energy < mid_energy < high_energy
        # Low tier never plays a crash cymbal (§48).
        assert all(h.piece != DrumPiece.CRASH for s in low.steps for h in s)
        # High tier includes a crash accent on beat 1 (§48).
        assert any(h.piece == DrumPiece.CRASH for h in high.steps[0])


def test_blues_patterns_carry_swing_and_swing_shifts_subdivisions_monotonically() -> None:
    blues = get_pattern("blues", "mid")
    assert blues.swing == pytest.approx(DEFAULT_BLUES_SWING)

    # Straight 16ths are spaced by 0.25 beats.
    straight = [step_beat_offset(i, swing=0.0) for i in range(STEPS_PER_BAR)]
    assert straight == [i * 0.25 for i in range(STEPS_PER_BAR)]

    # Triplet shuffle (swing = 1/3) delays off-beat 8ths (step 2) to 2/3 of a beat.
    swung = [step_beat_offset(i, swing=DEFAULT_BLUES_SWING) for i in range(STEPS_PER_BAR)]
    assert swung[0] == pytest.approx(0.0)
    assert swung[2] == pytest.approx(2.0 / 3.0)
    assert swung[4] == pytest.approx(1.0)
    assert all(swung[i] < swung[i + 1] for i in range(STEPS_PER_BAR - 1))


def test_simple_fill_contains_tom_tom_snare_crash() -> None:
    pieces = [hit.piece for step in FILL_PATTERN.steps for hit in step]
    assert pieces == [
        DrumPiece.TOM,
        DrumPiece.TOM,
        DrumPiece.SNARE,
        DrumPiece.CRASH,
    ]
    rock_mid = get_pattern("rock", "mid")
    combined = get_fill_pattern(rock_mid)
    assert combined.steps[:8] == rock_mid.steps[:8]
    assert combined.steps[8:] == FILL_PATTERN.steps[8:]

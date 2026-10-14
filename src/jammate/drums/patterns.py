"""16-step drum pattern library for the virtual drummer (Step 3.3, §44, §48, §49).

Patterns are 16-step grids (one 4/4 bar of 16th notes: `1 e & a 2 e & a 3 e & a 4 e & a`).
Each family (`rock`, `stoner_rock`, `blues`, `funk`, `metal`) provides `low`, `mid` and `high`
intensity variants, plus a shared 1-bar fill (`tom, tom, snare, crash`).
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from jammate.drums.sampler import DrumPiece

type StyleFamily = Literal["rock", "stoner_rock", "blues", "funk", "metal"]
type IntensityTier = Literal["low", "mid", "high"]

STEPS_PER_BAR = 16
STEPS_PER_BAR_12 = 12
STYLE_FAMILIES: tuple[StyleFamily, ...] = ("rock", "stoner_rock", "blues", "funk", "metal")
INTENSITY_TIERS: tuple[IntensityTier, ...] = ("low", "mid", "high")
DEFAULT_BLUES_SWING = 1.0 / 3.0


def steps_per_bar_for_beats(beats_per_bar: int) -> int:
    """Return the step grid length for `beats_per_bar` (`16` for 4/4, `12` for 3/4 and 6/8)."""
    if beats_per_bar in (3, 6):
        return STEPS_PER_BAR_12
    return STEPS_PER_BAR


class StepHit(NamedTuple):
    """A single drum piece hit on a grid step with normalized velocity `0.0..1.0`."""

    piece: DrumPiece
    velocity: float


@dataclass(frozen=True)
class DrumPattern:
    """Immutable 1-bar drum pattern (`16` steps in 4/4, `12` steps in 3/4 or 6/8)."""

    name: str
    family: StyleFamily
    tier: IntensityTier
    steps: tuple[tuple[StepHit, ...], ...]
    swing: float = 0.0

    def __post_init__(self) -> None:
        if len(self.steps) not in (STEPS_PER_BAR, STEPS_PER_BAR_12):
            raise ValueError(
                f"DrumPattern must have {STEPS_PER_BAR} or {STEPS_PER_BAR_12} steps, "
                f"got {len(self.steps)}"
            )
        if not 0.0 <= self.swing <= 0.5:
            raise ValueError(f"swing must lie in [0.0, 0.5], got {self.swing}")

    @property
    def steps_per_bar(self) -> int:
        return len(self.steps)


def step_beat_offset(
    step_index: int,
    swing: float = 0.0,
    beats_per_bar: int = 4,
    steps_per_bar: int | None = None,
) -> float:
    """Return the beat offset within a bar `[0, beats_per_bar)` for `step_index`.

    At `swing == 0.0`, steps are spaced evenly by `beats_per_bar / steps_per_bar` (0.25 beats in
    4/4 and 3/4, 0.5 beats in 6/8). When `swing > 0.0` in simple meters (`4/4`, `3/4`), off-beat
    8th notes (`step % 4 == 2`, the `&` of each beat) are delayed by `swing * 0.5` beats (so
    `swing = 1/3` gives a triplet shuffle at `2/3` of a beat), and odd 16th subdivisions
    (`step % 2 == 1`) are delayed by `swing * 0.25` beats.
    """
    total_steps = (
        steps_per_bar if steps_per_bar is not None else steps_per_bar_for_beats(beats_per_bar)
    )
    if not 0 <= step_index < total_steps:
        raise ValueError(f"step_index must be in 0..{total_steps - 1}, got {step_index}")
    step_beats = beats_per_bar / total_steps
    base = step_index * step_beats
    if swing <= 0.0:
        return base
    if beats_per_bar == 6:
        if step_index % 2 == 1:
            return base + swing * step_beats
        return base
    if step_index % 4 == 2:
        return base + swing * (2.0 * step_beats)
    if step_index % 2 == 1:
        return base + swing * step_beats
    return base


def _parse_track(
    grid: str,
    piece: DrumPiece,
    accent: float,
    normal: float,
    ghost: float,
    steps_per_bar: int = STEPS_PER_BAR,
) -> list[StepHit | None]:
    cleaned = grid.replace(" ", "")
    if len(cleaned) != steps_per_bar:
        raise ValueError(f"grid {grid!r} must have {steps_per_bar} characters")
    out: list[StepHit | None] = []
    for ch in cleaned:
        if ch == "X":
            out.append(StepHit(piece, accent))
        elif ch == "x":
            out.append(StepHit(piece, normal))
        elif ch == "g":
            out.append(StepHit(piece, ghost))
        elif ch == "-":
            out.append(None)
        else:
            raise ValueError(f"unknown grid character {ch!r}")
    return out


def _build_pattern(
    name: str,
    family: StyleFamily,
    tier: IntensityTier,
    tracks: dict[DrumPiece, str],
    *,
    velocity_scale: float = 1.0,
    swing: float = 0.0,
    steps_per_bar: int = STEPS_PER_BAR,
) -> DrumPattern:
    accent = min(1.0, 0.95 * velocity_scale)
    normal = min(1.0, 0.75 * velocity_scale)
    ghost = min(1.0, 0.40 * velocity_scale)
    parsed = {
        piece: _parse_track(grid, piece, accent, normal, ghost, steps_per_bar)
        for piece, grid in tracks.items()
    }
    steps: list[tuple[StepHit, ...]] = []
    for s in range(steps_per_bar):
        hits = tuple(
            hit for piece in DrumPiece if piece in parsed and (hit := parsed[piece][s]) is not None
        )
        steps.append(hits)
    return DrumPattern(name=name, family=family, tier=tier, steps=tuple(steps), swing=swing)


# --- Pattern definitions (§44, §48, §49) ---------------------------------------------------------

_PATTERNS: dict[tuple[StyleFamily, IntensityTier], DrumPattern] = {
    # Rock (§44)
    ("rock", "low"): _build_pattern(
        "rock_low",
        "rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x---x---x---x---",
            DrumPiece.SNARE: "----x-------x---",
            DrumPiece.KICK: "x-------x-------",
        },
        velocity_scale=0.70,
    ),
    ("rock", "mid"): _build_pattern(
        "rock_mid",
        "rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-x-X-x-x-x-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-------X-------",
        },
        velocity_scale=0.85,
    ),
    ("rock", "high"): _build_pattern(
        "rock_high",
        "rock",
        "high",
        {
            DrumPiece.CRASH: "X---------------",
            DrumPiece.CLOSED_HAT: "--X-x-x-X-x-x-x-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-x---x-X-x-----",
        },
        velocity_scale=1.0,
    ),
    # Stoner Rock (§44)
    ("stoner_rock", "low"): _build_pattern(
        "stoner_rock_low",
        "stoner_rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x---x---x---x---",
            DrumPiece.SNARE: "----x-------x---",
            DrumPiece.KICK: "x-----x-x-------",
        },
        velocity_scale=0.75,
    ),
    ("stoner_rock", "mid"): _build_pattern(
        "stoner_rock_mid",
        "stoner_rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-X-x-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-----x-X-------",
        },
        velocity_scale=0.90,
    ),
    ("stoner_rock", "high"): _build_pattern(
        "stoner_rock_high",
        "stoner_rock",
        "high",
        {
            DrumPiece.CRASH: "X-------x-------",
            DrumPiece.CLOSED_HAT: "--X-X-x---X-X-x-",
            DrumPiece.SNARE: "----X-------X--g",
            DrumPiece.KICK: "X-x---x-X-x-----",
        },
        velocity_scale=1.0,
    ),
    # Blues (with shuffle swing)
    ("blues", "low"): _build_pattern(
        "blues_low",
        "blues",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-g-x-g-x-g-x-g-",
            DrumPiece.SNARE: "----x-------x---",
            DrumPiece.KICK: "x-------x-------",
        },
        velocity_scale=0.68,
        swing=DEFAULT_BLUES_SWING,
    ),
    ("blues", "mid"): _build_pattern(
        "blues_mid",
        "blues",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-X-x-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-----x-X-----x-",
        },
        velocity_scale=0.85,
        swing=DEFAULT_BLUES_SWING,
    ),
    ("blues", "high"): _build_pattern(
        "blues_high",
        "blues",
        "high",
        {
            DrumPiece.CRASH: "X---------------",
            DrumPiece.CLOSED_HAT: "--x-X-x-X-x-X-x-",
            DrumPiece.SNARE: "----X--g----X--g",
            DrumPiece.KICK: "X-x---x-X-x---x-",
        },
        velocity_scale=1.0,
        swing=DEFAULT_BLUES_SWING,
    ),
    # Funk
    ("funk", "low"): _build_pattern(
        "funk_low",
        "funk",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-x-x-",
            DrumPiece.SNARE: "----x--g----x---",
            DrumPiece.KICK: "x---------x-----",
        },
        velocity_scale=0.72,
    ),
    ("funk", "mid"): _build_pattern(
        "funk_mid",
        "funk",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-x-X-x-x-x-",
            DrumPiece.SNARE: "----X--g-g--X--g",
            DrumPiece.KICK: "X-x-------x---x-",
        },
        velocity_scale=0.88,
    ),
    ("funk", "high"): _build_pattern(
        "funk_high",
        "funk",
        "high",
        {
            DrumPiece.CRASH: "X---------------",
            DrumPiece.CLOSED_HAT: "--x-X-x-x-x-X-x-",
            DrumPiece.SNARE: "----X--g-g--X-g-",
            DrumPiece.KICK: "X-x---x---x---x-",
        },
        velocity_scale=1.0,
    ),
    # Metal
    ("metal", "low"): _build_pattern(
        "metal_low",
        "metal",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-x-x-",
            DrumPiece.SNARE: "----x-------x---",
            DrumPiece.KICK: "x-x-----x-x-----",
        },
        velocity_scale=0.78,
    ),
    ("metal", "mid"): _build_pattern(
        "metal_mid",
        "metal",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-X-x-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-x---x-X-x-x---",
        },
        velocity_scale=0.92,
    ),
    ("metal", "high"): _build_pattern(
        "metal_high",
        "metal",
        "high",
        {
            DrumPiece.CRASH: "X-------X-------",
            DrumPiece.CLOSED_HAT: "--X-X-X---X-X-X-",
            DrumPiece.SNARE: "----X-------X---",
            DrumPiece.KICK: "X-x-x-x-X-x-x-x-",
        },
        velocity_scale=1.0,
    ),
}

# 12-step 3/4 patterns (3 quarter-note beats x 4 sixteenth steps: beat 1=0, beat 2=4, beat 3=8)
_PATTERNS_3_4: dict[tuple[StyleFamily, IntensityTier], DrumPattern] = {
    ("rock", "low"): _build_pattern(
        "rock_low_3_4",
        "rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x---x---x---",
            DrumPiece.SNARE: "--------x---",
            DrumPiece.KICK: "x-----------",
        },
        velocity_scale=0.70,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("rock", "mid"): _build_pattern(
        "rock_mid_3_4",
        "rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-x-x-x-",
            DrumPiece.SNARE: "--------X---",
            DrumPiece.KICK: "X-----x-----",
        },
        velocity_scale=0.85,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("rock", "high"): _build_pattern(
        "rock_high_3_4",
        "rock",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--X-x-x-x-x-",
            DrumPiece.SNARE: "----X---X---",
            DrumPiece.KICK: "X-x---x-----",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "low"): _build_pattern(
        "stoner_rock_low_3_4",
        "stoner_rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x---x---x---",
            DrumPiece.SNARE: "--------x---",
            DrumPiece.KICK: "x-----x-----",
        },
        velocity_scale=0.75,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "mid"): _build_pattern(
        "stoner_rock_mid_3_4",
        "stoner_rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-",
            DrumPiece.SNARE: "--------X---",
            DrumPiece.KICK: "X---x-x-----",
        },
        velocity_scale=0.90,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "high"): _build_pattern(
        "stoner_rock_high_3_4",
        "stoner_rock",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--X-X-x-X-x-",
            DrumPiece.SNARE: "----X---X--g",
            DrumPiece.KICK: "X-x---x-----",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "low"): _build_pattern(
        "blues_low_3_4",
        "blues",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-g-x-g-x-g-",
            DrumPiece.SNARE: "--------x---",
            DrumPiece.KICK: "x-----------",
        },
        velocity_scale=0.68,
        swing=DEFAULT_BLUES_SWING,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "mid"): _build_pattern(
        "blues_mid_3_4",
        "blues",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-",
            DrumPiece.SNARE: "--------X---",
            DrumPiece.KICK: "X-----x---x-",
        },
        velocity_scale=0.85,
        swing=DEFAULT_BLUES_SWING,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "high"): _build_pattern(
        "blues_high_3_4",
        "blues",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--x-X-x-X-x-",
            DrumPiece.SNARE: "----X--gX--g",
            DrumPiece.KICK: "X-x---x---x-",
        },
        velocity_scale=1.0,
        swing=DEFAULT_BLUES_SWING,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "low"): _build_pattern(
        "funk_low_3_4",
        "funk",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "----g---x---",
            DrumPiece.KICK: "x-----x-----",
        },
        velocity_scale=0.72,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "mid"): _build_pattern(
        "funk_mid_3_4",
        "funk",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-x-X-x-",
            DrumPiece.SNARE: "----X--gX--g",
            DrumPiece.KICK: "X-x---x---x-",
        },
        velocity_scale=0.88,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "high"): _build_pattern(
        "funk_high_3_4",
        "funk",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--x-X-x-X-x-",
            DrumPiece.SNARE: "----X-g-X-g-",
            DrumPiece.KICK: "X-x---x---x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "low"): _build_pattern(
        "metal_low_3_4",
        "metal",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "--------x---",
            DrumPiece.KICK: "x-x---x-----",
        },
        velocity_scale=0.78,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "mid"): _build_pattern(
        "metal_mid_3_4",
        "metal",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-x-X-x-",
            DrumPiece.SNARE: "--------X---",
            DrumPiece.KICK: "X-x-x-x-----",
        },
        velocity_scale=0.92,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "high"): _build_pattern(
        "metal_high_3_4",
        "metal",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--X-X-X-X-X-",
            DrumPiece.SNARE: "----X---X---",
            DrumPiece.KICK: "X-x-x-x-x-x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
}

# 12-step 6/8 patterns (6 eighth-note beats x 2 sixteenth steps: pulse 1=step 0, pulse 2=step 6)
_PATTERNS_6_8: dict[tuple[StyleFamily, IntensityTier], DrumPattern] = {
    ("rock", "low"): _build_pattern(
        "rock_low_6_8",
        "rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "------x-----",
            DrumPiece.KICK: "x-----------",
        },
        velocity_scale=0.70,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("rock", "mid"): _build_pattern(
        "rock_mid_6_8",
        "rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-X-x-x-",
            DrumPiece.SNARE: "------X-----",
            DrumPiece.KICK: "X---x-------",
        },
        velocity_scale=0.85,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("rock", "high"): _build_pattern(
        "rock_high_6_8",
        "rock",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--x-x-X-x-x-",
            DrumPiece.SNARE: "------X---g-",
            DrumPiece.KICK: "X-x-x-----x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "low"): _build_pattern(
        "stoner_rock_low_6_8",
        "stoner_rock",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "------x-----",
            DrumPiece.KICK: "x---x-------",
        },
        velocity_scale=0.75,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "mid"): _build_pattern(
        "stoner_rock_mid_6_8",
        "stoner_rock",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-X-X-x-x-",
            DrumPiece.SNARE: "------X-----",
            DrumPiece.KICK: "X---x-----x-",
        },
        velocity_scale=0.90,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("stoner_rock", "high"): _build_pattern(
        "stoner_rock_high_6_8",
        "stoner_rock",
        "high",
        {
            DrumPiece.CRASH: "X-----x-----",
            DrumPiece.CLOSED_HAT: "--X-x---x-x-",
            DrumPiece.SNARE: "------X---g-",
            DrumPiece.KICK: "X-x-x---x-x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "low"): _build_pattern(
        "blues_low_6_8",
        "blues",
        "low",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-X-x-x-",
            DrumPiece.SNARE: "------x-----",
            DrumPiece.KICK: "x-----------",
        },
        velocity_scale=0.68,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "mid"): _build_pattern(
        "blues_mid_6_8",
        "blues",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-X-x-x-",
            DrumPiece.SNARE: "------X-----",
            DrumPiece.KICK: "X---x-----x-",
        },
        velocity_scale=0.85,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("blues", "high"): _build_pattern(
        "blues_high_6_8",
        "blues",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--x-x-X-x-x-",
            DrumPiece.SNARE: "------X--g--",
            DrumPiece.KICK: "X-x-x---x-x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "low"): _build_pattern(
        "funk_low_6_8",
        "funk",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "----g-x-----",
            DrumPiece.KICK: "x-------x---",
        },
        velocity_scale=0.72,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "mid"): _build_pattern(
        "funk_mid_6_8",
        "funk",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-X-x-x-",
            DrumPiece.SNARE: "----g-X--g--",
            DrumPiece.KICK: "X-x-----x-x-",
        },
        velocity_scale=0.88,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("funk", "high"): _build_pattern(
        "funk_high_6_8",
        "funk",
        "high",
        {
            DrumPiece.CRASH: "X-----------",
            DrumPiece.CLOSED_HAT: "--x-x-X-x-x-",
            DrumPiece.SNARE: "----g-X-g-g-",
            DrumPiece.KICK: "X-x-x---x-x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "low"): _build_pattern(
        "metal_low_6_8",
        "metal",
        "low",
        {
            DrumPiece.CLOSED_HAT: "x-x-x-x-x-x-",
            DrumPiece.SNARE: "------x-----",
            DrumPiece.KICK: "x-x-x-------",
        },
        velocity_scale=0.78,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "mid"): _build_pattern(
        "metal_mid_6_8",
        "metal",
        "mid",
        {
            DrumPiece.CLOSED_HAT: "X-x-x-X-x-x-",
            DrumPiece.SNARE: "------X-----",
            DrumPiece.KICK: "X-x-x---x-x-",
        },
        velocity_scale=0.92,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
    ("metal", "high"): _build_pattern(
        "metal_high_6_8",
        "metal",
        "high",
        {
            DrumPiece.CRASH: "X-----X-----",
            DrumPiece.CLOSED_HAT: "--X-X---X-X-",
            DrumPiece.SNARE: "------X-----",
            DrumPiece.KICK: "X-x-x-X-x-x-",
        },
        velocity_scale=1.0,
        steps_per_bar=STEPS_PER_BAR_12,
    ),
}

# Simple deterministic fill: tom, tom, snare, crash (§49).
FILL_PATTERN: DrumPattern = _build_pattern(
    "fill_simple",
    "rock",
    "high",
    {
        DrumPiece.TOM: "--------X-X-----",
        DrumPiece.SNARE: "------------X---",
        DrumPiece.CRASH: "--------------X-",
    },
    velocity_scale=1.0,
)

FILL_PATTERN_12: DrumPattern = _build_pattern(
    "fill_simple_12",
    "rock",
    "high",
    {
        DrumPiece.TOM: "------X-X---",
        DrumPiece.SNARE: "----------X-",
        DrumPiece.CRASH: "-----------X",
    },
    velocity_scale=1.0,
    steps_per_bar=STEPS_PER_BAR_12,
)


def get_pattern(
    family: StyleFamily,
    tier: IntensityTier = "mid",
    *,
    swing: float | None = None,
    beats_per_bar: int = 4,
) -> DrumPattern:
    """Return the `DrumPattern` for `(family, tier, beats_per_bar)`."""
    if beats_per_bar == 3:
        pattern = _PATTERNS_3_4[(family, tier)]
    elif beats_per_bar == 6:
        pattern = _PATTERNS_6_8[(family, tier)]
    else:
        pattern = _PATTERNS[(family, tier)]
    if swing is not None and swing != pattern.swing:
        return DrumPattern(
            name=pattern.name,
            family=pattern.family,
            tier=pattern.tier,
            steps=pattern.steps,
            swing=swing,
        )
    return pattern


def get_fill_pattern(
    base_pattern: DrumPattern | None = None,
    *,
    swing: float | None = None,
    beats_per_bar: int = 4,
) -> DrumPattern:
    """Return a 1-bar fill pattern: first half groove (if given) + `tom, tom, snare, crash`."""
    use_12 = (base_pattern is not None and len(base_pattern.steps) == STEPS_PER_BAR_12) or (
        base_pattern is None and beats_per_bar in (3, 6)
    )
    fill_template = FILL_PATTERN_12 if use_12 else FILL_PATTERN
    target_swing = (
        swing
        if swing is not None
        else (base_pattern.swing if base_pattern is not None else fill_template.swing)
    )
    if base_pattern is None:
        if target_swing == fill_template.swing:
            return fill_template
        return DrumPattern(
            name=fill_template.name,
            family=fill_template.family,
            tier=fill_template.tier,
            steps=fill_template.steps,
            swing=target_swing,
        )
    half = len(fill_template.steps) // 2
    steps = base_pattern.steps[:half] + fill_template.steps[half:]
    return DrumPattern(
        name=f"{base_pattern.name}_fill",
        family=base_pattern.family,
        tier=base_pattern.tier,
        steps=steps,
        swing=target_swing,
    )

"""Deterministic drum microtiming, metric velocity shaping, and ghost notes (Step 5.4, §95, §147).

Every function in this module is pure and seeded deterministically by `(bar_index, step_index,
piece)` via a 64-bit SplitMix64 integer mixer (independent of `PYTHONHASHSEED`) so unit tests and
offline renders are 100% reproducible across runs and machines.

Key invariants:
- Beat-1 downbeats (`step_index == 0`) always receive **zero** microtiming offset (`0` frames) so
  the bar anchor never drifts relative to `BeatClock`.
- Non-downbeat steps receive microtiming offsets bounded to `±humanize_timing_ms`.
- Metric velocity weights shape hits (`downbeat > backbeat > 8th > 16th`) with deterministic
  per-hit variation (`±humanize_velocity_variation`) and ghost-note snares (`0.22..0.32` velocity)
  on selected 16th subdivisions at `mid` and `high` intensity.
"""

from typing import NamedTuple

from jammate.config import DrumConfig
from jammate.drums.patterns import IntensityTier, StepHit
from jammate.drums.sampler import DrumPiece

_MASK_64 = 0xFFFFFFFFFFFFFFFF
_GHOST_MIN_VELOCITY = 0.22
_GHOST_MAX_VELOCITY = 0.32
_GHOST_CANDIDATE_STEPS: frozenset[int] = frozenset({7, 9, 15})


class HumanizedHit(NamedTuple):
    """A single drum hit after deterministic velocity and microtiming humanization."""

    piece: DrumPiece
    velocity: float
    timing_offset_frames: int


def _splitmix64(value: int) -> int:
    """Pure 64-bit SplitMix64 finalizer (process-independent, deterministic across platforms)."""
    z = (value + 0x9E3779B97F4A7C15) & _MASK_64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (z ^ (z >> 31)) & _MASK_64


def deterministic_unit_float(
    bar_index: int,
    step_index: int,
    piece: DrumPiece | int,
    *,
    salt: int = 0,
) -> float:
    """Return a deterministic float in `[-1.0, 1.0]` for `(bar_index, step_index, piece)`."""
    seed = (
        ((bar_index & 0xFFFFF) << 32)
        ^ ((step_index & 0xFF) << 16)
        ^ ((int(piece) & 0xFF) << 8)
        ^ (salt & 0xFF)
    )
    mixed = _splitmix64(seed)
    unit_01 = (mixed >> 11) * (1.0 / (1 << 53))
    return unit_01 * 2.0 - 1.0


def metric_velocity_scale(step_index: int) -> float:
    """Metric velocity hierarchy: `downbeat (1.05) > backbeat (1.00) > 8th (0.94) > 16th (0.88)`."""
    if step_index == 0:
        return 1.05
    if step_index % 4 == 0:
        return 1.00
    if step_index % 2 == 0:
        return 0.94
    return 0.88


def humanize_timing_offset_frames(
    bar_index: int,
    step_index: int,
    piece: DrumPiece,
    *,
    max_timing_ms: float = 4.0,
    sample_rate_hz: int = 48_000,
) -> int:
    """Return a deterministic sample-frame offset in `[-max_frames, +max_frames]`.

    Beat-1 downbeats (`step_index == 0`) always return `0` so the bar anchor never drifts (§147).
    """
    if step_index == 0 or max_timing_ms <= 0.0:
        return 0
    max_frames = round((max_timing_ms / 1000.0) * sample_rate_hz)
    if max_frames <= 0:
        return 0
    u = deterministic_unit_float(bar_index, step_index, piece, salt=1)
    return round(u * max_frames)


def humanize_velocity(
    base_velocity: float,
    bar_index: int,
    step_index: int,
    piece: DrumPiece,
    *,
    velocity_variation: float = 0.06,
) -> float:
    """Apply metric velocity shaping and deterministic per-hit variation (`±velocity_variation`)."""
    # Preserve explicit ghost-note snare dynamics within the 0.22..0.32 pocket range (§147).
    if piece == DrumPiece.SNARE and step_index % 2 == 1 and base_velocity <= 0.42:
        u01 = (deterministic_unit_float(bar_index, step_index, piece, salt=2) + 1.0) * 0.5
        return _GHOST_MIN_VELOCITY + u01 * (_GHOST_MAX_VELOCITY - _GHOST_MIN_VELOCITY)

    scale = metric_velocity_scale(step_index)
    delta = deterministic_unit_float(bar_index, step_index, piece, salt=2) * velocity_variation
    shaped = base_velocity * scale + delta
    return max(0.05, min(1.0, shaped))


def maybe_ghost_snare(
    existing_hits: tuple[StepHit, ...],
    *,
    bar_index: int,
    step_index: int,
    tier: IntensityTier,
) -> StepHit | None:
    """Optionally synthesize a deterministic ghost-note snare (`0.22..0.32`) on 16th steps."""
    if tier not in ("mid", "high"):
        return None
    if step_index not in _GHOST_CANDIDATE_STEPS:
        return None
    if any(hit.piece in (DrumPiece.SNARE, DrumPiece.TOM, DrumPiece.CRASH) for hit in existing_hits):
        return None

    prob_threshold = 0.35 if tier == "mid" else 0.55
    gate_01 = (deterministic_unit_float(bar_index, step_index, DrumPiece.SNARE, salt=3) + 1.0) * 0.5
    if gate_01 >= prob_threshold:
        return None

    vel_01 = (deterministic_unit_float(bar_index, step_index, DrumPiece.SNARE, salt=4) + 1.0) * 0.5
    ghost_vel = _GHOST_MIN_VELOCITY + vel_01 * (_GHOST_MAX_VELOCITY - _GHOST_MIN_VELOCITY)
    return StepHit(DrumPiece.SNARE, ghost_vel)


def humanize_step(
    hits: tuple[StepHit, ...],
    *,
    bar_index: int,
    step_index: int,
    tier: IntensityTier = "mid",
    max_timing_ms: float = 4.0,
    velocity_variation: float = 0.06,
    sample_rate_hz: int = 48_000,
    enable_ghost_notes: bool = True,
    is_fill: bool = False,
) -> tuple[HumanizedHit, ...]:
    """Humanize all hits on `(bar_index, step_index)` and optionally add a ghost-note snare."""
    effective_hits = list(hits)
    if enable_ghost_notes and not is_fill:
        ghost = maybe_ghost_snare(
            hits,
            bar_index=bar_index,
            step_index=step_index,
            tier=tier,
        )
        if ghost is not None:
            effective_hits.append(ghost)

    out: list[HumanizedHit] = []
    for hit in effective_hits:
        vel = humanize_velocity(
            hit.velocity,
            bar_index,
            step_index,
            hit.piece,
            velocity_variation=velocity_variation,
        )
        offset = humanize_timing_offset_frames(
            bar_index,
            step_index,
            hit.piece,
            max_timing_ms=max_timing_ms,
            sample_rate_hz=sample_rate_hz,
        )
        out.append(HumanizedHit(hit.piece, vel, offset))
    return tuple(out)


class Humanizer:
    """Config-bound deterministic humanizer used by `DrumStepScheduler` and `DrumEngine`."""

    def __init__(self, config: DrumConfig, sample_rate_hz: int = 48_000) -> None:
        self._enabled = config.humanize_enabled
        self._max_timing_ms = config.humanize_timing_ms
        self._velocity_variation = config.humanize_velocity_variation
        self._sample_rate_hz = sample_rate_hz

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def max_timing_ms(self) -> float:
        return self._max_timing_ms

    @property
    def velocity_variation(self) -> float:
        return self._velocity_variation

    def humanize_step(
        self,
        hits: tuple[StepHit, ...],
        *,
        bar_index: int,
        step_index: int,
        tier: IntensityTier = "mid",
        is_fill: bool = False,
    ) -> tuple[HumanizedHit, ...]:
        if not self._enabled:
            return tuple(HumanizedHit(hit.piece, hit.velocity, 0) for hit in hits)
        return humanize_step(
            hits,
            bar_index=bar_index,
            step_index=step_index,
            tier=tier,
            max_timing_ms=self._max_timing_ms,
            velocity_variation=self._velocity_variation,
            sample_rate_hz=self._sample_rate_hz,
            enable_ghost_notes=True,
            is_fill=is_fill,
        )

"""Adaptive drummer accent tracking and pattern locking (Step 5.4, §93, §147).

Tracks an exponentially weighted 16-step onset accent grid across the last 2-4 bars and adapts
base `DrumPattern` grooves so the kick drum and hi-hat lock to the player's recurring accents
while respecting a minimum refractory step gap. All adaptations run ahead of time on the drum
scheduler / analysis threads; the realtime audio callback is completely untouched.
"""

from collections import OrderedDict
from collections.abc import Sequence

from jammate.config import DrumConfig
from jammate.drums.patterns import STEPS_PER_BAR, DrumPattern, StepHit
from jammate.drums.sampler import DrumPiece
from jammate.music.beat_clock import BeatClockSnapshot
from jammate.music.events import MusicalEvent, OnsetEvent


class AccentTracker:
    """Accumulates player onset strengths onto a 16-step bar grid over recent bars (§93, §147)."""

    def __init__(
        self,
        *,
        steps_per_bar: int = STEPS_PER_BAR,
        window_bars: int = 4,
        decay: float = 0.70,
    ) -> None:
        if steps_per_bar < 4:
            raise ValueError(f"steps_per_bar must be >= 4, got {steps_per_bar}")
        if window_bars < 1:
            raise ValueError(f"window_bars must be >= 1, got {window_bars}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0.0, 1.0], got {decay}")
        self._steps_per_bar = steps_per_bar
        self._window_bars = window_bars
        self._decay = decay
        self._bars: OrderedDict[int, list[float]] = OrderedDict()
        self._override_profile: tuple[float, ...] | None = None

    @property
    def steps_per_bar(self) -> int:
        return self._steps_per_bar

    def set_steps_per_bar(self, steps_per_bar: int) -> None:
        """Update the step grid resolution (`16` for 4/4, `12` for 3/4 and 6/8)."""
        if steps_per_bar < 4:
            raise ValueError(f"steps_per_bar must be >= 4, got {steps_per_bar}")
        if steps_per_bar == self._steps_per_bar:
            return
        self._steps_per_bar = steps_per_bar
        self.reset()

    @property
    def bars_tracked(self) -> int:
        """Number of recent bars with recorded onsets currently held in the window."""
        return len(self._bars)

    def reset(self) -> None:
        """Clear accumulated bar onset grids and any recalled profile override."""
        self._bars.clear()
        self._override_profile = None

    def set_accent_profile(self, profile: Sequence[float] | None) -> None:
        """Install or clear a recalled accent profile (e.g. from `SessionMemory`)."""
        if profile is None:
            self._override_profile = None
            return
        if len(profile) != self._steps_per_bar:
            raise ValueError(
                f"accent profile must have {self._steps_per_bar} steps, got {len(profile)}"
            )
        self._override_profile = tuple(max(0.0, min(1.0, float(v))) for v in profile)

    def record_step(self, bar: int, step_index: int, strength: float) -> None:
        """Record an onset of `strength` (`0.0..1.0`) at 1-based `bar` and `0..steps_per_bar-1`."""
        if bar < 1:
            return
        step = step_index % self._steps_per_bar
        clamped = max(0.0, min(1.0, float(strength)))
        if clamped <= 0.0:
            return
        self._override_profile = None
        grid = self._bars.get(bar)
        if grid is None:
            grid = [0.0] * self._steps_per_bar
            self._bars[bar] = grid
        grid[step] = max(grid[step], clamped)
        self._prune()

    def record_bar_onsets(self, onset_mask: Sequence[float], *, bar: int | None = None) -> None:
        """Record a full 16-step bar onset mask."""
        if len(onset_mask) != self._steps_per_bar:
            raise ValueError(
                f"onset_mask must have {self._steps_per_bar} steps, got {len(onset_mask)}"
            )
        target_bar = (max(self._bars.keys()) + 1 if self._bars else 1) if bar is None else bar
        if target_bar < 1:
            raise ValueError(f"bar must be >= 1, got {target_bar}")
        self._override_profile = None
        self._bars[target_bar] = [max(0.0, min(1.0, float(v))) for v in onset_mask]
        self._prune()

    def record_onset(self, onset: OnsetEvent, clock: BeatClockSnapshot | None) -> None:
        """Quantize `onset` against `clock` onto the nearest 16th-note step and record it."""
        if clock is None or clock.frames_per_beat <= 0.0:
            return
        total_beats = (onset.frame_index - clock.anchor_frame) / clock.frames_per_beat
        if total_beats < -0.5:
            return
        steps_per_beat = self._steps_per_bar / clock.beats_per_bar
        nearest_global_step = max(0, round(total_beats * steps_per_beat))
        bar = nearest_global_step // self._steps_per_bar + 1
        step_index = nearest_global_step % self._steps_per_bar
        self.record_step(bar, step_index, onset.strength)

    def handle_event(self, event: MusicalEvent, clock: BeatClockSnapshot | None) -> None:
        """Consume a `MusicalEvent` and record `OnsetEvent`s when `clock` is running."""
        if isinstance(event, OnsetEvent):
            self.record_onset(event, clock)

    @property
    def accent_profile(self) -> tuple[float, ...]:
        """Return the normalized 16-step accent profile (`0.0..1.0` per step).

        Requires at least 2 bars of history (or an explicit recalled profile) before steps can
        exceed typical lock thresholds (`>= 0.65`), preventing a single bar of isolated hits from
        prematurely locking the kick pattern.
        """
        if self._override_profile is not None:
            return self._override_profile
        if not self._bars:
            return (0.0,) * self._steps_per_bar

        sorted_bars = sorted(self._bars.keys())[-self._window_bars :]
        num_bars = len(sorted_bars)
        weighted = [0.0] * self._steps_per_bar
        total_weight = 0.0
        for idx, bar_num in enumerate(sorted_bars):
            weight = self._decay ** (num_bars - 1 - idx)
            total_weight += weight
            grid = self._bars[bar_num]
            for s in range(self._steps_per_bar):
                weighted[s] += weight * grid[s]

        if total_weight <= 0.0:
            return (0.0,) * self._steps_per_bar

        avg = [v / total_weight for v in weighted]
        peak = max(avg)
        if peak <= 1e-6:
            return (0.0,) * self._steps_per_bar

        # A single bar gets maturity 0.5 so locking happens once an accent repeats across >= 2 bars.
        maturity = 0.5 if num_bars < 2 else 1.0
        strength_gate = min(1.0, peak / 0.25)
        scale = maturity * strength_gate / peak
        return tuple(min(1.0, v * scale) for v in avg)

    def _prune(self) -> None:
        if len(self._bars) <= self._window_bars:
            return
        max_bar = max(self._bars.keys())
        min_keep = max_bar - self._window_bars + 1
        to_remove = [b for b in self._bars if b < min_keep]
        for b in to_remove:
            del self._bars[b]
        while len(self._bars) > self._window_bars:
            oldest = next(iter(self._bars))
            del self._bars[oldest]


def _step_distance(step_a: int, step_b: int, steps_per_bar: int = STEPS_PER_BAR) -> int:
    """Minimum circular 16th-step distance within a bar so step 15 and step 0 respect gap rules."""
    diff = abs(step_a - step_b)
    return min(diff, steps_per_bar - diff)


def adapt_pattern_to_accents(
    pattern: DrumPattern,
    accent_profile: Sequence[float],
    config: DrumConfig,
) -> DrumPattern:
    """Adapt `pattern` to lock kicks and hi-hats to high-confidence player accents (§93, §147).

    Rules:
    1. Fills (`*_fill` or `fill_simple`) and disabled configs return `pattern` untouched.
    2. When no step in `accent_profile` reaches `config.accent_lock_threshold`, return `pattern`
       untouched.
    3. Lock kicks onto steps with `score >= config.accent_lock_threshold`, respecting
       `config.kick_min_step_gap` (minimum 16th-step refractory distance) so kicks never
       machine-gun on adjacent 16th subdivisions.
    4. Thin unaccented syncopated kicks from the base pattern (`score < 0.5 * threshold` on
       off-beat steps, or base kicks within `< kick_min_step_gap` of a locked player accent).
       Step 0 (beat-1 downbeat) is always preserved as a structural anchor if present in the base
       pattern or accented.
    5. Boost closed hi-hat velocity on accented steps (`+0.12 * score`, clamped to `<= 1.0`).
    """
    if not config.adaptive_accents_enabled:
        return pattern
    if pattern.name in ("fill_simple", "fill_simple_12") or pattern.name.endswith("_fill"):
        return pattern
    steps_count = len(pattern.steps)
    if len(accent_profile) != steps_count:
        return pattern

    threshold = config.accent_lock_threshold
    min_gap = config.kick_min_step_gap
    accented_steps = [s for s, score in enumerate(accent_profile) if score >= threshold]
    if not accented_steps:
        return pattern

    base_kick_velocities: dict[int, float] = {}
    for s, step_hits in enumerate(pattern.steps):
        for hit in step_hits:
            if hit.piece == DrumPiece.KICK:
                base_kick_velocities[s] = hit.velocity

    # Select locked kick steps in priority order: beat-1 anchor first (if accented or in base),
    # then strongest player accents, then remaining non-clashing structural base kicks.
    chosen_kicks: dict[int, float] = {}
    tier_base_vel = {"low": 0.72, "mid": 0.85, "high": 0.95}[pattern.tier]

    if 0 in base_kick_velocities or 0 in accented_steps:
        score_0 = float(accent_profile[0])
        vel_0 = max(
            base_kick_velocities.get(0, 0.0),
            min(1.0, tier_base_vel + 0.12 * score_0),
        )
        chosen_kicks[0] = vel_0

    # Sort player-accented steps by descending accent score, then step index for determinism.
    for s in sorted(accented_steps, key=lambda idx: (-float(accent_profile[idx]), idx)):
        if s in chosen_kicks:
            continue
        if all(_step_distance(s, k, steps_count) >= min_gap for k in chosen_kicks):
            score = float(accent_profile[s])
            base_vel = base_kick_velocities.get(s, 0.0)
            vel = max(base_vel, min(1.0, tier_base_vel + 0.12 * score))
            chosen_kicks[s] = vel

    # Keep non-syncopated base kicks (downbeats of beats 2/3/4, i.e. s % 4 == 0) only if they
    # don't clash with a locked accent; keep syncopated base kicks only if moderately supported.
    for s, base_vel in sorted(base_kick_velocities.items()):
        if s in chosen_kicks:
            continue
        score = float(accent_profile[s])
        is_beat_anchor = s % 4 == 0
        if not is_beat_anchor and score < 0.5 * threshold:
            # Thin unaccented syncopated kick so we lock to the player's pocket.
            continue
        if all(_step_distance(s, k, steps_count) >= min_gap for k in chosen_kicks):
            chosen_kicks[s] = base_vel

    new_steps: list[tuple[StepHit, ...]] = []
    for s in range(steps_count):
        existing_by_piece: dict[DrumPiece, StepHit] = {hit.piece: hit for hit in pattern.steps[s]}
        # Update or remove kick according to chosen_kicks.
        if s in chosen_kicks:
            existing_by_piece[DrumPiece.KICK] = StepHit(DrumPiece.KICK, chosen_kicks[s])
        else:
            existing_by_piece.pop(DrumPiece.KICK, None)

        # Boost hi-hat velocity on accented steps.
        score = float(accent_profile[s])
        if score >= threshold and DrumPiece.CLOSED_HAT in existing_by_piece:
            hat = existing_by_piece[DrumPiece.CLOSED_HAT]
            boosted = min(1.0, hat.velocity + 0.12 * score)
            existing_by_piece[DrumPiece.CLOSED_HAT] = StepHit(DrumPiece.CLOSED_HAT, boosted)

        ordered_hits = tuple(
            existing_by_piece[piece] for piece in DrumPiece if piece in existing_by_piece
        )
        new_steps.append(ordered_hits)

    adapted_steps = tuple(new_steps)
    if adapted_steps == pattern.steps:
        return pattern
    return DrumPattern(
        name=pattern.name,
        family=pattern.family,
        tier=pattern.tier,
        steps=adapted_steps,
        swing=pattern.swing,
    )

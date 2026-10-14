"""Band Director musical orchestration layer (Step 5.5, §72, §73, §125, §148).

Sits between `MusicalState` / `SessionMemory` and the rendering engines (`DrumEngine`,
`AccompanimentEngine`). Evaluates on beat/bar boundaries and publishes an immutable
`BandDirective` coordinating:
1. Riff transition fills/crashes and tighter AI root locking (`ai_role="tight_root"`).
2. Groove recall (`recalled_pattern_id` and `recalled_accent_mask`) when returning to a memorized
   riff in `SessionMemory`.
3. High-register solo space (`ai_role="open_space"`, `ai_gain_multiplier=0.7`) when the player
   moves up the neck (`bass_register == "high"`).
4. Phrase breathing (`"variation"` on 4-bar boundaries and `"fill"` on 8-bar boundaries of a
   steady riff).
5. Harmonic key anchoring (`target_key_pc`) when `harmonic_confidence >= 0.65`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from jammate.analysis.notes import NOTE_NAMES
from jammate.music.riff import SessionMemory
from jammate.music.state import MusicalState, SectionType

if TYPE_CHECKING:
    from jammate.music.styles import StylePreset

type DrumRole = Literal[
    "silent",
    "groove",
    "accent_lock",
    "fill",
    "crash_transition",
    "hold_sparse",
]
type AiRole = Literal[
    "silent",
    "tight_root",
    "rhythm_support",
    "open_space",
    "lead_fill",
]


@dataclass(frozen=True, slots=True)
class BandDirective:
    """Immutable musical decision directive published by `BandDirector` (§148)."""

    section: SectionType = "listening"
    drum_role: DrumRole = "silent"
    recalled_pattern_id: str | None = None
    recalled_accent_mask: tuple[float, ...] | None = None
    trigger_fill_now: bool = False
    trigger_crash_now: bool = False
    ai_role: AiRole = "silent"
    effective_follow_improvise: float = 0.35
    target_key_pc: int | None = None
    ai_gain_multiplier: float = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _key_name_to_pc(key_name: str | None) -> int | None:
    if key_name is None:
        return None
    try:
        return NOTE_NAMES.index(key_name)
    except ValueError:
        return None


class BandDirector:
    """Evaluates `MusicalState` and `SessionMemory` into a `BandDirective` (§72, §73, §148)."""

    def __init__(
        self,
        *,
        harmonic_anchor_confidence: float = 0.65,
        solo_gain_multiplier: float = 0.70,
        variation_every_bars: int = 4,
        fill_every_bars: int = 8,
    ) -> None:
        self._harmonic_anchor_confidence = harmonic_anchor_confidence
        self._solo_gain_multiplier = solo_gain_multiplier
        self._variation_every_bars = max(2, variation_every_bars)
        self._fill_every_bars = max(4, fill_every_bars)
        self._directive = BandDirective()

    @property
    def directive(self) -> BandDirective:
        return self._directive

    def reset(self) -> None:
        self._directive = BandDirective()

    def evaluate(
        self,
        state: MusicalState,
        memory: SessionMemory | None = None,
        *,
        preset: StylePreset | None = None,
        follow_improvise: float = 0.35,
    ) -> BandDirective:
        """Compute and publish the `BandDirective` for `state`."""
        user_fi = _clamp(float(follow_improvise), 0.0, 1.0)
        fill_cycle = preset.fill_every_bars if preset is not None else self._fill_every_bars
        variation_cycle = max(2, min(self._variation_every_bars, fill_cycle // 2))

        # 5. Harmonic anchoring: supply tonic pitch class when harmonic confidence is high.
        target_key_pc: int | None = None
        if (
            state.estimated_key is not None
            and state.harmonic_confidence >= self._harmonic_anchor_confidence
        ):
            target_key_pc = _key_name_to_pc(state.estimated_key)

        if not state.tempo_locked or state.bar is None:
            self._directive = BandDirective(
                section="listening",
                drum_role="silent",
                ai_role="silent",
                effective_follow_improvise=user_fi,
                target_key_pc=target_key_pc,
                ai_gain_multiplier=0.0,
            )
            return self._directive

        if not state.user_is_playing:
            self._directive = BandDirective(
                section="breakdown",
                drum_role="hold_sparse",
                ai_role="silent",
                effective_follow_improvise=_clamp(user_fi - 0.15, 0.0, 1.0),
                target_key_pc=target_key_pc,
                ai_gain_multiplier=0.5,
            )
            return self._directive

        # 2. Groove recall from SessionMemory when playing a known riff.
        recalled_pattern_id: str | None = None
        recalled_accent_mask: tuple[float, ...] | None = None
        if memory is not None and state.riff_id is not None:
            entry = memory.get(state.riff_id)
            if entry is not None:
                recalled_pattern_id = entry.preferred_drum_pattern_id
                if any(v > 0.0 for v in entry.accent_mask):
                    recalled_accent_mask = entry.accent_mask

        # Default active values before applying transition / register / breathing rules.
        section: SectionType = "intro" if state.riff_id is None else "groove"
        drum_role: DrumRole = (
            "accent_lock"
            if (recalled_accent_mask is not None or state.riff_iteration >= 2)
            else "groove"
        )
        trigger_fill_now = False
        trigger_crash_now = False
        ai_role: AiRole = "rhythm_support"
        effective_fi = user_fi
        ai_gain_multiplier = 1.0

        # 4. Phrase breathing on steady riffs (4-bar variation, 8-bar fill).
        if state.riff_id is not None and not state.riff_changed and state.riff_iteration > 0:
            if state.riff_iteration % fill_cycle == 0:
                section = "fill"
                drum_role = "fill"
                trigger_fill_now = True
                ai_role = "lead_fill"
                effective_fi = _clamp(user_fi + 0.15, 0.0, 1.0)
            elif state.riff_iteration % variation_cycle == 0:
                section = "variation"
                drum_role = "accent_lock"
                effective_fi = _clamp(user_fi + 0.10, 0.0, 1.0)

        # 1. Riff transition response: fill/crash on the boundary + tighter root support.
        if state.riff_changed:
            section = "fill"
            drum_role = "crash_transition"
            trigger_fill_now = True
            trigger_crash_now = True
            ai_role = "tight_root"
            effective_fi = _clamp(user_fi - 0.25, 0.0, 1.0)

        # 3. Solo / high-register space: thin AI accompaniment so high solos aren't buried.
        if state.bass_register == "high":
            ai_role = "open_space"
            ai_gain_multiplier = self._solo_gain_multiplier
            if not state.riff_changed:
                effective_fi = _clamp(user_fi + 0.15, 0.0, 1.0)

        self._directive = BandDirective(
            section=section,
            drum_role=drum_role,
            recalled_pattern_id=recalled_pattern_id,
            recalled_accent_mask=recalled_accent_mask,
            trigger_fill_now=trigger_fill_now,
            trigger_crash_now=trigger_crash_now,
            ai_role=ai_role,
            effective_follow_improvise=effective_fi,
            target_key_pc=target_key_pc,
            ai_gain_multiplier=ai_gain_multiplier,
        )
        return self._directive

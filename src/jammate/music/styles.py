"""Style presets, Follow-Improvise mapping, and musical AI entry (Step 4.3, §§30, 74-76)."""

from __future__ import annotations

from dataclasses import dataclass

from jammate.config import AccompanimentConfig, StyleFamily
from jammate.drums.engine import DrummerState
from jammate.drums.patterns import DEFAULT_BLUES_SWING
from jammate.music.state import MusicalState


@dataclass(frozen=True, slots=True)
class StylePreset:
    """Structured configuration for a musical style preset (§75, §76, D7)."""

    name: StyleFamily
    label: str
    prompt: str
    drum_family: StyleFamily
    fill_every_bars: int
    density: float
    swing: float
    musiccoca_cfg_range: tuple[float, float]
    notes_cfg_range: tuple[float, float]
    drums_cfg: float
    temperature_range: tuple[float, float]
    top_k_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FollowImproviseMapping:
    """Model and drummer parameters derived from Follow-Improvise and intensity (§74)."""

    follow_improvise: float
    musiccoca_cfg: float
    notes_cfg: float
    drums_cfg: float
    temperature: float
    top_k: int
    fill_every_bars: int

    def cfg_scales(self) -> dict[str, float]:
        return {
            "musiccoca": self.musiccoca_cfg,
            "notes": self.notes_cfg,
            "drums": self.drums_cfg,
        }


STYLE_PRESETS: dict[StyleFamily, StylePreset] = {
    "rock": StylePreset(
        name="rock",
        label="Rock",
        prompt="tight rock bass, crunch rhythm guitar and warm organ accompaniment",
        drum_family="rock",
        fill_every_bars=8,
        density=0.55,
        swing=0.0,
        musiccoca_cfg_range=(2.8, 3.8),
        notes_cfg_range=(4.0, 0.8),
        drums_cfg=1.0,
        temperature_range=(1.10, 1.45),
        top_k_range=(28, 56),
    ),
    "stoner_rock": StylePreset(
        name="stoner_rock",
        label="Stoner Rock",
        prompt="fuzzy heavy slow dark stoner rock bass and desert rhythm guitar",
        drum_family="stoner_rock",
        fill_every_bars=8,
        density=0.50,
        swing=0.0,
        musiccoca_cfg_range=(3.2, 4.2),
        notes_cfg_range=(4.2, 1.0),
        drums_cfg=1.0,
        temperature_range=(1.05, 1.40),
        top_k_range=(24, 52),
    ),
    "blues": StylePreset(
        name="blues",
        label="Blues",
        prompt="warm shuffle blues bass, vintage electric piano and rhythm guitar",
        drum_family="blues",
        fill_every_bars=8,
        density=0.45,
        swing=DEFAULT_BLUES_SWING,
        musiccoca_cfg_range=(2.8, 3.6),
        notes_cfg_range=(4.0, 0.9),
        drums_cfg=1.0,
        temperature_range=(1.05, 1.40),
        top_k_range=(24, 52),
    ),
    "funk": StylePreset(
        name="funk",
        label="Funk",
        prompt="syncopated funk bass, clavinet keys and clean rhythm guitar groove",
        drum_family="funk",
        fill_every_bars=8,
        density=0.70,
        swing=0.0,
        musiccoca_cfg_range=(3.0, 4.0),
        notes_cfg_range=(3.8, 0.8),
        drums_cfg=1.0,
        temperature_range=(1.10, 1.50),
        top_k_range=(28, 60),
    ),
    "metal": StylePreset(
        name="metal",
        label="Metal",
        prompt="tight palm-muted heavy metal bass and distorted rhythm guitar",
        drum_family="metal",
        fill_every_bars=8,
        density=0.80,
        swing=0.0,
        musiccoca_cfg_range=(3.4, 4.4),
        notes_cfg_range=(4.4, 1.0),
        drums_cfg=1.0,
        temperature_range=(1.05, 1.38),
        top_k_range=(24, 48),
    ),
}


def get_style_preset(style: StyleFamily | str) -> StylePreset:
    """Return the `StylePreset` for `style` (raises `KeyError` if unknown)."""
    if style not in STYLE_PRESETS:
        raise KeyError(f"Unknown style preset {style!r}; expected one of {tuple(STYLE_PRESETS)}")
    return STYLE_PRESETS[style]  # type: ignore[index]


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def map_follow_improvise(
    preset: StylePreset,
    follow_improvise: float,
    intensity: float = 0.5,
) -> FollowImproviseMapping:
    """Map product parameter `follow_improvise` (`0.0..1.0`) and `intensity` (`0.0..1.0`) (§74).

    At `0.0` (Follow): high `notes_cfg`, lower temperature, smaller `top_k`, standard fill rate.
    At `1.0` (Improvise): lower `notes_cfg`, higher temperature, wider `top_k`, more frequent fills.
    """
    if not 0.0 <= follow_improvise <= 1.0:
        raise ValueError("follow_improvise must be between 0.0 and 1.0")
    t = float(follow_improvise)
    inten = _clamp(float(intensity), 0.0, 1.0)

    raw_musiccoca = _lerp(preset.musiccoca_cfg_range[0], preset.musiccoca_cfg_range[1], t)
    musiccoca_cfg = _clamp(raw_musiccoca + 0.5 * (inten - 0.5), -1.0, 7.0)
    notes_cfg = _clamp(_lerp(preset.notes_cfg_range[0], preset.notes_cfg_range[1], t), -1.0, 7.0)
    raw_temp = _lerp(preset.temperature_range[0], preset.temperature_range[1], t)
    temperature = _clamp(raw_temp + 0.06 * (inten - 0.5), 0.5, 2.0)
    top_k = round(_lerp(float(preset.top_k_range[0]), float(preset.top_k_range[1]), t))

    fill_every_bars = preset.fill_every_bars
    if t >= 0.75:
        fill_every_bars = max(4, preset.fill_every_bars // 2)

    return FollowImproviseMapping(
        follow_improvise=t,
        musiccoca_cfg=musiccoca_cfg,
        notes_cfg=notes_cfg,
        drums_cfg=preset.drums_cfg,
        temperature=temperature,
        top_k=top_k,
        fill_every_bars=fill_every_bars,
    )


class AiEntryController:
    """Coordinates musical AI entry after drums join and gates on long silence (Step 4.3, §30).

    - Waits until the drummer is in `PLAYING` (or `HOLDING`) and `bar >= drummer_entry_bar +
      entry_delay_bars`.
    - Gates out (`active = False`, `gate_gain = 0.0`) if the player is silent for at least
      `silence_gate_bars` bars, and re-enters on the next bar boundary when playing resumes.
    """

    def __init__(self, config: AccompanimentConfig, *, beats_per_bar: int = 4) -> None:
        self._config = config
        self._beats_per_bar = beats_per_bar
        self._active = False
        self._entry_bar: int | None = None
        self._silent_since_bar_pos: float | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def entry_bar(self) -> int | None:
        return self._entry_bar

    @property
    def gate_gain(self) -> float:
        return 1.0 if self._active else 0.0

    def reset(self) -> None:
        self._active = False
        self._entry_bar = None
        self._silent_since_bar_pos = None

    def update(
        self,
        musical_state: MusicalState,
        drummer_state: DrummerState,
        drummer_entry_bar: int | None,
    ) -> bool:
        """Update AI entry/gate state and return True when AI audio should be audible."""
        if (
            drummer_state not in (DrummerState.PLAYING, DrummerState.HOLDING)
            or drummer_entry_bar is None
            or musical_state.bar is None
            or musical_state.beat is None
        ):
            self.reset()
            return False

        bar = musical_state.bar
        beat = musical_state.beat
        phase = musical_state.beat_phase or 0.0
        bar_pos = (bar - 1) + ((beat - 1) + phase) / self._beats_per_bar

        if self._entry_bar is None:
            self._entry_bar = drummer_entry_bar + self._config.entry_delay_bars

        if not self._active:
            if bar >= self._entry_bar and musical_state.user_is_playing:
                self._active = True
                self._silent_since_bar_pos = None
            return self._active

        # Currently active: monitor silence duration.
        if musical_state.user_is_playing:
            self._silent_since_bar_pos = None
        else:
            if self._silent_since_bar_pos is None:
                self._silent_since_bar_pos = bar_pos
            silent_bars = bar_pos - self._silent_since_bar_pos
            if silent_bars >= self._config.silence_gate_bars:
                self._active = False
                self._entry_bar = bar + 1
                self._silent_since_bar_pos = None

        return self._active

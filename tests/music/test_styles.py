"""Tests for StylePreset, map_follow_improvise, and AiEntryController (Step 4.3)."""

from __future__ import annotations

import pytest

from jammate.ai.null import NullAccompaniment
from jammate.config import AccompanimentConfig, JamMateConfig
from jammate.drums.engine import DrummerState
from jammate.music.events import PlayingState
from jammate.music.state import MusicalState
from jammate.music.styles import (
    STYLE_PRESETS,
    AiEntryController,
    get_style_preset,
    map_follow_improvise,
)
from jammate.session import JamSession


def _state(
    *,
    bar: int | None = 1,
    beat: int | None = 1,
    phase: float | None = 0.0,
    playing: bool = True,
    intensity: float = 0.5,
) -> MusicalState:
    return MusicalState(
        current_note=40 if playing else None,
        recent_notes=(40,) if playing else (),
        bpm=100.0,
        tempo_confidence=0.85,
        tempo_locked=True,
        beat=beat,
        bar=bar,
        beat_phase=phase,
        intensity=intensity,
        playing_state=PlayingState.PLAYING if playing else PlayingState.STOPPED,
    )


def test_all_five_style_presets_are_defined_and_valid() -> None:
    expected = {"rock", "stoner_rock", "blues", "funk", "metal"}
    assert set(STYLE_PRESETS.keys()) == expected

    for name, preset in STYLE_PRESETS.items():
        assert get_style_preset(name) == preset
        assert preset.prompt
        assert len(preset.prompt.split()) <= 16  # short semantic prompt (§76)
        assert -1.0 <= preset.musiccoca_cfg_range[0] <= preset.musiccoca_cfg_range[1] <= 7.0
        assert -1.0 <= preset.notes_cfg_range[1] <= preset.notes_cfg_range[0] <= 7.0
        assert 0.5 <= preset.temperature_range[0] <= preset.temperature_range[1] <= 2.0
        assert 1 <= preset.top_k_range[0] <= preset.top_k_range[1]

    assert get_style_preset("blues").swing == pytest.approx(1.0 / 3.0)
    assert get_style_preset("rock").swing == 0.0

    with pytest.raises(KeyError):
        get_style_preset("polka")


def test_map_follow_improvise_steers_cfg_temperature_top_k_and_fills() -> None:
    preset = get_style_preset("stoner_rock")
    follow = map_follow_improvise(preset, 0.0, intensity=0.5)
    mid = map_follow_improvise(preset, 0.5, intensity=0.5)
    improvise = map_follow_improvise(preset, 1.0, intensity=0.5)

    # Higher Follow -> stronger note conditioning; higher Improvise -> looser note conditioning.
    assert follow.notes_cfg > mid.notes_cfg > improvise.notes_cfg
    # Higher Improvise -> higher temperature and wider top_k.
    assert follow.temperature < mid.temperature < improvise.temperature
    assert follow.top_k < mid.top_k < improvise.top_k
    # High Improvise (>= 0.75) doubles drum fill frequency (every 4 bars instead of 8).
    assert follow.fill_every_bars == 8
    assert improvise.fill_every_bars == 4

    # Higher playing intensity nudges style CFG and temperature upward.
    soft = map_follow_improvise(preset, 0.5, intensity=0.1)
    loud = map_follow_improvise(preset, 0.5, intensity=0.9)
    assert loud.musiccoca_cfg > soft.musiccoca_cfg
    assert loud.temperature > soft.temperature

    with pytest.raises(ValueError):
        map_follow_improvise(preset, 1.2)


def test_ai_entry_controller_joins_after_drums_and_gates_on_long_silence() -> None:
    controller = AiEntryController(
        AccompanimentConfig(entry_delay_bars=1, silence_gate_bars=4.0),
    )

    # While drummer is LISTENING or WAITING_FOR_BAR, AI stays gated out.
    assert not controller.update(_state(bar=1, beat=1), DrummerState.LISTENING, None)
    assert controller.gate_gain == 0.0
    assert not controller.update(_state(bar=1, beat=4), DrummerState.WAITING_FOR_BAR, 2)

    # Drummer enters on bar 2: with entry_delay_bars=1, AI waits through bar 2 and joins on bar 3!
    assert not controller.update(_state(bar=2, beat=1), DrummerState.PLAYING, 2)
    assert controller.entry_bar == 3
    assert controller.gate_gain == 0.0

    assert controller.update(_state(bar=3, beat=1), DrummerState.PLAYING, 2)
    assert controller.is_active is True
    assert controller.gate_gain == 1.0

    # 2 bars of silence (bar 4 beat 1 -> bar 6 beat 1 = 2.0 bars): AI stays active (< 4 bars).
    controller.update(_state(bar=4, beat=1, playing=False), DrummerState.PLAYING, 2)
    assert controller.update(_state(bar=6, beat=1, playing=False), DrummerState.HOLDING, 2)

    # 4 bars of silence (bar 8 beat 1): AI gates out (`gate_gain == 0.0`) and arms for next bar.
    assert not controller.update(_state(bar=8, beat=1, playing=False), DrummerState.HOLDING, 2)
    assert controller.gate_gain == 0.0

    # Player resumes in bar 8 beat 3 -> AI re-enters on bar 9 boundary!
    assert not controller.update(_state(bar=8, beat=3, playing=True), DrummerState.PLAYING, 2)
    assert controller.update(_state(bar=9, beat=1, playing=True), DrummerState.PLAYING, 2)
    assert controller.gate_gain == 1.0


def test_session_set_style_and_set_follow_improvise_update_drums_and_ai(fake_stream) -> None:
    del fake_stream
    null_ai = NullAccompaniment()
    session = JamSession(
        JamMateConfig(),
        accompaniment_factory=lambda _cfg, _sr: null_ai,
    )

    session.set_style("blues")
    assert session.style == "blues"
    assert session.drums.style == "blues"
    assert session.ai.stats().style == "blues"

    session.set_follow_improvise(0.85)
    assert session.follow_improvise == pytest.approx(0.85)
    assert session.ai.stats().follow_improvise == pytest.approx(0.85)
    assert session.drums.controller.fill_every_bars == 4

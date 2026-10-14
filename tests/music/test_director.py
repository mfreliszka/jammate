"""Unit and Milestone 5 integration tests for `BandDirector` (Step 5.5, §72, §73, §148, §152)."""

from __future__ import annotations

import numpy as np
from scripts.render_jam import render_jam_offline

from jammate.ai.base import NoteTracker
from jammate.ai.mrt import Mrt2Accompaniment
from jammate.ai.null import NullAccompaniment
from jammate.config import (
    AccompanimentConfig,
    AnalysisConfig,
    DrumConfig,
    JamMateConfig,
    TempoConfig,
)
from jammate.drums.adaptive import AccentTracker
from jammate.drums.engine import DrummerController
from jammate.music.director import BandDirector
from jammate.music.events import (
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SilenceEvent,
)
from jammate.music.riff import BarFingerprint, RiffDetector, RiffMemoryEntry, SessionMemory
from jammate.music.state import MusicalState
from jammate.music.styles import get_style_preset
from tests.signals import GUITAR_HARMONIC_AMPLITUDES, riff


def _playing_state(
    *,
    bar: int = 3,
    riff_id: str | None = "riff_A",
    riff_iteration: int = 2,
    riff_changed: bool = False,
    bass_register: str = "low",
    estimated_key: str | None = "E",
    estimated_mode: str | None = "minor",
    harmonic_confidence: float = 0.80,
    user_is_playing: bool = True,
) -> MusicalState:
    return MusicalState(
        frame_index=bar * 96_000,
        bpm=120.0,
        tempo_confidence=0.90,
        tempo_locked=True,
        bar=bar,
        beat=1,
        beat_phase=0.0,
        current_note=40,
        intensity=0.65,
        playing_state=PlayingState.PLAYING if user_is_playing else PlayingState.STOPPED,
        estimated_key=estimated_key,
        estimated_mode=estimated_mode,  # type: ignore[arg-type]
        harmonic_confidence=harmonic_confidence,
        bass_register=bass_register,  # type: ignore[arg-type]
        riff_id=riff_id,
        riff_iteration=riff_iteration,
        riff_changed=riff_changed,
        section="groove",
    )


def test_riff_changed_triggers_transition_fill_crash_and_tight_root() -> None:
    """Rule 1: `riff_changed=True` triggers fill + crash and `ai_role='tight_root'`."""
    director = BandDirector()
    state = _playing_state(bar=5, riff_id="riff_B", riff_iteration=1, riff_changed=True)
    directive = director.evaluate(state, follow_improvise=0.50)

    assert directive.section == "fill"
    assert directive.drum_role == "crash_transition"
    assert directive.trigger_fill_now is True
    assert directive.trigger_crash_now is True
    assert directive.ai_role == "tight_root"
    assert directive.effective_follow_improvise < 0.50

    controller = DrummerController(DrumConfig(), manual_bpm_active=True)
    controller.update(state)
    controller.apply_directive(directive, current_bar=5)
    assert 5 in controller.crash_bars


def test_a_b_a_groove_recall_restores_pattern_and_accent_mask() -> None:
    """Rule 2: Returning to Riff A (`A -> B -> A`) restores its stored pattern and accent mask."""
    detector = RiffDetector()
    mask_a = (1.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.90, 0.0, 1.0, 0.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0)
    fp_a = BarFingerprint(
        bar_index=1,
        onset_mask=mask_a,
        pitch_classes=(0, 0, 0, 3, 3, 3, 5, 5, 0, 0, 0, 0, 7, 7, 7, 7),
        root_pc=4,
        mean_intensity=0.75,
    )
    mask_b = (1.0, 0.0, 0.80, 0.0, 1.0, 0.0, 0.80, 0.0, 1.0, 0.0, 0.80, 0.0, 1.0, 0.0, 0.80, 0.0)
    fp_b = BarFingerprint(
        bar_index=3,
        onset_mask=mask_b,
        pitch_classes=(0, 0, 2, 2, 3, 3, 5, 5, 0, 0, 2, 2, 3, 3, 5, 5),
        root_pc=9,
        mean_intensity=0.45,
    )

    # Bars 1-2 lock Riff A with preferred pattern "stoner_rock_high".
    detector.observe_bar(fp_a, preferred_drum_pattern_id="stoner_rock_high")
    res_a2 = detector.observe_bar(
        BarFingerprint(
            bar_index=2,
            onset_mask=mask_a,
            pitch_classes=fp_a.pitch_classes,
            root_pc=fp_a.root_pc,
            mean_intensity=0.75,
        ),
        preferred_drum_pattern_id="stoner_rock_high",
    )
    assert res_a2.riff_id == "riff_A"

    # Bars 3-4 transition to Riff B with preferred pattern "stoner_rock_low".
    res_b1 = detector.observe_bar(fp_b, preferred_drum_pattern_id="stoner_rock_low")
    assert res_b1.riff_id == "riff_B"
    assert res_b1.riff_changed is True
    detector.observe_bar(
        BarFingerprint(
            bar_index=4,
            onset_mask=mask_b,
            pitch_classes=fp_b.pitch_classes,
            root_pc=fp_b.root_pc,
            mean_intensity=0.45,
        ),
        preferred_drum_pattern_id="stoner_rock_low",
    )

    # Bar 5 returns to Riff A (`A -> B -> A`).
    res_a_recall = detector.observe_bar(
        BarFingerprint(
            bar_index=5,
            onset_mask=mask_a,
            pitch_classes=fp_a.pitch_classes,
            root_pc=fp_a.root_pc,
            mean_intensity=0.75,
        ),
        preferred_drum_pattern_id="stoner_rock_low",
    )
    assert res_a_recall.riff_id == "riff_A"
    assert res_a_recall.riff_changed is True

    director = BandDirector()
    state_bar5 = _playing_state(
        bar=5,
        riff_id=res_a_recall.riff_id,
        riff_iteration=res_a_recall.riff_iteration,
        riff_changed=False,
    )
    directive = director.evaluate(state_bar5, detector.memory)
    assert directive.recalled_pattern_id == "stoner_rock_high"
    assert directive.recalled_accent_mask is not None
    assert directive.recalled_accent_mask[3] > 0.7

    controller = DrummerController(DrumConfig(), manual_bpm_active=True)
    controller.update(state_bar5)
    controller.apply_directive(directive, current_bar=5)
    pat = controller.pattern_for_bar(5)
    assert pat is not None
    assert pat.name == "stoner_rock_high"
    assert controller.accent_profile[3] > 0.7

    tracker = AccentTracker()
    tracker.set_accent_profile(directive.recalled_accent_mask)
    assert tracker.accent_profile[3] > 0.7


class _FakeMrt2System:
    def embed_style(self, text: str) -> np.ndarray:
        del text
        return np.zeros(8, dtype=np.float32)

    def init_state(self) -> object:
        return object()

    def generate(self, **kwargs: object) -> tuple[object, object]:
        del kwargs
        raise AssertionError("generate() should not be called in unit test")


def test_high_register_solo_switches_ai_to_open_space_and_thins_gain() -> None:
    """Rule 3: `bass_register == 'high'` switches `ai_role='open_space'` and reduces AI gain."""
    director = BandDirector(solo_gain_multiplier=0.70)
    state = _playing_state(bar=6, bass_register="high", riff_changed=False)
    directive = director.evaluate(state, follow_improvise=0.40)

    assert directive.ai_role == "open_space"
    assert abs(directive.ai_gain_multiplier - 0.70) < 1e-6
    assert abs(directive.effective_follow_improvise - 0.55) < 1e-6

    mrt = Mrt2Accompaniment(
        AccompanimentConfig(enabled=True, follow_improvise=0.40),
        sample_rate_hz=48_000,
        system_factory=lambda _size: _FakeMrt2System(),  # type: ignore[arg-type, return-value]
        preload_presets=False,
    )
    mrt.apply_directive(directive)
    assert abs(mrt.ai_gain_multiplier - 0.70) < 1e-6
    assert abs(mrt.effective_follow_improvise - 0.55) < 1e-6


def test_phrase_breathing_schedules_variation_at_4_bars_and_fill_at_8_bars() -> None:
    """Rule 4: Steady riffs breathe with 4-bar `'variation'` and 8-bar `'fill'` directives."""
    director = BandDirector(variation_every_bars=4, fill_every_bars=8)
    preset = get_style_preset("stoner_rock")

    # Iteration 2: standard groove.
    d2 = director.evaluate(
        _playing_state(bar=2, riff_iteration=2, riff_changed=False),
        preset=preset,
        follow_improvise=0.35,
    )
    assert d2.section == "groove"
    assert d2.trigger_fill_now is False

    # Iteration 4: 4-bar variation.
    d4 = director.evaluate(
        _playing_state(bar=4, riff_iteration=4, riff_changed=False),
        preset=preset,
        follow_improvise=0.35,
    )
    assert d4.section == "variation"
    assert d4.drum_role == "accent_lock"
    assert d4.trigger_fill_now is False
    assert d4.effective_follow_improvise > 0.35

    # Iteration 8: 8-bar phrase fill.
    d8 = director.evaluate(
        _playing_state(bar=8, riff_iteration=8, riff_changed=False),
        preset=preset,
        follow_improvise=0.35,
    )
    assert d8.section == "fill"
    assert d8.drum_role == "fill"
    assert d8.trigger_fill_now is True
    assert d8.ai_role == "lead_fill"


def test_harmonic_anchoring_supplies_tonic_during_brief_phrase_rests() -> None:
    """Rule 5: `harmonic_confidence >= 0.65` holds tonic root in `NoteTracker` on short rests."""
    director = BandDirector(harmonic_anchor_confidence=0.65)
    memory = SessionMemory()
    memory.store(
        RiffMemoryEntry(
            riff_id="riff_A",
            fingerprint=BarFingerprint(
                bar_index=1,
                onset_mask=(1.0,) + (0.0,) * 15,
                pitch_classes=(0,) + (-1,) * 15,
                root_pc=4,
                mean_intensity=0.6,
            ),
        )
    )
    state_high_conf = _playing_state(
        estimated_key="E",
        estimated_mode="minor",
        harmonic_confidence=0.82,
        user_is_playing=True,
    )
    directive = director.evaluate(state_high_conf, memory)
    assert directive.target_key_pc == 4  # E = pitch class 4

    tracker = NoteTracker(NullAccompaniment())
    tracker.set_harmonic_anchor(directive.target_key_pc, user_is_playing=True)

    # Play E2 (MIDI 40) then release it with a brief inter-note SilenceEvent inside a phrase.
    tracker.handle_event(
        PitchEvent(
            frame_index=0,
            timestamp_seconds=0.0,
            midi_note=40,
            frequency_hz=82.41,
            confidence=0.95,
        )
    )
    _ = tracker.consume_pianoroll()
    tracker.handle_event(
        SilenceEvent(
            frame_index=4_800,
            timestamp_seconds=0.10,
        )
    )
    # While user_is_playing=True and target_key_pc=4, the tonic E2 (MIDI 40) remains held (1).
    roll_during_rest = tracker.consume_pianoroll()
    assert int(roll_during_rest[40]) == 1

    # When the phrase ends (PlayingStateUpdate STOPPED / user_is_playing=False), no notes hang.
    tracker.handle_event(
        PlayingStateUpdate(
            frame_index=24_000,
            timestamp_seconds=0.50,
            state=PlayingState.STOPPED,
        )
    )
    roll_after_silence = tracker.consume_pianoroll()
    assert int(np.sum(roll_after_silence)) == 0


def test_milestone_5_multi_section_guitar_jam_end_to_end() -> None:
    """Milestone 5 (§152): multi-section guitar jam (`Riff A -> Riff B -> High Solo -> Riff A`)."""
    sr = 48_000
    bpm = 120.0

    # 3 bars of Riff A (low E minor guitar riff: E2, G2, A2, E2)
    riff_a_notes: list[int | None] = [40, 43, 45, 40] * 3
    sig_a1 = riff(
        riff_a_notes,
        bpm=bpm,
        step_beats=1.0,
        sample_rate_hz=sr,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    # 2 bars of Riff B (distinct mid-register eighth-note groove on B2, D3, E3)
    riff_b_notes: list[int | None] = [47, 50, 52, 50, 47, 50, 52, 55] * 2
    sig_b = riff(
        riff_b_notes,
        bpm=bpm,
        step_beats=0.5,
        sample_rate_hz=sr,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    # 2 bars of High Solo (high guitar register >= G3/E4: E4, G4, A4, B4)
    solo_notes: list[int | None] = [64, 67, 69, 71] * 2
    sig_solo = riff(
        solo_notes,
        bpm=bpm,
        step_beats=1.0,
        sample_rate_hz=sr,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    # 2 bars returning to Riff A (`E2, G2, A2, E2`)
    sig_a2 = riff(
        [40, 43, 45, 40] * 2,
        bpm=bpm,
        step_beats=1.0,
        sample_rate_hz=sr,
        harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES,
    )

    full_audio = np.concatenate([sig_a1.samples, sig_b.samples, sig_solo.samples, sig_a2.samples])
    cfg = JamMateConfig(
        analysis=AnalysisConfig(
            instrument="guitar",
            tempo=TempoConfig(manual_bpm=bpm),
        ),
        drums=DrumConfig(adaptive_accents_enabled=True, humanize_enabled=True),
    )

    output, summary = render_jam_offline(full_audio, cfg)
    assert output.shape == (len(full_audio), 2)
    assert summary.tempo_locked is True
    assert summary.drummer_entered is True
    assert summary.events_scheduled > 0
    assert summary.dropped_events == 0
    assert summary.riffs_memorized >= 2
    # Riff A was locked first and recalled in the final section (`A -> ... -> A`).
    assert len(summary.riffs_visited) >= 2
    assert summary.riffs_visited[0] == "riff_A"
    assert summary.riffs_visited[-1] == "riff_A"
    # High solo section triggered "open_space" AI role and riff transitions triggered "fill".
    assert "open_space" in summary.ai_roles_visited
    assert "fill" in summary.sections_visited
    assert len(summary.fill_bars) >= 1

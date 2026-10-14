"""Unit tests for bar-quantized riff detection and session memory (Step 5.3, §41, §98, §146)."""

from __future__ import annotations

import pytest

from jammate.analysis.notes import midi_to_hz, name_to_midi
from jammate.config import BeatClockConfig, RiffConfig
from jammate.music.beat_clock import BeatClock
from jammate.music.events import (
    IntensityUpdate,
    OnsetEvent,
    PitchEvent,
    SampleFrame,
    frames_to_seconds,
    seconds_to_frames,
)
from jammate.music.riff import (
    BarFingerprint,
    RiffDetector,
    SessionMemory,
    fingerprint_similarity,
    quantize_bar,
)
from jammate.music.state import MusicalStateManager

SR = 48_000
BPM = 120.0
BAR_FRAMES = seconds_to_frames(4.0 * 60.0 / BPM, SR)  # 96_000 frames (2.0 s)
STEP_FRAMES = BAR_FRAMES // 16  # 6_000 frames (125 ms)


def _make_bar_fp(
    bar_index: int,
    step_notes: list[tuple[int, str]],
    *,
    jitter_ms_by_step: dict[int, float] | None = None,
    stabilizer_delay_ms: float = 0.0,
    intensity: float = 0.7,
) -> BarFingerprint:
    """Build a `BarFingerprint` for `bar_index` from `(step_0_to_15, note_name)` pairs."""
    bar_start = (bar_index - 1) * BAR_FRAMES
    bar_end = bar_index * BAR_FRAMES
    onsets: list[OnsetEvent] = []
    notes: list[PitchEvent] = []
    jitter_map = jitter_ms_by_step or {}
    delay_frames = seconds_to_frames(stabilizer_delay_ms / 1000.0, SR)

    for step_idx, note_name in step_notes:
        jitter_frames = seconds_to_frames(jitter_map.get(step_idx, 0.0) / 1000.0, SR)
        onset_frame = max(0, bar_start + step_idx * STEP_FRAMES + jitter_frames)
        note_frame = onset_frame + delay_frames
        midi = name_to_midi(note_name)
        onsets.append(
            OnsetEvent(
                frame_index=onset_frame,
                timestamp_seconds=frames_to_seconds(onset_frame, SR),
                strength=0.8,
            )
        )
        notes.append(
            PitchEvent(
                frame_index=note_frame,
                timestamp_seconds=frames_to_seconds(note_frame, SR),
                midi_note=midi,
                frequency_hz=midi_to_hz(midi),
                confidence=0.92,
            )
        )

    return quantize_bar(
        bar_index,
        bar_start,
        bar_end,
        onsets=onsets,
        notes=notes,
        mean_intensity=intensity,
    )


def test_bar_fingerprint_quantizes_16_steps_and_relative_pitch_contour() -> None:
    # Riff A: E2 (step 0), G2 (step 4), A2 (step 8), E2 (step 12)
    fp = _make_bar_fp(1, [(0, "E2"), (4, "G2"), (8, "A2"), (12, "E2")], stabilizer_delay_ms=65.0)
    assert fp.bar_index == 1
    assert fp.steps_per_bar == 16
    assert not fp.is_silent
    assert fp.root_pc == name_to_midi("E2") % 12
    assert fp.onset_mask[0] == pytest.approx(1.0)
    assert fp.onset_mask[4] == pytest.approx(1.0)
    assert fp.onset_mask[8] == pytest.approx(1.0)
    assert fp.onset_mask[12] == pytest.approx(1.0)
    assert fp.onset_mask[2] == pytest.approx(0.0)
    # Relative pitch classes: E=0, G=3, A=5, E=0
    assert fp.pitch_classes[0] == 0
    assert fp.pitch_classes[4] == 3
    assert fp.pitch_classes[8] == 5
    assert fp.pitch_classes[12] == 0


def test_repeating_1bar_riff_locks_riff_a_and_increments_iteration() -> None:
    detector = RiffDetector()
    riff_a_notes = [(0, "E2"), (2, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]

    r1 = detector.observe_bar(_make_bar_fp(1, riff_a_notes))
    assert r1.riff_id is None
    assert r1.riff_iteration == 0
    assert not r1.riff_changed

    r2 = detector.observe_bar(_make_bar_fp(2, riff_a_notes))
    assert r2.riff_id == "riff_A"
    assert r2.riff_iteration == 2
    assert not r2.riff_changed
    assert r2.similarity >= 0.95

    r3 = detector.observe_bar(_make_bar_fp(3, riff_a_notes))
    assert r3.riff_id == "riff_A"
    assert r3.riff_iteration == 3
    assert not r3.riff_changed

    r4 = detector.observe_bar(_make_bar_fp(4, riff_a_notes))
    assert r4.riff_id == "riff_A"
    assert r4.riff_iteration == 4
    assert len(detector.memory) == 1


def test_repeating_2bar_riff_locks_riff_a_and_increments_iteration() -> None:
    detector = RiffDetector()
    bar_a1 = [(0, "E2"), (4, "G2"), (8, "A2"), (12, "B2")]
    bar_a2 = [(0, "D3"), (6, "B2"), (10, "A2"), (14, "G2")]

    # First 2-bar cycle (bars 1 & 2)
    r1 = detector.observe_bar(_make_bar_fp(1, bar_a1))
    r2 = detector.observe_bar(_make_bar_fp(2, bar_a2))
    assert r1.riff_id is None
    assert r2.riff_id is None

    # Second 2-bar cycle (bars 3 & 4) locks riff_A on bar 4
    r3 = detector.observe_bar(_make_bar_fp(3, bar_a1))
    assert r3.riff_id is None
    r4 = detector.observe_bar(_make_bar_fp(4, bar_a2))
    assert r4.riff_id == "riff_A"
    assert r4.riff_iteration == 2
    assert r4.recalled_entry is not None
    assert r4.recalled_entry.window_bars == 2

    # Third 2-bar cycle (bars 5 & 6) stays on riff_A and increments iteration to 3
    r5 = detector.observe_bar(_make_bar_fp(5, bar_a1))
    assert r5.riff_id == "riff_A"
    assert not r5.riff_changed
    r6 = detector.observe_bar(_make_bar_fp(6, bar_a2))
    assert r6.riff_id == "riff_A"
    assert r6.riff_iteration == 3
    assert not r6.riff_changed


def test_switching_to_contrasting_riff_sets_riff_changed_and_mints_riff_b() -> None:
    detector = RiffDetector()
    riff_a = [(0, "E2"), (2, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]
    riff_b = [(0, "A2"), (6, "C3"), (10, "D3"), (14, "E3")]

    detector.observe_bar(_make_bar_fp(1, riff_a))
    r2 = detector.observe_bar(_make_bar_fp(2, riff_a))
    assert r2.riff_id == "riff_A"

    # Switch to contrasting riff B on bar 3
    r3 = detector.observe_bar(_make_bar_fp(3, riff_b))
    assert r3.riff_changed is True
    assert r3.riff_id == "riff_B"
    assert r3.riff_iteration == 1

    # Continue riff B on bar 4
    r4 = detector.observe_bar(_make_bar_fp(4, riff_b))
    assert r4.riff_changed is False
    assert r4.riff_id == "riff_B"
    assert r4.riff_iteration == 2
    assert len(detector.memory) == 2


def test_a_to_b_to_a_recalls_riff_a_metadata_from_session_memory() -> None:
    detector = RiffDetector()
    riff_a = [(0, "E2"), (2, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]
    riff_b = [(0, "C3"), (4, "B2"), (8, "G2"), (12, "D3")]

    detector.observe_bar(
        _make_bar_fp(1, riff_a),
        preferred_drum_pattern_id="stoner_mid",
        preferred_intensity_tier="mid",
        estimated_key="E minor",
        bass_register="low",
    )
    r2 = detector.observe_bar(
        _make_bar_fp(2, riff_a),
        preferred_drum_pattern_id="stoner_mid",
        preferred_intensity_tier="mid",
        estimated_key="E minor",
        bass_register="low",
    )
    assert r2.riff_id == "riff_A"

    # Play contrasting riff B for 2 bars with high-tier groove
    r3 = detector.observe_bar(
        _make_bar_fp(3, riff_b),
        preferred_drum_pattern_id="stoner_high",
        preferred_intensity_tier="high",
        estimated_key="C major",
        bass_register="mid",
    )
    assert r3.riff_id == "riff_B"
    assert r3.riff_changed is True

    r4 = detector.observe_bar(
        _make_bar_fp(4, riff_b),
        preferred_drum_pattern_id="stoner_high",
        preferred_intensity_tier="high",
        estimated_key="C major",
        bass_register="mid",
    )
    assert r4.riff_id == "riff_B"
    assert r4.riff_iteration == 2

    # Return to riff A on bar 5: must match riff_A in SessionMemory and recall its metadata
    r5 = detector.observe_bar(_make_bar_fp(5, riff_a))
    assert r5.riff_changed is True
    assert r5.riff_id == "riff_A"
    assert r5.riff_iteration == 1
    assert r5.recalled_entry is not None
    assert r5.recalled_entry.riff_id == "riff_A"
    assert r5.recalled_entry.preferred_drum_pattern_id == "stoner_mid"
    assert r5.recalled_entry.preferred_intensity_tier == "mid"
    assert r5.recalled_entry.estimated_key == "E minor"
    assert r5.recalled_entry.bass_register == "low"
    assert r5.recalled_entry.times_played == 3
    assert r5.recalled_entry.last_seen_bar == 5
    assert len(detector.memory) == 2


def test_tolerance_to_15ms_timing_jitter_and_missed_or_passing_note() -> None:
    detector = RiffDetector()
    base_riff = [(0, "E2"), (2, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]

    # Bar 1: clean
    detector.observe_bar(_make_bar_fp(1, base_riff))

    # Bar 2: +/- 15 ms timing jitter on every attack (including -15 ms early beat 1!)
    jittered_fp = _make_bar_fp(
        2,
        base_riff,
        jitter_ms_by_step={0: -15.0, 2: +14.0, 4: -12.0, 8: +15.0, 12: -14.0},
        stabilizer_delay_ms=70.0,
    )
    r2 = detector.observe_bar(jittered_fp)
    assert r2.riff_id == "riff_A"
    assert r2.riff_iteration == 2
    assert not r2.riff_changed
    assert r2.similarity >= 0.90

    # Bar 3: one missed note (step 2 omitted) + jitter
    missed_note_riff = [(0, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]
    r3 = detector.observe_bar(
        _make_bar_fp(3, missed_note_riff, jitter_ms_by_step={0: 10.0, 4: -13.0, 8: 12.0})
    )
    assert r3.riff_id == "riff_A"
    assert r3.riff_iteration == 3
    assert not r3.riff_changed

    # Bar 4: one extra chromatic passing note (Bb2 at step 10) + jitter
    passing_note_riff = [(0, "E2"), (2, "E2"), (4, "G2"), (8, "A2"), (10, "A#2"), (12, "E2")]
    r4 = detector.observe_bar(
        _make_bar_fp(4, passing_note_riff, jitter_ms_by_step={0: -11.0, 10: 14.0, 12: -10.0})
    )
    assert r4.riff_id == "riff_A"
    assert r4.riff_iteration == 4
    assert not r4.riff_changed


def test_session_memory_is_bounded_to_max_riffs() -> None:
    memory = SessionMemory(max_riffs=3)
    fp_a = _make_bar_fp(1, [(0, "E2"), (4, "G2")])
    fp_b = _make_bar_fp(2, [(0, "A2"), (6, "C3")])
    fp_c = _make_bar_fp(3, [(2, "D3"), (10, "F3")])
    fp_d = _make_bar_fp(4, [(0, "B2"), (8, "F#3")])

    for idx, fp in enumerate([fp_a, fp_b, fp_c, fp_d], start=1):
        riff_id = memory.next_riff_id()
        from jammate.music.riff import RiffMemoryEntry

        memory.store(
            RiffMemoryEntry(
                riff_id=riff_id,
                fingerprint=fp,
                accent_mask=fp.onset_mask,
                last_seen_bar=idx,
            )
        )

    assert len(memory) == 3
    assert memory.get("riff_A") is None  # oldest evicted
    assert memory.get("riff_B") is not None
    assert memory.get("riff_C") is not None
    assert memory.get("riff_D") is not None


def test_beat_clock_and_state_manager_integration_tracks_riffs_over_stream() -> None:
    clock = BeatClock(BeatClockConfig(), sample_rate_hz=SR, manual_bpm=BPM)
    detector = RiffDetector(RiffConfig())
    state_mgr = MusicalStateManager(manual_bpm=BPM, config=detector_state_cfg())

    riff_steps = [(0, "E2"), (4, "G2"), (8, "A2"), (12, "E2")]
    for bar_idx in range(1, 4):
        bar_start = (bar_idx - 1) * BAR_FRAMES
        for step_idx, note_name in riff_steps:
            frame: SampleFrame = bar_start + step_idx * STEP_FRAMES
            midi = name_to_midi(note_name)
            onset = OnsetEvent(
                frame_index=frame,
                timestamp_seconds=frames_to_seconds(frame, SR),
                strength=0.85,
            )
            pitch = PitchEvent(
                frame_index=frame + 1_920,
                timestamp_seconds=frames_to_seconds(frame + 1_920, SR),
                midi_note=midi,
                frequency_hz=midi_to_hz(midi),
                confidence=0.95,
            )
            detector.handle_event(onset)
            detector.handle_event(pitch)
            state_mgr.apply(pitch)
        detector.handle_event(
            IntensityUpdate(
                frame_index=bar_start + BAR_FRAMES // 2,
                timestamp_seconds=frames_to_seconds(bar_start + BAR_FRAMES // 2, SR),
                intensity=0.72,
            )
        )
        end_frame = bar_idx * BAR_FRAMES
        res = detector.advance_to(
            end_frame,
            clock.snapshot,
            preferred_drum_pattern_id="rock_mid",
            preferred_intensity_tier="mid",
            estimated_key="E minor",
        )
        assert res is not None
        state_mgr.update_riff(
            end_frame,
            riff_id=res.riff_id,
            riff_iteration=res.riff_iteration,
            riff_changed=res.riff_changed,
            section="groove" if res.riff_id else "intro",
        )

    snap = state_mgr.snapshot
    assert snap.riff_id == "riff_A"
    assert snap.riff_iteration == 3
    assert snap.riff_changed is False
    assert snap.section == "groove"
    assert (
        fingerprint_similarity(
            detector.memory.get("riff_A").fingerprint,  # type: ignore[union-attr]
            _make_bar_fp(1, riff_steps),
        )
        >= 0.90
    )


def detector_state_cfg():
    from jammate.config import MusicalStateConfig

    return MusicalStateConfig(instrument="guitar")

"""Unit tests for manual escape hatches, Tap Tempo, and multi-meter support (Step 6.1)."""

from __future__ import annotations

import pytest

from jammate.ai.null import NullAccompaniment
from jammate.audio.engine import AudioEngine
from jammate.config import (
    AnalysisConfig,
    BeatClockConfig,
    DrumConfig,
    JamMateConfig,
    KeyConfig,
    TempoConfig,
)
from jammate.drums.engine import DrummerController, DrumStepScheduler, step_to_frame
from jammate.drums.patterns import (
    INTENSITY_TIERS,
    STYLE_FAMILIES,
    get_fill_pattern,
    get_pattern,
    steps_per_bar_for_beats,
)
from jammate.drums.sampler import DrumEventRing, DrumPiece
from jammate.main import build_config, build_parser
from jammate.music.beat_clock import BeatClock
from jammate.music.events import (
    KeyUpdate,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    TempoUpdate,
)
from jammate.music.overrides import (
    ManualOverrides,
    TapTempoTracker,
    format_meter,
    meter_beats_per_bar,
    meter_steps_per_bar,
    parse_meter,
    validate_manual_key,
)
from jammate.session import JamSession
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, FakeClock


def test_tap_tempo_math_and_gap_reset() -> None:
    """4 taps spaced 0.5 s apart yield 120.0 BPM; a gap > 2.5 s resets the tap buffer."""
    clock = FakeClock(start_frame=480_000)
    tracker = TapTempoTracker(monotonic=clock, max_gap_seconds=2.5)

    # First tap primes the buffer and returns None.
    assert tracker.tap() is None
    assert tracker.tap_count == 1

    # Three more taps spaced 0.5 s apart -> 120.0 BPM.
    for _ in range(3):
        clock.advance(0.5)
        bpm = tracker.tap()
    assert tracker.tap_count == 4
    assert bpm == pytest.approx(120.0)
    assert tracker.current_bpm == pytest.approx(120.0)

    # A gap > 2.5 s resets the tap buffer so the next tap starts fresh.
    clock.advance(3.0)
    assert tracker.tap() is None
    assert tracker.tap_count == 1
    assert tracker.current_bpm is None

    # Two taps spaced 0.6 s apart -> 100.0 BPM.
    clock.advance(0.6)
    assert tracker.tap() == pytest.approx(100.0)


def test_parse_and_format_meters_and_key_validation() -> None:
    assert parse_meter("4/4") == (4, 4)
    assert parse_meter("3/4") == (3, 4)
    assert parse_meter("6/8") == (6, 8)
    assert parse_meter((3, 4)) == (3, 4)
    assert format_meter((6, 8)) == "6/8"
    assert meter_beats_per_bar("4/4") == 4
    assert meter_beats_per_bar("3/4") == 3
    assert meter_beats_per_bar("6/8") == 6
    assert meter_steps_per_bar("4/4") == 16
    assert meter_steps_per_bar("3/4") == 12
    assert meter_steps_per_bar("6/8") == 12

    assert validate_manual_key(None) is None
    assert validate_manual_key("Auto") is None
    assert validate_manual_key("Em") == "E minor"
    assert validate_manual_key("C major") == "C major"

    overrides = ManualOverrides(bpm=110.0, meter=(3, 4), key="A minor")
    assert overrides.meter_name == "3/4"
    assert overrides.beats_per_bar == 3
    assert overrides.steps_per_bar == 12


def test_manual_bpm_override_bypasses_tempo_updates(fake_stream) -> None:
    """Manual BPM override bypasses TempoTracker updates and keeps BeatClock locked at 110.0 BPM."""
    del fake_stream
    cfg = JamMateConfig(
        analysis=AnalysisConfig(tempo=TempoConfig(manual_bpm=110.0)),
    )
    session = JamSession(
        cfg,
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        accompaniment_factory=lambda ai_cfg, _sr: NullAccompaniment(
            style=ai_cfg.style,
            follow_improvise=ai_cfg.follow_improvise,
        ),
    )
    assert session.state.bpm == pytest.approx(110.0)
    assert session.state.tempo_locked is True
    assert session.state.tempo_confidence == pytest.approx(1.0)
    assert session.beat_clock.bpm == pytest.approx(110.0)

    # Feeding a conflicting TempoUpdate must not alter the manual BPM in BeatClock or MusicalState.
    session._handle_event(
        TempoUpdate(
            frame_index=24_000,
            timestamp_seconds=0.5,
            bpm=145.0,
            confidence=0.92,
            locked=True,
        )
    )
    assert session.state.bpm == pytest.approx(110.0)
    assert session.beat_clock.bpm == pytest.approx(110.0)

    # Switching manual BPM live to 96.0 updates BeatClock and MusicalState immediately.
    session.set_manual_bpm(96.0)
    assert session.state.bpm == pytest.approx(96.0)
    assert session.beat_clock.bpm == pytest.approx(96.0)

    # Clearing manual BPM returns to auto-tempo mode.
    session.set_manual_bpm(None)
    assert session.overrides.bpm is None
    assert session.state.tempo_locked is False


def test_meter_3_4_and_6_8_beat_clock_and_drum_patterns() -> None:
    """Meter 3/4 advances bar every 3 beats and schedules 12-step patterns; 6/8 advances every
    6 eighth-note beats with compound-pulse snare backbeat on beat 4 (step 6)."""
    sample_rate_hz = 48_000

    # Verify all style families and tiers provide 12-step 3/4 and 6/8 patterns.
    for style in STYLE_FAMILIES:
        for tier in INTENSITY_TIERS:
            pat_3_4 = get_pattern(style, tier, beats_per_bar=3)
            pat_6_8 = get_pattern(style, tier, beats_per_bar=6)
            assert len(pat_3_4.steps) == 12
            assert len(pat_6_8.steps) == 12
            fill_3_4 = get_fill_pattern(pat_3_4, beats_per_bar=3)
            assert len(fill_3_4.steps) == 12
            # In 6/8, step 6 is beat 4 (the compound backbeat) and carries a snare hit.
            snare_on_beat_4 = any(hit.piece == DrumPiece.SNARE for hit in pat_6_8.steps[6])
            assert snare_on_beat_4, f"{pat_6_8.name} missing snare on compound backbeat step 6"

    # Meter 3/4 BeatClock + DrumStepScheduler check:
    clock_3_4 = BeatClock(
        BeatClockConfig(meter="3/4"),
        sample_rate_hz=sample_rate_hz,
        manual_bpm=120.0,
    )
    snap_3_4 = clock_3_4.snapshot
    assert snap_3_4 is not None
    assert snap_3_4.beats_per_bar == 3
    assert steps_per_bar_for_beats(snap_3_4.beats_per_bar) == 12

    # At 120 BPM (24,000 frames/beat), 3 beats = 72,000 frames per bar.
    assert clock_3_4.position_at(0).bar == 1
    assert clock_3_4.position_at(0).beat == 1
    assert clock_3_4.position_at(24_000).beat == 2
    assert clock_3_4.position_at(48_000).beat == 3
    assert clock_3_4.position_at(72_000).bar == 2
    assert clock_3_4.position_at(72_000).beat == 1
    assert step_to_frame(12, snap_3_4) == 72_000

    controller_3_4 = DrummerController(
        DrumConfig(style="rock", humanize_enabled=False),
        beats_per_bar=3,
        manual_bpm_active=True,
    )
    ring_3_4 = DrumEventRing(capacity=128)
    scheduler_3_4 = DrumStepScheduler(ring_3_4)
    events_3_4 = scheduler_3_4.schedule_window(
        current_frame=0,
        lookahead_frames=72_000,
        clock=snap_3_4,
        pattern_for_bar=controller_3_4.pattern_for_bar,
    )
    assert len(events_3_4) > 0
    # Snare in rock_mid_3_4 is on step 8 -> beat 3 -> frame 48,000.
    snare_frames_3_4 = [e.frame_index for e in events_3_4 if e.piece == DrumPiece.SNARE]
    assert snare_frames_3_4 == [48_000]

    # Meter 6/8 BeatClock + DrumStepScheduler check:
    clock_6_8 = BeatClock(
        BeatClockConfig(meter="6/8"),
        sample_rate_hz=sample_rate_hz,
        manual_bpm=120.0,
    )
    snap_6_8 = clock_6_8.snapshot
    assert snap_6_8 is not None
    assert snap_6_8.beats_per_bar == 6
    # 6 beats at 24,000 frames/beat = 144,000 frames per bar; beat 4 is at frame 72,000 (step 6).
    assert clock_6_8.position_at(72_000).bar == 1
    assert clock_6_8.position_at(72_000).beat == 4
    assert clock_6_8.position_at(144_000).bar == 2
    assert clock_6_8.position_at(144_000).beat == 1
    assert step_to_frame(6, snap_6_8) == 72_000

    controller_6_8 = DrummerController(
        DrumConfig(style="rock", humanize_enabled=False),
        beats_per_bar=6,
        manual_bpm_active=True,
    )
    ring_6_8 = DrumEventRing(capacity=128)
    scheduler_6_8 = DrumStepScheduler(ring_6_8)
    events_6_8 = scheduler_6_8.schedule_window(
        current_frame=0,
        lookahead_frames=144_000,
        clock=snap_6_8,
        pattern_for_bar=controller_6_8.pattern_for_bar,
    )
    snare_frames_6_8 = [e.frame_index for e in events_6_8 if e.piece == DrumPiece.SNARE]
    assert snare_frames_6_8 == [72_000]


def test_manual_key_override_and_reset_musical_context(fake_stream) -> None:
    """Manual key override locks estimated_key/mode with confidence 1.0, and reset_musical_context
    clears tempo lock, key estimate, RiffDetector, and SessionMemory while keeping audio running."""
    del fake_stream
    cfg = JamMateConfig()
    session = JamSession(
        cfg,
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        accompaniment_factory=lambda ai_cfg, _sr: NullAccompaniment(
            style=ai_cfg.style,
            follow_improvise=ai_cfg.follow_improvise,
        ),
    )
    session.start()
    try:
        assert session.is_running

        # Apply manual key override -> E minor with harmonic_confidence == 1.0.
        session.set_manual_key("E minor")
        assert session.state.estimated_key == "E"
        assert session.state.estimated_mode == "minor"
        assert session.state.harmonic_confidence == pytest.approx(1.0)
        assert session.overrides.key == "E minor"

        # Even if analysis emits a different KeyUpdate, the manual override wins.
        session._handle_event(
            KeyUpdate(
                frame_index=4_800,
                timestamp_seconds=0.1,
                key_name="C",
                mode="major",
                confidence=0.75,
                bass_register="low",
            )
        )
        assert session.state.estimated_key == "E"
        assert session.state.estimated_mode == "minor"
        assert session.state.harmonic_confidence == pytest.approx(1.0)

        # Clear manual key override back to Auto.
        session.set_manual_key(None)
        assert session.overrides.key is None
        assert session.state.estimated_key is None

        # Populate tempo lock, key estimate, and riff detector state across 2 bars.
        session._handle_event(
            PlayingStateUpdate(
                frame_index=4_800,
                timestamp_seconds=0.1,
                state=PlayingState.PLAYING,
            )
        )
        session._handle_event(
            TempoUpdate(
                frame_index=9_600,
                timestamp_seconds=0.2,
                bpm=120.0,
                confidence=0.90,
                locked=True,
            )
        )
        session._handle_event(
            KeyUpdate(
                frame_index=9_600,
                timestamp_seconds=0.2,
                key_name="A",
                mode="minor",
                confidence=0.82,
                bass_register="low",
            )
        )
        for bar_idx in range(3):
            f_idx = 9_600 + bar_idx * 96_000
            session._handle_event(
                OnsetEvent(
                    frame_index=f_idx,
                    timestamp_seconds=f_idx / 48_000,
                    strength=0.9,
                )
            )
            session._handle_event(
                PitchEvent(
                    frame_index=f_idx,
                    timestamp_seconds=f_idx / 48_000,
                    midi_note=45,
                    frequency_hz=110.0,
                    confidence=0.95,
                )
            )

        assert session.state.tempo_locked is True
        assert session.state.estimated_key == "A"
        assert session.riff_detector.riff_id is not None
        assert len(session.riff_detector.memory) >= 1

        # Reset musical context while the stream continues running.
        session.reset_musical_context()
        assert session.is_running
        assert session.engine.is_running
        assert session.state.tempo_locked is False
        assert session.state.bpm is None
        assert session.state.estimated_key is None
        assert session.state.riff_id is None
        assert session.riff_detector.riff_id is None
        assert len(session.riff_detector.memory) == 0
    finally:
        session.stop()


def test_cli_build_config_supports_meter_and_key() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--bpm", "108", "--meter", "6/8", "--key", "D minor"])
    cfg = build_config(args)
    assert cfg.analysis.tempo.manual_bpm == pytest.approx(108.0)
    assert cfg.beat_clock.meter == "6/8"
    assert cfg.beat_clock.beats_per_bar == 6
    assert cfg.analysis.key.manual_key == "D minor"
    assert KeyConfig(manual_key="F#m").manual_key == "F# minor"

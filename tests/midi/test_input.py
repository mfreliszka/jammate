"""Unit and integration tests for MIDI keyboard input (Step 6.3, §55, §150, §153)."""

from __future__ import annotations

import numpy as np
import pytest

from jammate.ai.base import TOKEN_HELD, TOKEN_ONSET
from jammate.config import AnalysisConfig, JamMateConfig
from jammate.main import build_config, build_parser
from jammate.midi.input import (
    ALL_NOTES_OFF_CC,
    FakeMidiPort,
    MidiDeviceResolutionError,
    MidiInputWorker,
    MidiStatus,
    analyze_midi_notes,
    format_midi_device_list,
    list_midi_inputs,
    resolve_midi_input,
    velocity_to_strength,
)
from jammate.music.events import (
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingState,
    PlayingStateUpdate,
    SilenceEvent,
    seconds_to_frames,
)
from jammate.session import JamSession


def test_midi_device_discovery_and_resolution() -> None:
    ports = list_midi_inputs(lambda: ["KeyStep 37", "Launchkey Mini MK3", "KeyStep 37"])
    assert ports == ["KeyStep 37", "Launchkey Mini MK3"]

    assert resolve_midi_input(None, ports) is None
    assert resolve_midi_input("none", ports) is None
    assert resolve_midi_input("KeyStep 37", ports) == "KeyStep 37"
    assert resolve_midi_input("launchkey", ports) == "Launchkey Mini MK3"

    with pytest.raises(MidiDeviceResolutionError, match="No MIDI input device matches"):
        resolve_midi_input("Minilogue", ports)

    with pytest.raises(MidiDeviceResolutionError, match="ambiguous"):
        resolve_midi_input("Key", ports)

    formatted = format_midi_device_list(ports, selected="KeyStep 37")
    assert "* [1] KeyStep 37" in formatted
    assert "  [2] Launchkey Mini MK3" in formatted


def test_analyze_midi_notes_monophonic_and_polyphonic_chords() -> None:
    assert analyze_midi_notes([]) is None

    # Monophonic single note
    single = analyze_midi_notes([40])
    assert single is not None
    assert single.root_midi == 40
    assert single.chord_notes == ()
    assert single.chord_type is None

    # Power chord (E2 + B2 + E3)
    power = analyze_midi_notes([40, 47, 52])
    assert power is not None
    assert power.root_midi == 40
    assert power.chord_notes == (40, 47, 52)
    assert power.chord_type == "power"

    # Root-position E minor triad (E3 + G3 + B3)
    minor_triad = analyze_midi_notes([52, 55, 59])
    assert minor_triad is not None
    assert minor_triad.root_midi == 52
    assert minor_triad.chord_notes == (52, 55, 59)
    assert minor_triad.chord_type == "minor"

    # Root-position C major triad (C3 + E3 + G3)
    major_triad = analyze_midi_notes([48, 52, 55])
    assert major_triad is not None
    assert major_triad.root_midi == 48
    assert major_triad.chord_notes == (48, 52, 55)
    assert major_triad.chord_type == "major"

    # First-inversion C major triad (E3 + G3 + C4): extracts C4 (60) as the triad root
    inverted_major = analyze_midi_notes([52, 55, 60])
    assert inverted_major is not None
    assert inverted_major.root_midi == 60
    assert inverted_major.chord_notes == (52, 55, 60)
    assert inverted_major.chord_type == "major"


def test_midi_worker_monophonic_and_polyphonic_events() -> None:
    port = FakeMidiPort("KeyStep 37")
    emitted: list[MusicalEvent] = []
    frame = 2400
    worker = MidiInputWorker(
        AnalysisConfig(instrument="guitar"),
        48_000,
        port=port,
        frame_source=lambda: frame,
        on_event=emitted.append,
    )

    # Monophonic note_on at E2 (MIDI 40), velocity 100
    port.note_on(40, velocity=100)
    assert worker.process_available() == 1

    onsets = [e for e in emitted if isinstance(e, OnsetEvent)]
    pitches = [e for e in emitted if isinstance(e, PitchEvent)]
    playing = [e for e in emitted if isinstance(e, PlayingStateUpdate)]
    assert len(onsets) == 1
    assert onsets[0].frame_index == 2400
    assert onsets[0].strength == pytest.approx(velocity_to_strength(100))
    assert len(pitches) == 1
    assert pitches[0].midi_note == 40
    assert pitches[0].chord_notes == ()
    assert pitches[0].chord_type is None
    assert playing[-1].state == PlayingState.PLAYING

    # Add B2 (47) and G2 (43) while holding E2 (40) -> E minor chord
    emitted.clear()
    frame = 4800
    port.note_on(47, velocity=95)
    port.note_on(43, velocity=90)
    assert worker.process_available() == 2

    pitches = [e for e in emitted if isinstance(e, PitchEvent)]
    assert pitches[-1].midi_note == 40
    assert pitches[-1].chord_notes == (40, 43, 47)
    assert pitches[-1].chord_type == "minor"

    # Release G2 (43) -> remaining (40, 47) becomes an E5 power chord
    emitted.clear()
    frame = 7200
    port.note_off(43)
    assert worker.process_available() == 1
    pitches = [e for e in emitted if isinstance(e, PitchEvent)]
    assert len(pitches) == 1
    assert pitches[0].midi_note == 40
    assert pitches[0].chord_notes == (40, 47)
    assert pitches[0].chord_type == "power"

    # Release remaining notes (velocity=0 note_on and note_off) -> SilenceEvent + STOPPED
    emitted.clear()
    frame = 9600
    port.note_on(47, velocity=0)
    port.note_off(40)
    assert worker.process_available() == 2
    assert any(isinstance(e, SilenceEvent) for e in emitted)
    playing = [e for e in emitted if isinstance(e, PlayingStateUpdate)]
    assert playing[-1].state == PlayingState.STOPPED
    assert worker.active_notes == ()


def test_midi_worker_zero_stuck_notes_on_disconnect_stop_and_cc123() -> None:
    port = FakeMidiPort("Launchkey")
    emitted: list[MusicalEvent] = []
    worker = MidiInputWorker(
        AnalysisConfig(),
        48_000,
        port=port,
        frame_source=lambda: 4800,
        on_event=emitted.append,
    )

    # CC 123 (All Notes Off) releases held notes
    port.note_on(45, velocity=110)
    worker.process_available()
    assert worker.active_notes == (45,)
    emitted.clear()
    port.control_change(ALL_NOTES_OFF_CC, 0)
    worker.process_available()
    assert worker.active_notes == ()
    assert any(isinstance(e, SilenceEvent) for e in emitted)

    # Unexpected USB MIDI disconnect releases all held notes cleanly
    port.note_on(40, velocity=100)
    port.note_on(47, velocity=100)
    worker.process_available()
    assert worker.active_notes == (40, 47)
    emitted.clear()
    port.disconnect()
    assert worker.process_available() == 0
    assert worker.status == MidiStatus.DISCONNECTED
    assert worker.active_notes == ()
    assert any(isinstance(e, SilenceEvent) for e in emitted)

    # Calling stop() on a fresh worker with held notes releases all active notes
    port2 = FakeMidiPort("KeyStep")
    emitted2: list[MusicalEvent] = []
    worker2 = MidiInputWorker(
        AnalysisConfig(),
        48_000,
        port=port2,
        frame_source=lambda: 9600,
        on_event=emitted2.append,
    )
    port2.note_on(52, velocity=100)
    worker2.process_available()
    assert worker2.active_notes == (52,)
    emitted2.clear()
    worker2.stop()
    assert worker2.active_notes == ()
    assert any(isinstance(e, SilenceEvent) for e in emitted2)


def test_midi_drives_jamsession_state_beatclock_key_and_notetracker(fake_stream) -> None:
    port = FakeMidiPort("KeyStep 37")
    cfg = JamMateConfig()
    session = JamSession(cfg, midi_port=port)
    session.set_instrument("guitar")

    # Feed 8 quarter-note chords at 120 BPM (0.5 s = 24,000 frames apart) in E minor
    chord_progression = [
        (40, 43, 47),  # E minor
        (40, 43, 47),
        (43, 47, 50),  # G major
        (45, 48, 52),  # A minor
        (40, 43, 47),  # E minor
        (40, 43, 47),
        (47, 50, 54),  # B minor
        (40, 43, 47),  # E minor
    ]
    for i, triad in enumerate(chord_progression):
        onset_frame = seconds_to_frames(i * 0.5, 48_000)
        for note in triad:
            port.note_on(note, velocity=110)
        session.midi_worker.process_available(frame_index=onset_frame)

        # Verify NoteTracker has both root and full chord notes in the 128-pitch pianoroll
        roll = session.note_tracker.consume_pianoroll()
        for note in triad:
            assert roll[note] in (TOKEN_ONSET, TOKEN_HELD)

        release_frame = onset_frame + seconds_to_frames(0.35, 48_000)
        for note in triad:
            port.note_off(note)
        session.midi_worker.process_available(frame_index=release_frame)

    # Tempo should be locked near 120 BPM and BeatClock running
    assert session.state.tempo_locked is True
    assert session.state.bpm == pytest.approx(120.0, abs=1.5)
    assert session.beat_clock.is_running is True
    assert session.state.estimated_key == "E"
    assert session.state.estimated_mode == "minor"

    # Hold a chord and verify session.stop() leaves zero stuck notes in NoteTracker
    port.note_on(40, velocity=100)
    port.note_on(47, velocity=100)
    session.midi_worker.process_available(frame_index=seconds_to_frames(4.5, 48_000))
    assert session.note_tracker.active_note == 40
    session.stop()
    assert session.note_tracker.active_note is None
    assert int(np.sum(session.note_tracker.consume_pianoroll())) == 0


def test_cli_midi_flags_populate_config() -> None:
    parser = build_parser()
    args = parser.parse_args(["--midi-input", "KeyStep", "--list-midi"])
    cfg = build_config(args)
    assert args.list_midi is True
    assert cfg.midi.input_device_name == "KeyStep"

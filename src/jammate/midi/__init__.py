"""Optional MIDI keyboard input subpackage (Step 6.3, §55, §150)."""

from jammate.midi.input import (
    FakeMidiPort,
    MidiChordAnalysis,
    MidiDeviceResolutionError,
    MidiInputWorker,
    MidiPort,
    MidiStats,
    MidiStatus,
    NoteObservation,
    OnsetObservation,
    analyze_midi_notes,
    format_midi_device_list,
    list_midi_inputs,
    open_midi_input_port,
    resolve_midi_input,
    velocity_to_strength,
)

__all__ = [
    "FakeMidiPort",
    "MidiChordAnalysis",
    "MidiDeviceResolutionError",
    "MidiInputWorker",
    "MidiPort",
    "MidiStats",
    "MidiStatus",
    "NoteObservation",
    "OnsetObservation",
    "analyze_midi_notes",
    "format_midi_device_list",
    "list_midi_inputs",
    "open_midi_input_port",
    "resolve_midi_input",
    "velocity_to_strength",
]

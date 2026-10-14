# midi/ — MIDI keyboard input

Optional MIDI keyboard input (`mido` + `python-rtmidi`, Step 6.3, §55, §150). Translates incoming
`note_on` and `note_off` messages into `MusicalEvent`s (`PitchEvent`, `OnsetEvent`, `SilenceEvent`,
`PlayingStateUpdate`, `IntensityUpdate`, `TempoUpdate`, `KeyUpdate`) stamped with the audio
stream's `frame_index`, driving `MusicalStateManager`, `BeatClock`, `KeyEstimator`, `RiffDetector`,
`DrumEngine`, and `NoteTracker` with zero pitch-detection latency.

| Module | Status | Thread |
|---|---|---|
| `input.py` | done (6.3) | `list_midi_inputs` / `resolve_midi_input` on main/GUI thread; `MidiInputWorker` on `jammate-midi` worker thread |

## Invariants

- Never call MIDI port methods from the PortAudio callback thread.
- Every emitted event carries `frame_index` from `AudioEngine.processor.frames_processed` and
  `timestamp_seconds` from `frames_to_seconds(frame_index, sample_rate_hz)`.
- Polyphonic MIDI chords extract the lowest/triad root note (`root_midi`) for root tracking while
  passing all held `chord_notes` to `NoteTracker` for 128-pitch MRT2 pianoroll conditioning.
- Calling `stop()`, receiving CC `120`/`123`, or encountering a USB MIDI disconnect immediately
  releases all active notes (`SilenceEvent` + `PlayingStateUpdate(STOPPED)`) so no stuck notes
  remain.
- Unit tests use `FakeMidiPort` and `process_available(frame_index=...)` without requiring physical
  MIDI hardware (§66, §67).

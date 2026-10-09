# music/ — shared musical vocabulary

Holds the types every layer uses to talk about music: analysis events, the Musical State, the
BeatClock and style presets. It processes no audio.

| Module | Status | Thread |
|---|---|---|
| `events.py` | done (2.1) | any thread; immutable values |
| `state.py` | Step 2.5 | `MusicalStateManager` written by one thread, snapshots read by any |
| `beat_clock.py` | Step 3.1 | pure conversions; any thread |
| `styles.py` | Step 4.3 | any thread |

## Invariants

- Events and snapshots are frozen dataclasses. Consumers never mutate them.
- Events take keyword arguments only and validate their values on construction. They are built
  on worker threads, never in the audio callback, so raising `ValueError` there is fine.
- Every event has `frame_index` (the master clock, D5) and `timestamp_seconds` (for logs and
  display). Compute both from one frame index with `frames_to_seconds`.
- `events.py` imports nothing else from `jammate`. `analysis/notes.py` imports it, so an import
  back would be circular.
- Note names aren't stored on events. Derive them with `analysis.notes.midi_to_name`.
- `PlayingState.user_is_playing` is the single definition of "playing": PLAYING or SUSTAINING.
  `SilenceEvent` means the *note* ended; `PlayingStateUpdate` is about the *player*, who may
  still be strumming chords with no detectable note.

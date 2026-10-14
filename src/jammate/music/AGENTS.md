# music/ — shared musical vocabulary

Holds the types every layer uses to talk about music: analysis events, the Musical State, the
BeatClock and style presets. It processes no audio.

| Module | Status | Thread |
|---|---|---|
| `events.py` | done (2.1) | any thread; immutable values |
| `state.py` | done (2.5) | `MusicalStateManager` written by one thread (the analysis worker), snapshots read by any |
| `beat_clock.py` | done (3.1) | `BeatClock` updated on the analysis worker thread; `BeatClockSnapshot` and pure conversions on any thread |
| `styles.py` | done (4.3) | any thread; `StylePreset`, `map_follow_improvise`, and `AiEntryController` |
| `riff.py` | done (5.3) | `RiffDetector` and `SessionMemory` updated on the analysis worker thread; `BarFingerprint` and `RiffMemoryEntry` immutable |
| `director.py` | done (5.5) | `BandDirector` evaluated on the analysis worker thread; `BandDirective` immutable |
| `overrides.py` | done (6.1) | `TapTempoTracker`, `ManualOverrides`, and multi-meter helpers (`4/4`, `3/4`, `6/8`) |

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
- `MusicalState` is the single source of what JamMate believes musically (§38). Drums, AI, the
  GUI and the headless status line read `JamSession.state`. They never read the analysis
  trackers, and they never keep their own copy. `AnalysisStats` carries diagnostics only.
- `MusicalStateManager.apply` and `update_beat` are called only by its writer thread, which is
  the first thread to call either: the analysis worker. A second writer raises `RuntimeError`.
- Each update builds a new frozen snapshot and swaps one reference. Readers never lock or wait.
  Fields are only added when a step needs them (§37).
- `BeatClock` computes beat frames from `round(anchor_frame + k * frames_per_beat)` so there is
  no cumulative rounding drift. When BPM updates while locked, `anchor_frame` is adjusted at the
  update frame so the continuous beat count never jumps (§33).
- Offline tools and tests rebuild the state from events with `replay(...)`.

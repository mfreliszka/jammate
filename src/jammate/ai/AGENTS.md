# ai/ — generative accompaniment adapters

Isolates all generative model logic behind the `AccompanimentEngine` Protocol (§51, §52) and
manages the monophonic note-conditioning lifecycle (`NoteTracker`, §69) and failure boundary
(`GuardedAccompaniment`, §82, §137).

| Module | Status | Thread |
|---|---|---|
| `base.py` | done (4.1) | `NoteTracker` called on analysis worker and shutdown; `GuardedAccompaniment` wraps any adapter |
| `null.py` | done (4.1) | any thread; no-op `NullAccompaniment` when AI is disabled or unavailable |
| `mrt.py` | done (4.2) | model load/warmup at startup; `Mrt2Accompaniment` generation worker thread (`jammate-ai`) |
| `native_bridge.py` | done (7.1) | shared-memory SPSC audio ring (`SharedMemoryAudioRing`) + `NativeBridgeAccompaniment` worker thread (`jammate-native-ai-bridge`) |
| `model_manager.py` | done (7.2) | startup verification of `MAGENTA_HOME` checkpoints/resources (`inspect_model_assets`) and non-blocking `ModelSetupTask` (`jammate-model-setup`) |

## Invariants

- No module outside `ai/mrt.py` may import `magenta_rt` (§51, §52).
- The audio callback **never** calls into `AccompanimentEngine` or `NoteTracker`; it only reads
  pre-generated 48 kHz stereo float32 audio from `output_ring` (`RingBuffer(channels=2)`).
- `NoteTracker` tracks the active root note (`active_note`) plus any accompanying guitar chord tones
  (`active_chord_notes`, Step 5.1): `note_off` is always sent before the next `note_on`, `note_off`
  is sent on `SilenceEvent`, and `all_notes_off()` is sent on session stop/shutdown so no note ever
  hangs (§69).
- Any exception raised by an adapter method or generation loop transitions the engine status to
  `AccompanimentStatus.FAILED` and logs the error without crashing instrument monitoring or drums
  (§82, §137).

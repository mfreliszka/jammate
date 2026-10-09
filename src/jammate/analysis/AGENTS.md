# analysis/ — instrument understanding

Turns raw input audio into musical observations: what note, when it was attacked, how hard, and
at what tempo. Analysis reports facts and confidence values. It makes no decisions about what the
band should do and renders nothing (§125).

| Module | Status | Thread |
|---|---|---|
| `notes.py` | done (2.1) | any thread; pure functions |
| `worker.py` | done (2.2) | `AnalysisWorker`, its own thread with explicit start/stop; `analyze_offline` drives it without a thread |
| `pitch.py` | done (2.2) | analysis worker; `yin` is pure |
| `stabilizer.py` | done (2.2) | analysis worker; pure logic |
| `onset.py` | Step 2.3 | analysis worker |
| `tempo.py` | Step 2.4 | analysis worker |

## Invariants

- Nothing in this package runs in the audio callback. Heavy DSP (librosa, FFTs) is allowed here
  and only here, on the analysis worker.
- The worker is the **only consumer** of the engine's input ring (`RingBuffer` is SPSC).
- Output is the event types from `music/events.py`. Every event's `frame_index` is on the input
  ring's frame clock, which is the stream's sample-frame index (D5).
- The input is a monophonic bass or a single-note electric guitar (D7). From Step 2.2, the
  `instrument` setting selects the pitch range. Chords must yield no note rather than a wrong
  one; onsets, intensity and tempo still work on them.
- Note numbers and names are sounding pitch (D6). Use `notes.py` for every Hz/MIDI/name
  conversion; don't re-derive `69 + 12·log2(f/440)` elsewhere.
- Keep the math in pure functions that take NumPy arrays and sample positions. Test them with
  `tests/signals.py` and `tests/data/` recordings, without hardware.
- `on_event` and `on_pitch_frame` run on the worker thread. Hand events to other threads through
  something bounded and lock-free, such as `collections.deque(maxlen=…)`. In Step 2.5 that
  becomes the `MusicalStateManager`, written only by this thread.
- The worker wakes every 1 ms (`POLL_SECONDS`). Don't lengthen it without measuring callback
  arrival gaps on hardware; see the root AGENTS.md realtime rules.
- Level gates use the **band-limited** window RMS, so DC, rumble and hiss never open them. A DC
  step otherwise leaves a smooth filtered curve that YIN calls periodic.
- Tune against the recordings with `scripts/analyze_wav.py … --hops`.
  `tests/analysis/test_recordings.py` pins the results.

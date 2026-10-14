# drums/ — virtual drummer

Turns the BeatClock and Musical State into sample-accurate drum accompaniment (§13, §43–§49).
Drums are always symbolic events rendered by a sample player, never generated AI audio.

| Module | Status | Thread |
|---|---|---|
| `sampler.py` | done (3.2) | init on main thread; `DrumEventRing.push` on scheduler thread; `DrumSampler.render_block` on the PortAudio callback thread |
| `patterns.py` | done (3.3) | any thread; pure 16-step pattern definitions and swing math |
| `adaptive.py` | done (5.4) | analysis/scheduler thread; `AccentTracker` and `adapt_pattern_to_accents` |
| `humanize.py` | done (5.4) | scheduler thread; pure deterministic microtiming, metric velocity shaping, and ghost notes |
| `engine.py` | done (3.3–3.4, 5.4) | `DrumEngine` scheduler worker thread (`jammate-drums`); `DrummerController` decision state machine |

## Invariants

- All WAV files are loaded and resampled to the engine's sample rate (48 kHz) at initialization,
  before the audio stream opens (§45, §83). Never touch disk on the callback thread.
- `DrumEventRing` is strictly single-producer/single-consumer. Events are pushed in chronological
  (`frame_index`) order by the scheduler thread and popped by `DrumSampler.render_block` on the
  callback thread.
- `DrumSampler.render_block` is called from the PortAudio callback and must stay non-blocking and
  allocation-free: preallocated voice pool (`max_voices`), preallocated scratch array, and in-place
  `np.multiply` / `np.add` ufuncs.
- Even when the drums bus is muted in `BlockProcessor`, `DrumSampler.render_block` must still run
  each block so due events are consumed and voices advance cleanly instead of accumulating in the
  ring.

# AGENTS.md — JamMate

Instructions for AI coding agents working in this repository. Read this file before every task.

## What JamMate is

JamMate is a **local, offline, realtime desktop app for macOS on Apple Silicon**. A bassist plugs
in through a USB audio interface, presses Jam and plays. JamMate then:

- listens to the bass and detects notes, onsets, intensity, tempo and beat position;
- keeps a **Musical State**;
- runs a deterministic **drummer** that joins on a bar boundary;
- conditions a generative model (**Magenta RealTime 2**) to add accompaniment.

> Product statement: **"Start playing. Your band joins in."**
> Success test: a bassist plays an arbitrary riff and after a few seconds feels *"this actually
> feels like jamming."* Musical feel and timing matter more than architectural elegance.

JamMate is **not**:
- a DAW;
- a text-to-song generator;
- a web app or a backend service.

## Key documents

| File | Purpose |
|---|---|
| [implementation_journal.md](implementation_journal.md) | **The plan and progress log.** Find the next `[-]` step here and update it when done. |
| [project_knowledge.md](project_knowledge.md) | The full product and architecture specification. The journal cites its sections as `§N`. Read the relevant sections before each step. |
| `src/jammate/<subpackage>/AGENTS.md`, `scripts/AGENTS.md` | Module-specific notes. Create one when you create a subpackage. |

## Workflow for every task

1. Read the current step in `implementation_journal.md` and the `§` sections it cites.
2. Inspect the existing code before changing anything. Respect the module boundaries.
3. Implement **only that step**. Keep tasks small and bounded (§61). Do not start later steps or
   backlog items.
4. State the realtime implications of the change: which thread runs it, and what the callback
   touches.
5. Write or extend unit tests for all deterministic logic.
6. Run `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .`. All must pass.
7. Update `implementation_journal.md`:
   - tick the step `[x]` only when its *Done when* list is fully met;
   - add a dated log entry;
   - record any plan change under *Decisions*.
8. Report limitations, and list exactly what the developer must test with a real bass. **You
   cannot hear audio.** Steps that need a hardware test stay `[-]`, marked `awaiting hardware
   test`, until the developer confirms.

Commit only when asked.

## Commands

Everything below works except the GUI, which arrives in Step 4.4. Until then plain
`uv run jammate` exits with a "not implemented yet" error.

The developer's interface is named "USB Audio CODEC". It is not the system default, so pass
`--input usb --output usb`.

The `Makefile` wraps the common commands (`make help` lists them):
- `make check` (lint + format check + tests) is the gate before finishing any change;
- `make run`, `make devices` and `make capture FILE=… DURATION=…` take `INPUT`, `OUTPUT`,
  `CHANNEL`, `BLOCK` and `GAIN` overrides (defaults: the USB interface, channel 1, 128 frames,
  gain 1.0).

```bash
uv sync                                  # install / update environment
uv run jammate                           # GUI (Phase 4+)
uv run jammate --headless --input usb --output usb   # CLI runner + live status line (Phases 1–3)
#   --input-channel N (1-based)  --monitor-gain G (0 = silent / direct monitoring)  --block-size 256
uv run jammate --list-devices [--input NAME --output NAME]   # devices + what NAME selects
uv run python scripts/capture_input.py out.wav --input usb --output usb --seconds 20  # record input
uv run pytest                            # unit tests (no hardware required)
uv run ruff check . && uv run ruff format --check .
uv add <pkg>    /  uv add --dev <pkg>    # dependencies (never pip, never Poetry)

# MRT2 sanity check (model already downloaded to ~/Documents/Magenta/magenta-rt-v2)
uv run mrt mlx generate --prompt "stoner rock" --duration 4.0 --model=mrt2_small
```

## Tech stack

**In use:**

| Area | Choice |
|---|---|
| Language | Python 3.12 (PEP 695 `type X = …` allowed) |
| Environment | `uv` |
| Platform | macOS on Apple Silicon only |
| Audio | `sounddevice` (PortAudio / CoreAudio). 48 kHz, mono input, stereo output, block size **128** (fallback 256) |
| DSP | NumPy, SciPy, `librosa` where useful (YIN pitch, 35–400 Hz) |
| File I/O | `soundfile` (drum samples, recordings) |
| Config | `pydantic` typed models in `jammate/config.py` |
| AI | `magenta-rt[mlx]` 2.0.3 with `mrt2_small`, in-process (decision D1) |
| GUI | PySide6 / Qt Widgets (added in Step 4.4). **No** QML, Electron, React or webviews. |
| Dev tools | `pytest`, `ruff` |

**Not used unless the D1 fallback requires it:** `mido` and `python-rtmidi`.

## Architecture

### Mental model

Keep these layers separate. This is the central rule of the project (§142):

```text
RAW AUDIO → MUSICAL OBSERVATIONS → MUSICAL STATE → MUSICAL DECISIONS → RENDERING (audio)
waveform    E2 @ frame 201600,      bar 5 beat 1,    kick on next       samples mixed at
            onset, 92 BPM           intensity 0.7     downbeat           frame 211200
```

- **Analysis** answers *what, when, how loud, what tempo*. **Decisions** answer *what should the
  band do*. **Rendering** turns decisions into samples or generated audio. Do not mix these
  (§125, §126).
- Use **deterministic DSP** for timing-critical work: pitch, onsets, tempo, BeatClock, drum
  scheduling. Use **AI** only where creativity adds value: accompaniment and, later, variation
  (§12).
- Drums are **symbolic events played back as samples**, never generated audio (§13).

### Data flow and threads

```text
AUDIO THREAD (PortAudio callback)
  in  → ring buffer ──────────────► ANALYSIS WORKER (pitch, onset, intensity, tempo)
  out ← monitor + drum voices + AI ring          │
                                                 ▼
                                    MusicalStateManager ──► immutable snapshots
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                DRUM SCHEDULER WORKER    AI GENERATION WORKER       GUI THREAD (QTimer poll)
                → event ring (frames)    → AI audio ring             reads snapshots only
```

- `JamSession` (`session.py`) owns the lifecycle of every component. Startup and shutdown are
  explicit and ordered (§83, §84).
- Threads are joined on stop. **No daemon threads as a substitute for lifecycle management.**
- Do not use `asyncio` for the realtime path (§120).

### Repository layout (target)

```text
src/jammate/
  main.py          entry point: argparse, logging, GUI or --headless
  headless.py      terminal runner: status line, Ctrl+C/SIGTERM shutdown (until the GUI)
  config.py        central typed config (no magic numbers scattered in modules, §132)
  session.py       JamSession: wiring + startup/shutdown
  audio/           devices.py, engine.py (stream, callback, mixer), ring_buffer.py, recorder.py
  analysis/        worker.py, pitch.py, onset.py, tempo.py, notes.py (Hz ↔ MIDI ↔ name)
  music/           events.py, state.py, beat_clock.py, styles.py
  drums/           patterns.py, engine.py (scheduler + drummer decisions), sampler.py
  ai/              base.py (AccompanimentEngine Protocol), null.py, mrt.py (MRT2 adapter)
  ui/              main_window.py, debug_panel.py
assets/drums/      kit WAVs (dev placeholders, see Licensing)
scripts/           dev tools: capture_input.py, analyze_wav.py, render_jam.py, …
tests/             mirrors src/; fakes.py + conftest.py = fake PortAudio stream (`fake_stream`);
                   signals.py = synthetic audio (2.1); data/ = short riff WAVs
recordings/        jam recordings (gitignored)
```

Create modules when a step needs them. Do not add layers, such as services, repositories,
controllers or an event framework, without a concrete need (§118, §127).

## Realtime audio rules (hard rules)

Anything the audio callback calls must be **bounded, non-blocking and allocation-free**. The
callback only does the following:

- copies input into a preallocated ring buffer;
- reads already-prepared data (drum events, AI audio);
- mixes into `outdata`;
- updates counters.

**Never do any of the following inside the callback or code it calls:**

- AI inference
- `librosa` or any heavy DSP / FFT
- logging or `print`
- file, disk or network I/O
- Qt calls
- MIDI
- `threading.Lock`, `queue.Queue`, `Event.wait` or `sleep`
- waiting on other threads
- creating NumPy arrays (`np.zeros`, concatenation, fancy indexing that copies)
- Python-level loops over samples
- raising exceptions

Python-specific rules:

- Preallocate buffers. Write in place with `out[:] = …`, `np.copyto`, `np.multiply(a, g, out=…)`.
- Always fully write `outdata`: zero-fill first, then add buses.
- Read `status.input_overflow` / `status.output_underflow` into counters. Report them outside the
  callback.
- Cross-thread handoff uses **single-producer/single-consumer rings with monotonically increasing
  indices**, or **atomic reference swaps** of immutable objects. A plain attribute assignment is
  atomic under the GIL. The callback must never depend on a lock that the GUI or a worker holds
  (§39).
- Any Python thread that holds the GIL for a long time (tight Python loops, the MRT2 per-frame
  generation loop) can delay the callback. Keep worker loops chunked, and **measure xruns** with
  every subsystem on and off.
- All buffers are **bounded**. When a consumer falls behind, drop old data and count the drops.
  Never accumulate stale audio (§23).
- Load all samples and models **before** the stream opens (§45, §83).

Checklist before finishing any audio-related change (§64): logging, file I/O, MIDI, GUI, network
or inference in the callback; blocking locks; unbounded queues; allocations; expensive NumPy;
librosa; sleeps; thread shutdown bugs; underrun and overflow handling. Treat any hit as a bug.

## Musical State rules

- `MusicalStateManager` is the **single owner** of what JamMate currently believes musically:
  tempo, beat, note, intensity and playing state. No other component keeps its own competing
  copy (§38).
- It publishes **frozen snapshots**. Consumers (drums, AI, GUI) read snapshots and never mutate
  them.
- Start with the minimal §36 schema. Do **not** add sections, chords, genre, emotion or key until
  a step requires it (§37).
- Every estimate carries a **confidence** value. Decisions use it, e.g. no drums while tempo
  confidence is low (§79).
- Raw audio stays in the audio and analysis layers. Everything downstream works on semantic events
  and state (§24).

## Timing & units

- The master clock is the **audio stream's sample-frame index** (decision D5, §130). Schedule drum
  hits and AI entry in sample frames, computed ahead of time from the BeatClock (§78).
- Use `time.monotonic()` / `perf_counter` for non-audio timing. Use wall-clock time only for
  filenames (§129).
- Tempo and BeatClock change **gradually**. Never reset the clock on small BPM changes; use
  bounded phase correction (§33). Prefer reacting on musical boundaries (beat or bar) (§30).
- Put the unit in every name: `timestamp_seconds`, `duration_samples`, `frame_index`,
  `frequency_hz`, `bpm`, `gain_db`. Avoid bare `time`, `duration` and `position` (§131).
- Type aliases: `type MidiNote = int`, `type SampleFrame = int`.

## MRT2 integration facts (magenta-rt 2.0.3, MLX backend)

All model-specific code lives in `ai/mrt.py`, behind the `AccompanimentEngine` Protocol (§51, §52).
No other module imports `magenta_rt`.

- `from magenta_rt.mlx.system import MagentaRT2System`;
  `MagentaRT2System(size="mrt2_small")`. Loading and warmup are slow; do them at startup, off the
  audio thread.
- Models and resources live under `~/Documents/Magenta/magenta-rt-v2/` (override with
  `MAGENTA_HOME`). They are already downloaded (`mrt models init` / `mrt models download`).
- `embed_style(text)` returns a style embedding. Cache one per style preset.
- `generate(conditioning=…, cfg_scales=…, temperature=…, top_k=…, frames=N, state=state)` returns
  `(Waveform, state)`, where the waveform is **48 kHz stereo float32**.
  - Pass `state` back in for continuous streaming.
  - **1 frame = 40 ms = 1920 samples**; 25 frames = 1 s.
  - Conditioning is fixed for the whole call, so generate small chunks to update notes.
- Conditioning keys (use the constants from `magenta_rt.config`):
  - `MUSICCOCA.key`: the style embedding.
  - `PIANOROLL_WITH_ONSETS.key`: 128 ints per pitch. `-1` masked, `0` off, `1` held, `2` onset,
    `3` model's choice.
  - `DRUM_PIANOROLL.key`: 1 int. `-1` masked, `0` no drum, `1` drum. JamMate's own drummer plays
    the drums.
- `cfg_scales` keys are `musiccoca`, `notes` and `drums`, each in the range [-1, 7]. Defaults: 3.0,
  1.0, 1.0. Default temperature is 1.3, default `top_k` is 40. Follow↔Improvise maps onto these
  values.
- If the AI fails, JamMate degrades gracefully: bass monitoring and drums continue, and nothing
  crashes or leaves stuck state (§82, §137).

## Testing

- Unit-test all deterministic logic (§65):
  - Hz/MIDI/name conversions;
  - pitch stabilization;
  - onset and tempo math, half/double normalization;
  - BeatClock math and correction;
  - drum grid → frame scheduling;
  - Musical State updates;
  - note lifecycle (no hanging notes).
- Never `sleep` in tests. Use a fake clock or explicit sample positions (§66).
- Tests must not need audio hardware (§67). Keep the transport (`sounddevice`) separate from the
  logic. Test the callback logic by calling `BlockProcessor.process` with NumPy arrays. Test
  the engine and runner with the `fake_stream` fixture (`tests/conftest.py`), which drives
  `AudioEngine._callback` directly. Use
  `tests/signals.py` (synthetic bass and riffs) and short WAVs in `tests/data/`.
- The offline tools `scripts/analyze_wav.py` and `scripts/render_jam.py` let you check
  analysis and drums against recorded riffs without a live instrument.
- Realtime behavior is validated **only** by the developer playing a real bass (§62, §68).

## Code style

- Small modules, typed interfaces, frozen `@dataclass`es for events and snapshots, pure functions
  for math, explicit `start()` / `stop()` lifecycles, dependency injection only where it helps
  testing (§119).
- No abstract factories, deep inheritance, enterprise patterns or framework-like internal systems.
- Configurable values go in `config.py` (sample rate, block size, pitch range, thresholds, tempo
  range, look-ahead, buffer leads). Do not scatter magic numbers.
- Log useful diagnostics outside realtime code: device config, xruns, tempo lock/unlock, BeatClock
  corrections, AI lifecycle and errors. Never log per audio block (§80).

## Dependencies & licensing

- Before adding a dependency, check whether the stdLib, NumPy or SciPy already solve the problem,
  then check maintenance, transitive weight, removability and **license** (§133).
- No GPL-only dependencies without explicit approval. The product may become commercial (§134).
- `magenta-rt` code is Apache-2.0. Verify the **model-weight** license before any distribution.
- Drum samples in `assets/drums/` are development placeholders. Do not add random internet WAVs
  (§135).

## Do not build (unless explicitly requested)

- **Infrastructure:**
  - web backends (FastAPI, Django), REST APIs, WebSockets;
  - databases, Redis, Celery, Kafka;
  - Docker, Kubernetes, cloud services or cloud inference;
  - accounts, authentication.
- **Platforms:**
  - Windows, Linux, mobile, web;
  - VST/AU plugins or hosting.
- **Music features:**
  - guitar chord recognition;
  - full song generation;
  - notation and tabs;
  - DAW features;
  - custom model training.
- **Speculative work:**
  - any feature from the post-MVP backlog in the journal (riff detection, Band Director, key
    detection, …) before its step exists;
  - global architecture changes to solve a local problem;
  - replacing a deterministic algorithm with AI without showing the benefit (§141).

When unsure, choose the **simplest solution that keeps the audio callback safe**, and ask the
developer.

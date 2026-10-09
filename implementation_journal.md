# JamMate — Implementation Journal

This is the implementation plan and progress log for the JamMate MVP. JamMate is described in
[project_knowledge.md](project_knowledge.md), referenced below as `§N`. Agent rules are in
[AGENTS.md](AGENTS.md).

## How to use this journal

- `[-]` = not implemented, `[x]` = implemented.
- Mark a **step** `[x]` only when every item in its **Done when** list is satisfied, tests pass and
  ruff is clean.
- A step that needs a real instrument test stays `[-]` until the developer confirms the test. Note
  `awaiting hardware test` in the log.
- Mark a **phase** `[x]` only when all of its steps are `[x]`.
- Work one step at a time, in order. Do not start work from a later phase early.
- After finishing work on a step, add a dated entry to the [Journal log](#journal-log): what was
  done, deviations, measurements, and open issues.
- Record decisions that change this plan in [Decisions](#decisions) instead of silently editing
  the plan.

## Overview

| Phase | Goal | Milestone |
|---|---|---|
| 1. Foundation & Bass Audio Pipeline | Bass → Python → headphones, stable at 48 kHz | Bass Audio Pipeline (§111) |
| 2. Bass Understanding | Stable notes, onsets, intensity, tempo in Musical State | Bass Understanding (§112) |
| 3. Virtual Drummer | Drummer joins on a bar boundary and follows the player | Virtual Drummer (§113) |
| 4. First AI Band, UI & Recording | Bass + drums + MRT2 accompaniment, simple GUI, recording | First AI Band (§114) |

The development order is vertical (§60): each phase leaves the app runnable and testable with a
real bass. The GUI is built only after the musical loop works (Phase 4). Phases 1–3 run through a
headless CLI.

## Decisions

- **D1 — AI integration runs in-process via `magenta_rt.mlx` (`mrt2_small`).** This replaces the
  standalone host + MIDI bridge described in §18. Reason: `magenta-rt[mlx]` is already a project
  dependency and `mrt2_small` is downloaded to `~/Documents/Magenta/magenta-rt-v2`. The Python API
  accepts note conditioning directly as a pianoroll, so MIDI is not needed. The model still sits
  behind the `AccompanimentEngine` adapter (§51). Fallback if in-process generation causes audio
  xruns (the generation loop holds the GIL): first move generation to a subprocess with a
  shared-memory ring buffer, then fall back to the standalone MRT2 host + `mido`/`python-rtmidi`.
  `mido` and `python-rtmidi` are added only if that fallback is used.
- **D2 — A headless CLI runner is used until Phase 4.** `uv run jammate --headless …` drives the
  engine and prints a status line. The PySide6 GUI comes in Step 4.4.
- **D3 — Configuration uses typed pydantic models** in `jammate/config.py` (pydantic is already a
  dependency). User preferences are persisted as a small JSON file (Step 4.5). No database.
- **D4 — `jammate/session.py` (`JamSession`) is the single lifecycle owner.** It wires the audio
  engine, analysis, state, drums and AI, and implements the startup/shutdown flows (§83, §84).
- **D5 — The master timing reference is the audio stream's sample-frame index** (§130). All
  scheduling (drums, AI entry, BeatClock) is expressed in sample frames. `time.monotonic()` is used
  only for non-audio timing.

---

## [-] Phase 1 — Foundation & Bass Audio Pipeline

Milestone **Bass Audio Pipeline** (§111): the interface is detected, the bass is monitored through
the app, 48 kHz operation is stable, there are no clicks or dropouts in a sustained test, and the
callback is realtime-safe.

### [x] Step 1.1 — Project scaffolding & tooling

- Create the `src/jammate/` package with `__init__.py`, `main.py` and `config.py`. Add other
  subpackages only when a step needs them. Each new subpackage gets a short `AGENTS.md`
  covering its responsibility, invariants and the thread it runs on.
- `pyproject.toml`:
  - add `[build-system]` (`uv_build`);
  - add `[project.scripts] jammate = "jammate.main:main"`;
  - list `numpy` explicitly;
  - add a dev dependency group with `pytest` and `ruff`;
  - write a real project description.
- Configure ruff (target `py312`, lint and format) and pytest (`testpaths = ["tests"]`).
- `config.py`: typed `AudioConfig`:
  - `sample_rate=48000`;
  - `block_size=128` with fallback `256`;
  - input channel index, monitor gain.
  
  Further config sections are added by the steps that need them (§132).
- `main.py`: argparse (`--headless`, `--list-devices`, `--input`, `--output`, `--block-size`) and
  logging setup. Logging is never called from the audio callback.
- Add `recordings/` to `.gitignore`.

**Done when:**
- `uv sync` succeeds.
- `uv run jammate --help` works.
- `uv run pytest` passes (smoke test).
- `uv run ruff check .` and `uv run ruff format --check .` are clean.

### [-] Step 1.2 — Audio device discovery

- `audio/devices.py`:
  - frozen `AudioDevice` dataclass (name, index, host API, max input/output channels, default
    sample rate);
  - list input-capable and output-capable devices through `sounddevice`;
  - resolve devices by name or substring, never by a hard-coded index (§18);
  - check 48 kHz support with `sd.check_input_settings` / `sd.check_output_settings`.
- `uv run jammate --list-devices` prints the devices.
- Unit tests for filtering and name resolution use fake device lists (no hardware).

**Done when:**
- The USB interface (e.g. Scarlett) is listed with correct channel counts.
- Resolving it by name works.
- Tests pass.

### [-] Step 1.3 — Ring buffer & realtime audio engine (passthrough)

- `audio/ring_buffer.py`: preallocated float32 single-producer/single-consumer ring buffer.
  - Writes never block.
  - On overrun it drops the oldest data and increments a counter (§23).
  - The reader can consume sequentially or skip to the latest N frames.
  - A monotonically increasing `frames_written` counter gives the sample position.
- `audio/engine.py`: `AudioEngine` owns an `sd.Stream`:
  - duplex, 48 kHz, mono input from the selected channel, stereo output;
  - block size 128 with fallback 256;
  - explicit `start()` / `stop()` / `close()`.
- The per-block logic lives in a pure `process_block(indata, outdata)` class, testable without
  PortAudio. For each block it:
  - zero-fills the output;
  - copies the input into the ring;
  - writes the bass monitor (input × gain) to both channels;
  - advances the stream frame counter;
  - counts `input_overflow` / `output_underflow` from `status`.
- Callback timing is measured with `time.perf_counter_ns` into a preallocated array and read
  outside the callback.

**Done when:**
- Unit tests cover ring wrap-around, overrun and skip-to-latest, plus `process_block` on NumPy
  arrays.
- The §64 realtime checklist has been reviewed for the callback.

### [-] Step 1.4 — Headless runner, diagnostics & hardware validation

- `jammate --headless --input X --output Y` starts the engine. About twice per second the main
  thread prints a status line: xruns, ring fill, mean and max callback time.
- Ctrl+C performs an explicit clean shutdown: stop stream, close devices, join threads (§84).
- `scripts/capture_input.py` records raw bass input to WAV through the ring buffer. It is a
  developer tool, used to build real-riff fixtures for offline analysis in Phases 2–3.
- Manual test: 10 minutes of bass passthrough at 128 frames, then at 256 if xruns occur. Record the
  results in the journal log.

**Done when:** the Milestone 1 definition of done (§111) is confirmed on real hardware.

---

## [-] Phase 2 — Bass Understanding

Milestone **Bass Understanding** (§112): stable bass notes are displayed, onsets are detected,
tempo is estimated and Musical State receives updates.

### [-] Step 2.1 — Musical primitives, events & test signals

- `analysis/notes.py`: pure functions `hz_to_midi`, `midi_to_hz`, `midi_to_name` (`40 → "E2"`)
  and cents offset.
- `music/events.py`:
  - type aliases `MidiNote`, `SampleFrame`;
  - frozen dataclasses `PitchEvent`, `OnsetEvent`, `TempoUpdate`, `IntensityUpdate`,
    `SilenceEvent` (§24), with timestamps as sample frames plus seconds.
- `tests/signals.py`: synthetic signal generators for tests:
  - bass-like plucked tone (fundamental + harmonics + decay);
  - riff generator at a known BPM with a known note list;
  - silence and noise.
- A fake clock / sample-position helper for deterministic tests (§66).
- Store a few real riff recordings from `capture_input.py` in `tests/data/` (short, mono, 48 kHz).

**Done when:** unit tests pass for conversions (`82.41 Hz → 40 → E2`), events and signal
generators.

### [-] Step 2.2 — Analysis worker, pitch detection & stabilization

- `analysis/worker.py`: `AnalysisWorker` thread with explicit start/stop (no daemon threads).
  - Reads the ring buffer in hops (window and hop size are config values).
  - When it falls behind, it skips to the latest audio and counts the dropped frames (§23).
  - It measures processing time per hop.
- `analysis/pitch.py`: YIN pitch detection in the 35–400 Hz range (§15), with light band-limiting
  and a level gate.
  - Start with `librosa.yin`; switch to a NumPy YIN only if profiling shows it is too slow.
  - Returns frequency and confidence.
- Pitch stabilizer (§26): emits `PitchEvent` / `SilenceEvent` only when the note musically changes.
  It applies:
  - a confidence threshold;
  - a minimum note duration;
  - hysteresis in cents;
  - octave-error suppression;
  - a silence threshold.
- `scripts/analyze_wav.py` runs the analysis pipeline over a WAV file in block-sized chunks and
  prints the event timeline. It is the offline tuning tool (§67).

**Done when:**
- Synthetic and recorded E2/A2/D3/G3 are detected correctly with no octave flips.
- Silence returns `None`.
- The bass note test (§68) passes on real hardware.

### [-] Step 2.3 — Onset detection, intensity & playing state

- `analysis/onset.py` (§27):
  - detection from RMS change + spectral flux + adaptive threshold (median + k·MAD), with a
    refractory period;
  - emits `OnsetEvent` with a sample-frame timestamp and strength.
- Intensity (§47): RMS level + onset strength + note density, smoothed and normalized to 0.0–1.0.
- Playing state (§70): `user_is_playing` with a gate and hold time. It distinguishes a sustained
  note, the player stopping, a signal that is too quiet, and noise only.

**Done when:**
- Synthetic riff onsets are detected within ±1 hop.
- Intensity rises monotonically with playing level.
- Recorded riffs give plausible results in `analyze_wav.py`.

### [-] Step 2.4 — Tempo estimation & locking

- `analysis/tempo.py`, built mostly from pure functions (§28):
  - inter-onset intervals over a sliding window;
  - tempo candidates;
  - half/double normalization into the 50–200 BPM range;
  - rolling evidence;
  - `TempoState(bpm, confidence, locked)`.
- Gradual adaptation (§28): one unusual note never moves the BPM.
- Lock/unlock hysteresis (§29): lock after consistent evidence over N onsets, unlock only on
  sustained disagreement.
- Tests use synthetic onset lists with jitter, missing notes, syncopation, half-time feel and
  slow tempo drift.

**Done when:**
- Tests pass.
- In the stable riff test (§68), BPM stabilizes within roughly 2–4 s and does not jump.

### [-] Step 2.5 — MusicalStateManager & live monitor

- `music/state.py`:
  - frozen `MusicalState` snapshot following the §36 schema;
  - `recent_notes` is a bounded tuple;
  - beat, bar and phase stay `None` until Phase 3.
- `MusicalStateManager` is the single writer (§38):
  - controlled update methods;
  - publishes snapshots by atomic reference swap;
  - readers never block the writer or the audio thread (§39).
- `session.py`: `JamSession` wires the engine, analysis worker and state manager, and implements
  the startup and shutdown order (§83, §84).
- The headless status line shows note, BPM, confidence, locked state, intensity and playing state.

**Done when:**
- State manager unit tests pass.
- The Milestone 2 definition of done (§112) is confirmed live with a bass.

---

## [-] Phase 3 — Virtual Drummer

Milestone **Virtual Drummer** (§113):
- the player starts a riff;
- the tempo stabilizes and the BeatClock starts;
- drums enter on a bar boundary and stay synchronized;
- intensity changes affect the groove.

### [-] Step 3.1 — BeatClock & phase correction

- `music/beat_clock.py`: the clock is anchored to a sample frame and holds `bpm` and
  `beats_per_bar=4` (§31, §32). Pure conversions:
  - sample frame ↔ (bar, beat, phase);
  - next beat frame, next bar frame.
- Start-up: the clock starts from a locked tempo. Its phase is aligned to recent onsets. The
  downbeat guess is the strongest or most repeated onset position in the riff cycle.
- Gradual correction (§33): a small bounded per-onset phase correction (PLL-like). Tempo updates
  are smoothed and never reset the clock. A re-anchor happens only after a large error that
  persists.
- Beat, bar and phase are published to `MusicalState`.

**Done when:**
- Tests with sample positions pass:
  - no drift over 10 simulated minutes;
  - correction converges;
  - ±2 BPM estimator noise produces no jumps.
- The beat counter stays steady during the live riff test.

### [-] Step 3.2 — Drum sampler & output mixer

- `scripts/make_placeholder_kit.py`: generates synthetic kick, snare, closed hat, crash and tom
  into `assets/drums/`. These are license-safe placeholders (§135).
- `drums/sampler.py`:
  - at init, loads kit WAVs into float32 at 48 kHz, resampling with `scipy.signal.resample_poly`
    if needed (§45);
  - uses a preallocated voice pool.
- Event path (§46):
  - the scheduler thread writes into a bounded, preallocated event ring (frame, sample id, gain);
  - the callback starts voices at the exact offset inside the block and mixes active voices
    with vectorized NumPy and no allocation.
- Mixer in the engine:
  - monitor, drums and AI buses (AI silent for now), each with gain and mute set through atomic
    fields;
  - a safety soft-clip on the master.

**Done when:**
- Tests show that an event at frame N renders at the exact offset across block boundaries, and that
  voice-pool overflow is handled.
- A callback-cost micro-benchmark at 128 frames is within budget.
- A fixed test pattern plays click-free on hardware.

### [-] Step 3.3 — Pattern library & look-ahead scheduler

- `drums/patterns.py`:
  - 16-step grids with velocities (§44);
  - families for Rock, Stoner Rock, Blues (with a swing parameter), Funk and Metal, each with
    low/mid/high intensity variants;
  - a simple fill (tom, tom, snare, crash) (§49).
- `drums/engine.py`: the scheduler worker looks ahead (window size in config).
  - It converts upcoming grid steps into sample frames through the BeatClock and pushes each event
    exactly once (§78).
  - BeatClock corrections never cause double or missed hits.
- Add the `--bpm` override flag (§58). It bypasses tempo detection for debugging and doubles as
  the first manual escape hatch.

**Done when:**
- Tests pass:
  - grid → frame math at several BPMs;
  - no duplicate events across overlapping windows;
  - a tempo change in mid-bar.
- With `--bpm 92` the drums play cleanly alongside live bass.

### [-] Step 3.4 — Band entry, dynamics & silence behavior

- Drummer decision logic in `drums/engine.py`, kept separate from rendering (§125, §126). It is a
  state machine: `LISTENING → WAITING_FOR_BAR → PLAYING → HOLDING → STOPPED`.
- Entry (§30): drums enter on the next bar boundary once `tempo_locked` holds and confidence is at
  or above the threshold (§79), with a crash on entry.
- Dynamics (§48): intensity tiers select the pattern variant. Changes use hysteresis and apply only
  at bar boundaries. A fill plays every 8 bars.
- Silence (§71, all thresholds configurable):
  - 1 beat of silence: keep playing;
  - 2 or more bars: play lighter;
  - 8 or more bars: stop;
  - re-enter when the player resumes.

**Done when:** the state machine is unit-tested with synthetic `MusicalState` sequences
(entry, intensity changes, pauses, stop and resume).

### [-] Step 3.5 — Jam feel tuning & milestone validation

- `scripts/render_jam.py`: renders a recorded riff WAV through analysis and the drummer into a
  mixed WAV. This allows offline listening and tuning without playing live, and agents can use it.
- Measure the output latency reported by the stream. Add a configurable drum timing offset if the
  drums feel late.
- Tune thresholds (tempo lock, entry confidence, intensity tiers) on recorded and live riffs.
- Run the drum test (§68): 10 minutes of continuous playing.

**Done when:**
- The Milestone 3 definition of done (§113) is confirmed: no clicks, no major drift, no crashes,
  and the drums feel usable.
- The findings are logged.

---

## [-] Phase 4 — First AI Band, UI & Recording

Milestone **First AI Band** (§114): bass, drums and AI accompaniment run together, bass notes
condition the AI, and the user can jam for several minutes. This phase also covers Product UI
(§60 Phase 6) and Recording (§59).

### [-] Step 4.1 — Accompaniment interface, note lifecycle & graceful degradation

- `ai/base.py`: the `AccompanimentEngine` Protocol (§51). It has `start`, `stop`, `set_style`,
  `note_on`, `note_off`, `set_intensity` and `set_follow_improvise`, and exposes an AI output ring
  buffer (stereo float32, 48 kHz).
- The callback reads that ring into the AI bus. On underrun it outputs silence and increments a
  counter.
- `ai/null.py`: `NullAccompaniment` for when AI is disabled or unavailable.
- `NoteTracker` (§69) turns `PitchEvent` / `SilenceEvent` into `note_on` / `note_off`:
  - mono: at most one active note;
  - `note_off` always comes before the next `note_on`;
  - `note_off` on silence;
  - all notes off on stop and shutdown.
- Failure boundary (§82, §137): adapter exceptions are caught in the AI worker and the AI status
  becomes `failed`. Bass monitoring and drums keep running.

**Done when:** tests pass for the note lifecycle (no hanging notes), the null engine, AI-bus
underrun and the failure path.

### [-] Step 4.2 — MRT2 adapter (in-process `mrt2_small`)

- `ai/mrt.py`: `Mrt2Accompaniment` uses `magenta_rt.mlx.system.MagentaRT2System(size="mrt2_small")`.
  - The model loads and warms up during startup, before the audio stream opens (§83).
  - Style embeddings are cached per style.
- A generation worker thread loops `generate()` in small chunks of 40 ms frames (1920 samples)
  while keeping the streaming state. Conditioning is fixed for each call, so chunk size trades
  responsiveness against overhead. Per chunk, the conditioning is:
  - notes: a 128-slot pianoroll from `NoteTracker` (2 = onset frame, 1 = held, 0 = off);
  - drums: token 0, because JamMate's drummer plays the drums; try masked (-1) as an alternative;
  - CFG scales and temperature.
- The worker writes into the AI ring with a target lead buffer (configurable, e.g. 200–400 ms). It
  measures ms per frame and buffer level.
- Measure callback xruns with AI on and off. If the GIL causes degradation, apply the D1 fallback
  (subprocess + shared memory) and log it.
- If the model is missing or fails to load, the app falls back to `NullAccompaniment` and reports
  this in the status.

**Done when:**
- Bass, drums and AI run together for several minutes.
- Note changes audibly steer the AI.
- Killing or removing the model leaves bass and drums running.

### [-] Step 4.3 — Style presets, Follow↔Improvise & musical AI entry

- `music/styles.py`: structured `StylePreset` (§75) holding:
  - a short AI prompt;
  - drum pattern family;
  - fill frequency;
  - density, swing;
  - CFG and temperature ranges.
  
  Five presets: Rock, Stoner Rock, Blues, Funk, Metal.
- Follow↔Improvise (§74) is a 0.0–1.0 product parameter. It maps to notes and style CFG scales,
  temperature and drum variation. Keep the mapping in one place.
- AI entry (§30): the AI joins at a suitable boundary after the drums are in (e.g. 1–2 bars later).
  It fades or gates on long silence and follows intensity through `set_intensity`.
- Prompts stay short and are combined with structured state (§76).

**Done when:**
- Switching style changes both the drums and the AI.
- The Follow↔Improvise slider has an audible effect.
- Preset and mapping logic is unit-tested.

### [-] Step 4.4 — Product UI (PySide6 / Qt Widgets)

- Add the `PySide6` dependency. `ui/main_window.py` follows the §56 layout:
  - input and output device selectors;
  - style selector;
  - Drums ON/OFF + volume, AI ON/OFF + volume;
  - Follow↔Improvise slider;
  - START/STOP JAM;
  - detected note and BPM, 4-dot beat indicator;
  - plain-language status ("Listening…", "Playing", "AI unavailable") (§139).
- `ui/debug_panel.py`: a toggleable developer panel (§138) showing:
  - RMS;
  - pitch and confidence;
  - tempo, confidence and locked state;
  - beat, bar and phase;
  - intensity;
  - current drum pattern;
  - AI status, buffer level and ms/frame;
  - xruns.
- A `QTimer` (~30 Hz) polls the `MusicalState` snapshots and session metrics. The UI talks to
  `JamSession` only through thread-safe setters. No Qt calls are made from the audio or worker
  threads.
- The model loads in the background at launch with visible progress. Closing the window runs the
  full shutdown flow (§84).
- `uv run jammate` opens the GUI; `--headless` keeps the CLI.

**Done when:**
- A full jam can be started, controlled and stopped from the GUI.
- Closing the app leaves no running threads, stuck notes or open devices.

### [-] Step 4.5 — Recording & preferences

- `audio/recorder.py` (§59):
  - the callback copies the master stereo block into a bounded ring;
  - a writer thread streams it to `recordings/YYYY-MM-DD_HH-MM-jam.wav` with `soundfile` (wall-time
    filename, §129);
  - overflow is counted and never blocks;
  - the file is flushed on stop and shutdown.
- A Record toggle in the UI.
- Preferences (§85): a pydantic model saved as JSON in
  `~/Library/Application Support/JamMate/settings.json`, holding:
  - last input and output device;
  - block size;
  - style;
  - drums and AI volume.
  
  Loaded at startup, saved on change and exit.

**Done when:**
- Tests pass for the writer with a fake ring and for the preferences round trip.
- A recorded 5-minute jam plays back cleanly with no xruns caused by recording.

### [-] Step 4.6 — End-to-end jam validation (Go / No-Go)

- Run all manual scenarios (§68) plus continuous full-band jams of 10 minutes or more.
- Ask testers the §108 questions. Log metrics (xruns, AI ms/frame, latency) and the main things
  that break immersion.
- Make the Go / No-Go decision (§115): *"Is this fun?"*. Write the next-iteration priorities in the
  log.

**Done when:**
- The Milestone 4 definition of done (§114) is confirmed.
- The findings and the decision are recorded.

---

## Post-MVP backlog (not planned — do not implement without explicit request)

- Riff detection & session memory (§41, §98)
- Band Director (§72)
- Adaptive drummer that mirrors bass accents (§93)
- Humanization (§95)
- Key detection (§90)
- Manual overrides beyond `--bpm`: tap tempo, meter, reset (§58)
- Stem recording (§59)
- Native C++ MRT2 bridge (§19, §87)
- macOS packaging (§86)
- MIDI keyboard input (§55)
- Guitar input (§53)

---

## Journal log

Newest entries go at the bottom. Format: `### YYYY-MM-DD — Step X.Y — short title`, followed by
bullets covering what was done, deviations, measurements, and open issues / hardware tests pending.

### 2026-10-09 — Plan created

- Read `project_knowledge.md` and created this plan (4 phases, 20 steps) and `AGENTS.md`.
- Repository state: initial commit only. `pyproject.toml` declares `magenta-rt[mlx]`, `mlx`,
  `pydantic`, `scipy`, `sounddevice`, `soundfile`. `librosa` and `numpy` are installed as
  transitive dependencies.
- `mrt2_small`, `musiccoca` and `spectrostream` are already downloaded to
  `~/Documents/Magenta/magenta-rt-v2`.
- Decision D1 (in-process MRT2 instead of standalone host + MIDI) recorded above.

### 2026-10-09 — Step 1.1 — Project scaffolding & tooling

- Added `src/jammate/` with `__init__.py`, `config.py` and `main.py`, plus `tests/test_config.py`
  and `tests/test_main.py` (16 tests).
- `pyproject.toml` changes:
  - `uv_build` backend;
  - `jammate` console script;
  - explicit `numpy>=2.3.5`;
  - dev group with `pytest` 9.1.1 and `ruff` 0.16.10;
  - ruff config (py312, line length 100, rules `E W F I UP B SIM RUF`);
  - pytest config.
- `config.py`: frozen, `extra="forbid"` pydantic models `JamMateConfig` and `AudioConfig`:
  - `sample_rate_hz=48000`;
  - `block_size_frames=128`, `fallback_block_size_frames=256`, validated to never be smaller than
    the block size;
  - `input_channel_index=0`;
  - `monitor_gain=1.0` (linear, 0–4).
- `main.py`: argparse with `--headless`, `--list-devices`, `--input`, `--output`, `--block-size`
  and `--log-level`, plus logging setup. Config validation errors become argparse usage errors
  (exit 2). The not-yet-built modes log which step delivers them and exit 1.
- Additions beyond the step's spec:
  - `input_device_name` / `output_device_name` (`None` = system default) are in `AudioConfig`, so
    the CLI and, later, preferences (Step 4.5) feed one config object;
  - `--block-size N` raises the fallback to `max(N, 256)`, so an explicit block size never falls
    back to a smaller one;
  - a `--log-level` flag.
- Ruff 0.16 also formats Python snippets inside Markdown, which would rewrite
  `project_knowledge.md`. Markdown is excluded from `ruff format` (`[tool.ruff.format] exclude`).
- `uv` 0.7.6 labels the `uv_build` backend experimental. It builds the editable install without
  problems. Switch to hatchling if it ever causes trouble.
- Verified:
  - `uv sync`;
  - `uv run jammate --help`;
  - `uv run pytest` (16 passed);
  - `uv run ruff check .` and `uv run ruff format --check .` (clean).
- No realtime code yet; nothing runs on the audio thread. No hardware test needed for this step.

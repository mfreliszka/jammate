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
- **D6 — Notes are sounding pitch in scientific pitch notation (A4 = 440 Hz = MIDI 69).** A
  4-string bass's open strings are E1, A1, D2 and G2 (MIDI 28, 33, 38, 43; 41.2–98 Hz), an octave
  below a guitar's. §25's example (82.36 Hz → MIDI 40 → E2) is correct, but it is a guitar's low
  E: on a bass it is the E string at the 12th fret. §68's bass note test (E2 A2 D3 G3) names the
  same 12th-fret notes. Reason: MRT2's pianoroll needs the real pitch, and an octave mislabel would
  hide the octave errors the pitch tests must catch. Step 2.2 therefore tests both the open strings
  and the 12th fret. A 5-string's low B0 (30.9 Hz) is below §15's 35 Hz floor; revisit in
  Step 2.2 if the developer plays a 5-string.
- **D7 — Single-note electric guitar is a supported input and the development instrument.** The
  developer has no bass, so every hardware test is played on guitar. The Phase 1 tests were also
  played on guitar, although the log there says "bass". The product target stays bass (§7), and
  chord recognition stays out of scope (§9, §53). Changes:
  - Step 2.2 adds an `instrument` setting (`bass` | `guitar`, CLI `--instrument`) that selects
    the pitch range:
    - bass: 35–400 Hz (§15);
    - guitar: about 60–1,400 Hz, from drop tunings down to C2 up to the 24th fret.
    
    The default stays `bass`. The Makefile passes `guitar` for the developer.
  - At sounding pitch (D6), the guitar's open strings are E2 A2 D3 G3 B3 E4 (MIDI 40 45 50 55 59
    64), and the 12th fret is an octave higher. These are the §68 bass test notes, plus B3 and
    E4.
  - Riffs are played as single notes with the other strings muted.
  - Chords still produce onsets, intensity and tempo, so the drummer can follow, but they produce
    no notes. The pitch stabilizer must report no note rather than a wrong one.
  - Known trap: a power chord's root and fifth share a period of half the root frequency, so YIN
    can report a note an octave below the root. It is measured in Step 2.2 only if the optional
    power-chord recording is made; the recorded chord fixture uses open major chords.
  - Where the plan says "bass" for a live test, read "the input instrument".
  - In Step 4.3, style prompts should ask MRT2 for the parts a guitarist lacks, such as bass or
    keys.
- **D8 — Pitch detection uses an in-house NumPy YIN** (`analysis/pitch.py`) instead of
  `librosa.yin`. The Step 2.2 plan said to start with librosa. Reasons:
  - `librosa.yin` returns no confidence, and the plan requires one;
  - `librosa.pyin` runs an HMM over a whole clip, so it can't stream;
  - librosa is only a transitive dependency (through magenta-rt).
  
  The in-house YIN takes about 80 µs per window offline and adds no dependency.

---

## [x] Phase 1 — Foundation & Bass Audio Pipeline

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

### [x] Step 1.2 — Audio device discovery

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

### [x] Step 1.3 — Ring buffer & realtime audio engine (passthrough)

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

### [x] Step 1.4 — Headless runner, diagnostics & hardware validation

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

### [x] Step 2.1 — Musical primitives, events & test signals

- `analysis/notes.py`: pure functions `hz_to_midi`, `midi_to_hz`, `midi_to_name` (`40 → "E2"`)
  and cents offset.
- `music/events.py`:
  - type aliases `MidiNote`, `SampleFrame`;
  - frozen dataclasses `PitchEvent`, `OnsetEvent`, `TempoUpdate`, `IntensityUpdate`,
    `SilenceEvent` (§24), with timestamps as sample frames plus seconds.
- `tests/signals.py`: synthetic signal generators for tests:
  - bass- and guitar-like plucked tones (fundamental + harmonics + decay, D7);
  - riff generator at a known BPM with a known note list;
  - silence and noise.
- A fake clock / sample-position helper for deterministic tests (§66).
- Store a few real riff recordings from `capture_input.py` in `tests/data/` (short, mono, 48 kHz).
  *Moved to Step 2.2 by the developer (see the 2.1 sign-off log entry).*

**Done when:** unit tests pass for conversions (`82.41 Hz → 40 → E2`), events and signal
generators.

### [x] Step 2.2 — Analysis worker, pitch detection & stabilization

- `analysis/worker.py`: `AnalysisWorker` thread with explicit start/stop (no daemon threads).
  - Reads the ring buffer in hops (window and hop size are config values).
  - When it falls behind, it skips to the latest audio and counts the dropped frames (§23).
  - It measures processing time per hop.
- `analysis/pitch.py`: YIN pitch detection in the range of the `instrument` setting (bass
  35–400 Hz per §15, guitar about 60–1,400 Hz per D7), with light band-limiting and a level gate.
  - Start with `librosa.yin`; switch to a NumPy YIN only if profiling shows it is too slow.
  - Returns frequency and confidence.
- Pitch stabilizer (§26): emits `PitchEvent` / `SilenceEvent` only when the note musically changes.
  It applies:
  - a confidence threshold;
  - a minimum note duration;
  - hysteresis in cents;
  - octave-error suppression;
  - a silence threshold.
- Guitar fixtures (carried over from Step 2.1) were recorded on 2026-10-09. Their ground truth
  is in `tests/data/README.md`.
- `scripts/analyze_wav.py` runs the analysis pipeline over a WAV file in block-sized chunks and
  prints the event timeline. It is the offline tuning tool (§67).

**Done when:**
- With no octave flips (D6, D7), these are detected correctly:
  - synthetic bass and guitar notes;
  - the recorded guitar open strings (E2 A2 D3 G3 B3 E4) and 12th-fret notes (E3 A3 D4 G4 B4
    E5).
- Silence returns `None`.
- The recorded chord riff (`guitar_chords_e_g_a_100bpm.wav`, open major chords) produces no
  wrong notes.
- The note test (§68) passes on real hardware, played on guitar (D7).

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
  - half/double normalization into the 50–200 BPM range. Range alone can't decide for 8th-note
    riffs: at 92 BPM their pulse is 184/min, also inside the range (see the 2026-10-09 fixtures
    entry). A preference for the slower candidate or accent evidence must choose;
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

### 2026-10-09 — Step 1.2 — Audio device discovery (awaiting hardware test)

- Added `src/jammate/audio/` with `__init__.py`, `devices.py` and a module `AGENTS.md`:
  - frozen `AudioDevice`: index, name, host API, max input/output channels, default sample rate,
    default low input/output latency;
  - frozen `DeviceInventory`: devices plus system default indices, with `inputs` / `outputs` /
    `default(kind)`;
  - `inventory_from_portaudio()`, a pure conversion from `sounddevice` query results, and
    `query_inventory()`, which calls PortAudio;
  - `resolve_device(inventory, kind, query)`. `None` selects the system default. Otherwise a
    case-insensitive exact name wins, then a unique case-insensitive substring. A numeric query is
    text, never an index. Raises `DeviceNotFoundError` / `AmbiguousDeviceError` (both
    `DeviceResolutionError`) with the available or matching names;
  - `supports_settings()` wraps `sd.check_input_settings` / `sd.check_output_settings` (float32);
  - `format_device_list()` renders the terminal table.
- `jammate --list-devices` prints the input and output tables:
  - columns: channels, default rate, low latency, support for the configured 48 kHz rate;
  - the system default is marked `*`;
  - the table is followed by `Selected input/output: …`, which resolves `--input` / `--output`
    (exit 1 if a name can't be resolved).
  - Use this to verify name resolution.
- Tests: `tests/audio/test_devices.py` (fake PortAudio data including a Scarlett and its loopback
  device for the ambiguity case) and `--list-devices` tests in `tests/test_main.py`. 35 tests
  pass; ruff is clean.
- `tests/` and `tests/audio/` are now packages (`__init__.py`), so tests mirror `src/` and can
  import shared helpers such as the planned `tests/signals.py`.
- Checked on this Mac with only built-in devices (MacBook Pro Microphone/Speakers, Microsoft Teams
  Audio): the listing is correct, the defaults are marked, `--input macbook` resolves, and
  `--input o` reports ambiguity.
- Notes for later steps (also in `audio/AGENTS.md`):
  - PortAudio only sees hot-plugged devices after re-initialization;
  - `sd.CoreAudioSettings(channel_map=…)` selects the bass input channel;
  - `fail_if_conversion_required=True` prevents silent resampling (Step 1.3);
  - CoreAudio's settings check may report "yes" for rates it would convert.
- No realtime code; everything runs on the main thread before any stream opens.
- **Pending hardware test:** connect the USB interface, then run
  `uv run jammate --list-devices --input <name> --output <name>`. Confirm that the interface
  appears with the correct channel counts and that both selections resolve to it.

### 2026-10-09 — Step 1.2 — Hardware test passed

- The developer connected a USB guitar interface, which CoreAudio names **"USB Audio CODEC"**.
- `uv run jammate --list-devices` lists it as an input with 2 ch, default 48000 Hz and 4.4 ms
  low latency, and as an output with 2 ch, 48000 Hz and 3.4 ms. The system defaults are still the
  MacBook microphone and speakers.
- `--input/--output` values `usb`, `CODEC` and `usb audio codec` all resolve to the interface on
  both sides.
- Finding: the interface appears as **two PortAudio devices with the same name**: index 0 is
  output-only (0 in / 2 out) and index 1 is input-only (2 in / 0 out). Per-kind resolution handles
  this. Step 1.3 must open the duplex stream with `device=(input_index, output_index)`. Both
  devices share the physical clock, so drift between them is not expected.
- Finding: `sd.check_*_settings` accepts 44.1, 48 and 96 kHz even with
  `CoreAudioSettings(fail_if_conversion_required=True)`. The "48000 Hz: yes" column therefore
  only proves direction and channel compatibility. Short input-only test streams (0.2 s, no
  output) opened at all three rates. The reported input latency was:
  - 48 kHz: 16.4 ms, the lowest, which suggests 48 kHz is the native rate;
  - 44.1 kHz: 29.5 ms;
  - 96 kHz: 48.2 ms.
  
  Step 1.3 must verify the rate from the opened stream (`stream.samplerate`, `stream.latency`)
  and log both.
- Input RMS during the probe was about 7e-5 (around −83 dBFS): the noise floor, with nothing
  played.
- Open for Step 1.3: the interface exposes 2 input channels, so find out which one carries the
  instrument (`input_channel_index`, default 0).

### 2026-10-09 — Step 1.3 — Ring buffer & realtime audio engine

- `audio/ring_buffer.py`: `RingBuffer(capacity_frames, channels)`, float32 SPSC with absolute
  frame positions.
  - `write()` (producer): never blocks, never inspects the consumer, and overwrites the oldest
    data when full.
  - `read()`: sequential; after an overrun it resumes at the oldest valid frame and counts
    `overrun_count` / `dropped_frames`.
  - `read_latest()`: skip-to-latest. It allows overlapping windows, and skipped frames count as
    dropped but not as overruns.
  - Both reads return `RingRead(start_frame, frame_count)`.
  - NumPy can release the GIL during copies. So the producer announces the range it is about to
    overwrite before copying (`_write_end`), and the reader re-validates after its copy and
    discards torn frames. A test covers this.
- `audio/engine.py`:
  - `BlockProcessor` holds the per-block work for the PortAudio thread:
    - copies the selected input channel into the ring;
    - monitor = input × `monitor_gain` into every output channel, fully writing `outdata`;
    - frame and block counters;
    - overflow/underflow counters;
    - exceptions produce a silent block plus `errors`;
    - `perf_counter_ns` durations go into a preallocated 2048-block window, read with
      `timing()`.
  - `AudioEngine` handles the lifecycle (main thread):
    - `start()` opens a duplex `sd.Stream` with `device=(in, out)`,
      `channels=(input_channel_index + 1, min(2, out))`, 48 kHz, float32, `latency="low"`. It
      tries 128 frames, then the 256 fallback;
    - it refuses a stream whose actual sample rate differs (§141);
    - `stop()` pauses, `close()` releases the devices, `start()` after `stop()` resumes;
    - it is a context manager;
    - it validates the input channel and output device;
    - `set_monitor_gain()`, `stats()` → `EngineStats`, `StreamInfo` (actual rate, block size,
      latencies).
- Config: `AudioConfig.input_ring_seconds=2.0` → `input_ring_capacity_frames` (96,000). Validated
  to hold at least one fallback block.
- Measured on the developer's "USB Audio CODEC", with silent test streams (monitor gain 0):
  - PortAudio-reported latency at 128 frames: 16.4 ms in + 7.4 ms out (about 24 ms round
    trip); at 256: 29.8 + 12.8 ms;
  - `change_device_parameters=True` made no difference, so it isn't used;
  - callback arrival every 2.67 ms, p99 gap 2.74 ms; max 3.50 ms with a consumer thread reading
    every 10 ms;
  - `BlockProcessor.process`: 1.8 µs mean isolated, about 9 µs mean / under 45 µs max live,
    against a 2,667 µs block period;
  - the consumer thread kept up with no ring overruns.
- Finding: every stream start reports an input overflow on block 0 (sometimes block 1). This is
  a CoreAudio start-up transient. Xruns in the first 50 ms after each start are now counted
  separately as `startup_xruns` (`STARTUP_SETTLE_SECONDS`), so they don't mask real dropouts.
- Finding: sporadic mid-run input overflows were seen: 2 in about 90 s of test runs, and none in
  a dedicated 2 × 20 s comparison with and without a consumer. The cause is still open (GIL
  jitter vs PortAudio's bridging of two CoreAudio devices). The Step 1.4 10-minute test must
  report the rate.
- Risk for Step 1.4: about 24 ms reported round trip for software monitoring is more than you'd
  want for bass (§77). If it feels late, options are (a) the interface's direct/hardware
  monitoring with `monitor_gain=0`, or (b) a measured loopback test to confirm the real figure.
- §64 checklist reviewed for `_callback` → `BlockProcessor.process` → `RingBuffer.write`:
  - no logging, I/O, MIDI, Qt, inference, librosa, locks, waits or sleeps;
  - no sample allocations (only NumPy views and ints);
  - the ring is bounded;
  - exceptions are contained;
  - xruns are counted.
- Tests: `tests/audio/test_ring_buffer.py` and `tests/audio/test_engine.py`, using a fake
  `sd.Stream` for the lifecycle, fallback, sample-rate refusal and settling window. Plus config
  tests. 65 tests pass; ruff is clean.
- Not done here (Step 1.4): headless runner, status line, Ctrl+C shutdown, `capture_input.py`,
  and listening to the monitor with a real instrument.

### 2026-10-09 — Step 1.4 — Headless runner & capture tool (awaiting hardware test)

- `src/jammate/headless.py`: `run_headless(config)`, used by `jammate --headless`.
  - Resolves devices via `AudioEngine.from_config()` and starts the engine.
  - Every 0.5 s the main thread drains the input ring (its only consumer until Step 2.2) and
    prints a status line:
    - elapsed time;
    - input RMS and peak in dBFS, with `CLIP` at -0.1 dBFS or above;
    - mid-run xruns in/out, plus start-up xruns;
    - ring fill;
    - callback mean/max µs and load % of the block period;
    - `CALLBACK ERRORS` if any.
  - On a TTY it rewrites one line, otherwise prints one line per update.
  - Ctrl+C or SIGTERM triggers an explicit shutdown. A second Ctrl+C is ignored during shutdown,
    the stream is closed, the devices released and the signal handlers restored. A session
    summary is logged (duration, blocks, xruns, whole-run callback max, input peak, errors).
  - If the stream stops on its own (interface unplugged), it exits 1.
- `AudioEngine.from_config(audio)` resolves the configured device names and builds the engine.
  It is shared by the runner and the capture script. Channel numbers in messages are now 1-based.
- New CLI options:
  - `--input-channel N`: 1-based, as printed on interfaces; 0 is a usage error;
  - `--monitor-gain G`: 0–4; use 0 for silent tests or the interface's direct monitoring.
- `scripts/capture_input.py PATH [--input --output --input-channel --monitor-gain --seconds
  --subtype --overwrite]`:
  - records raw input through the engine's ring with the main thread draining to `soundfile`
    (PCM_24 by default), while still monitoring;
  - refuses to overwrite without `--overwrite`;
  - deletes the file if the stream fails to start;
  - warns if ring overruns left gaps.
  - Added `scripts/AGENTS.md`.
- Tests:
  - shared fakes in `tests/fakes.py` (devices, `FakeStream`) and a `fake_stream` fixture in
    `tests/conftest.py`;
  - `tests/test_headless.py`: real `AudioEngine` over `FakeStream`, with a fake sleep that feeds
    callback blocks then simulates Ctrl+C;
  - new CLI option tests in `tests/test_main.py`;
  - `tests/scripts/test_capture_input.py`: the script is loaded by path.
  - 86 tests pass; ruff is clean.
- Live check on "USB Audio CODEC" with `--monitor-gain 0` (silent, nobody playing):
  - 6 s headless run: 1 start-up xrun, 0 mid-run xruns, callback max 70 µs (3% load), ring fill
    steady at 25%, input channel 1 at about -90 dBFS RMS (noise floor);
  - SIGINT gave a clean shutdown and exit 0;
  - a 3 s capture produced a valid 48 kHz mono PCM_24 WAV.
- **Pending hardware test (developer):**
  1. Plug the bass into the interface and wear headphones.
  2. Find the channel: `uv run jammate --headless --input usb --output usb`, then play. If the
     level stays around -90 dBFS, stop and retry with `--input-channel 2`. Aim for peaks between
     about -12 and -3 dBFS without `CLIP`, using the interface's gain knob.
  3. Play for 10 minutes at the default 128-frame blocks. Watch `xruns in/out` and listen for
     clicks or dropouts. If xruns grow, repeat with `--block-size 256`.
  4. Judge the monitoring latency (about 24 ms reported round trip). If the bass feels late, try
     the interface's direct monitoring with `--monitor-gain 0`.
  5. Report the session summary line, the channel, any clicks, and how the latency felt.
- Optional: record 2–3 short riffs (10–20 s each, steady tempo, known notes/BPM) for Phase 2
  fixtures, e.g.
  `uv run python scripts/capture_input.py tests/data/riff_e_g_a_92bpm.wav --input usb --output usb --seconds 20`.
- Hardware test, part 1 (developer, output to MacBook speakers):
  - The bass is on **input channel 1** (the default). Channel 2 stays at about -90 dBFS.
  - The first cable was faulty. Symptoms: only buzz audible; about -40 dBFS RMS with hands off,
    and playing peaks of only about -49 dBFS; buzz louder when touching the strings, which
    suggests a broken ground.
  - After changing the cable: idle -64.6 dBFS RMS (peak -52.6), playing -14.4 dBFS RMS
    (peak -3.8), no xruns.
  - Troubleshooting rule: if the idle level is above about -55 dBFS RMS, or the hum changes when
    you touch the strings, check the cable and jack first.
- Hardware test, part 2 (developer, bass on input channel 1):

  | Output | Duration | Blocks | Mid-run xruns | Start-up xruns | Callback max | Input peak |
  |---|---|---|---|---|---|---|
  | MacBook Pro Speakers | 2:50 | 63,779 | 0 | 0 | 79 µs (3%) | -2.8 dBFS |
  | USB Audio CODEC (interface) | 1:56 | 43,508 | 1 input overflow | 1 | 76 µs (3%) | -3.0 dBFS |

  Both runs had 0 callback errors. Idle noise floor was about -63 to -65 dBFS RMS. Shutdown was
  clean in both runs.
- **Deviation accepted by the developer:** these two runs (about 4.75 min total) replace the
  planned single 10-minute run as the Milestone 1 sign-off. No clicks or latency complaints were
  reported. **Step 1.4 and Phase 1 are complete.**
- Carry-over items:
  - **Sporadic mid-run input overflow with interface in + interface out.** Seen 1 in 116 s here
    and about 2 in 90 s during Step 1.3 probes, but 0 in 170 s with output to the MacBook
    speakers. The callback is not the cause (max 79 µs of 2,667 µs). It probably comes from
    PortAudio bridging the interface's two same-named CoreAudio devices. Re-check the rate once
    the analysis worker adds load (Step 2.2). If clicks become audible, try a larger stream
    latency or a macOS aggregate device.
  - Playing peaks of about -3 dBFS leave little headroom. Lower the interface gain so normal
    playing peaks at about -10 to -6 dBFS.
  - Monitoring latency feel was not assessed explicitly. The reported figure is about 24 ms
    through the interface and about 71 ms with speaker output. Revisit if it bothers the player.

### 2026-10-09 — Step 2.1 — Musical primitives, events & test signals (awaiting riff recordings)

- `music/events.py` (new `music/` package):
  - `type MidiNote = int`, `type SampleFrame = int`, MIDI range constants;
  - `frames_to_seconds()` / `seconds_to_frames()` (half-up rounding, no banker's rounding);
  - frozen, keyword-only events `PitchEvent(midi_note, frequency_hz, confidence)`,
    `OnsetEvent(strength)`, `TempoUpdate(bpm | None, confidence, locked)`,
    `IntensityUpdate(intensity)` and `SilenceEvent`. Each also carries `frame_index` and
    `timestamp_seconds`, and validates its values on construction (`ValueError`);
  - `type MusicalEvent` is the union of all five.
- `analysis/notes.py` (new `analysis/` package): `hz_to_midi_float`, `hz_to_midi` (nearest, half
  up), `midi_to_hz`, `midi_to_name` (sharps), `name_to_midi` (sharps, flats, lowercase) and
  `cents_offset` (against the nearest or a given note). Out-of-range notes and non-positive
  frequencies raise `ValueError`. NumPy integers are accepted; float note numbers raise
  `TypeError`.
- Deviation from §24: `PitchEvent` has no `note_name` field. A stored name could disagree with
  `midi_note`, and deriving it inside `events.py` would create an import cycle with `notes.py`.
  Consumers call `midi_to_name(event.midi_note)`.
- Plan change **D6** (sounding pitch; a bass's open strings are E1 A1 D2 G2). Step 2.2's
  *Done when* now names both the open strings and the 12th-fret notes.
- `tests/signals.py`:
  - `plucked_note()`: fundamental plus harmonics, with harmonic k decaying k times faster, a 2 ms
    attack and a 5 ms release (click-free). It has options for detuning (`cents`), a weak
    fundamental (to provoke octave errors) and peak amplitude. Harmonics above Nyquist are
    skipped;
  - `riff(steps, bpm, …)`: note names or MIDI numbers, `None` for rests, `(note, beats)` for
    custom lengths, plus `repeats`, `gate`, lead-in and tail. It returns `Riff` with the samples
    and exact ground truth (`onset_frames`, `end_frames`, `midi_notes`). Onsets are rounded per
    beat, so there is no drift at fractional frames-per-beat (92 BPM: 400 beats checked);
  - `silence()`; `noise()`: seeded white noise at an exact RMS. Its default of -65 dBFS is the
    interface's measured idle floor.
- `tests/fakes.py`: `FakeClock` (§66) counts whole sample frames, so `frame_index` and `seconds`
  always agree. It can be called like `time.monotonic`, and `clock.advance` stands in for
  `time.sleep`.
- New AGENTS.md files for `analysis/` and `music/`. The root AGENTS.md now covers note naming
  (D6), the event time fields and the test helpers. `tests/data/README.md` defines the fixture
  format, naming and a ground-truth table.
- No realtime code. All of this runs on worker threads or in tests; the audio callback is
  untouched.
- Tests: `tests/analysis/test_notes.py` (including all 128 notes round-tripping, the bass
  strings in both registers, and 50-cent rounding), `tests/music/test_events.py`,
  `tests/test_signals.py` (FFT pitch checks, onset/rest/gate ground truth, drift, noise level)
  and `tests/test_fakes.py`. 167 tests pass; ruff is clean.
- **Done when** is met: `82.41 Hz → 40 → E2` and the event and signal-generator tests pass.
- **Open (developer):** the step also asks for real riff recordings in `tests/data/`, and none
  exist yet. Step 2.2's *Done when* needs them. The step stays `[-]` until they are recorded;
  record them with `make capture` as listed in the report and add each one to
  `tests/data/README.md`.

### 2026-10-09 — Step 2.1 — Guitar as the development instrument (D7)

- The developer plays electric guitar and has no bass. Plan change **D7**:
  - single-note guitar is a supported input;
  - chords give rhythm but no notes;
  - the `instrument` setting arrives in Step 2.2.
  
  Steps 2.1 and 2.2 are updated to match.
- `tests/signals.py`: added a `GUITAR_HARMONIC_AMPLITUDES` preset and a `harmonic_amplitudes`
  option on `riff()`. In the preset the 2nd harmonic is stronger than the fundamental, as is
  common on an electric guitar's low strings (especially on the bridge pickup). In the first
  ~100 ms after an onset, the window pitch detection sees first, the spectral peak therefore sits
  an octave up; the fundamental takes over as the harmonics decay. That is the octave-error case
  Step 2.2 must handle.
- `analysis/notes.py` docs and tests now cover the guitar's open strings and 12th fret.
- `tests/data/README.md`: fixtures are guitar recordings named
  `<instrument>_<content>[_<bpm>bpm].wav`. The recording plan is clean DI (no pedals or amp sim),
  single notes, unused strings muted.
- **Open (developer):** record the guitar fixtures (commands in the report). Step 2.1 stays `[-]`
  until they exist.

### 2026-10-09 — Step 2.1 — Signed off by the developer

- The developer asked for Step 2.1 to be marked done without the real recordings. **No
  recordings were made**, and `tests/data/` holds only its README. Everything else in the step
  is implemented, and the *Done when* tests pass (181 tests, ruff clean). **Step 2.1 is
  complete.**
- Carry-over: the guitar recordings move to Step 2.2. Its *Done when* checks need them: the
  recorded open strings, the 12th fret and the power-chord riff. The list and commands are under
  *To record* in `tests/data/README.md`.

### 2026-10-09 — Guitar fixtures recorded (3 of 5, for Step 2.2)

- The developer recorded `guitar_open_strings.wav`, `guitar_12th_fret.wav` and
  `guitar_riff_e2_g2_a2_92bpm.wav`. All three are 48 kHz mono PCM_24, 20.1–20.2 s long, with
  sample peaks of -2.9 to -3.5 dBFS and no clipping. The ground truth is in
  `tests/data/README.md`.
- I checked them with a throwaway script, not project code, using `librosa.pyin` (60–1,400 Hz,
  5 ms hop) and `librosa.onset`.
- **Held notes.** All 12 notes were detected as the right note with no octave errors, including
  in the first 100 ms after the pluck. The held notes are 98–100% voiced and 98–100% on the
  note. The tuning sits within -19 to +31 cents, at pYIN's 10-cent resolution.
- **Riff.** It has 48 real notes (E2×8, G2×8, A2×8, played twice), each pitched correctly. A
  linear fit gives 91.9 BPM in both halves of the riff, with 21 ms SD of human timing and 53 ms
  at most.
- Findings for Steps 2.2–2.4:
  - **Attack pitch glide** (2.2): the 12th-fret D4 reads D#4 for its first ~100 ms, then settles
    about 30 cents sharp. The stabilizer must not commit to a note during the attack, and its
    cents hysteresis must hold a note that is 30 cents sharp.
  - **Ringing open strings** (2.2, 2.3) sit at about -49 to -55 dBFS between notes and before
    playing, with real pitches (A2, B3, E4, G3). The electrical floor is about -62 dBFS. A
    fixed -60 dB gate would report the ringing strings as notes. The level gate needs to sit
    above them, or be relative to recent playing.
  - **Spurious onsets** (2.3):
    - handling noise before playing (about -54 dBFS, with bumps to -36);
    - the first note split into two onsets 70 ms apart;
    - a ghost onset before the first A2 of each cycle, at the same spot every time (slide or
      position-shift noise).
    
    A refractory period plus a level-relative strength threshold must reject these.
  - **Half/double tempo** (2.4): `librosa.beat.beat_track` reports 184.6 BPM, the 8th-note
    pulse. On-beat and off-beat 8ths have similar onset strength (2.7 vs 3.1), so accents can't
    break the tie. The plan text of Step 2.4 now notes this.
  - **Capture artifacts**: every file starts with 512 frames of exact zeros and ends with 3,328,
    from CoreAudio stream start and close. It is harmless silence, but the analysis must not treat
    the zeros-to-noise step at the start as an onset. librosa reported one at 0.025 s.
- Still to record: a 120 BPM single-note riff and a power-chord riff (see *To record* in
  `tests/data/README.md`).

### 2026-10-09 — Guitar fixtures complete (5 of 5, for Step 2.2)

- The developer recorded `guitar_riff_e2_g2_a2_120bpm.wav` and a chord riff. Both are 48 kHz mono
  PCM_24 with peaks around -3 dBFS and no clipping. They have the same 512/3,328-frame zero
  edges as the other fixtures. I checked them with the same throwaway pYIN/onset scripts.
- **120 BPM riff.** It has 48 notes (E2×8, G2×8, A2×8, played twice), each pitched correctly. A
  linear fit gives 119.2 BPM, with 20 ms timing SD and 59 ms at most. This time
  `librosa.beat.beat_track` reports 118.8, the quarter-note pulse; at 92 BPM it chose the 8th
  pulse.
- **The chord riff contains open major chords, not power chords.** Chroma shows E = E G# B,
  G = G B D and A = A C# E. I renamed the file from `guitar_chords_power_100bpm.wav` to
  `guitar_chords_e_g_a_100bpm.wav`. It holds 8th-note strums, E×8 G×8 A×8 twice, and the take
  stops mid-chord at 15.07 s.
  - Tempo: an onset fit gives 99.8 BPM, and the beat tracker 100.8. The strum timing SD is 48 ms,
    and librosa's onset detector missed about 4 of 47 strums.
  - pYIN on the chords is often unvoiced (0–50% of frames), which is the behavior we want. But it
    also gives confident wrong notes: C3 at 100% voiced on G chords, and C#3, F#2 and D3 on A
    chords. E chords mostly read the root, E2.
- Findings for Steps 2.2–2.4, on top of the 3-of-5 entry:
  - **Slides** (2.2): in both riffs the last G of each cycle slides to A2 in about 100 ms,
    passing through G#2. The stabilizer's minimum note duration must skip the pass-through. The
    "ghost onset" in the 92 BPM riff is this slide.
  - **Chords** (2.2): a voicing threshold alone won't keep wrong notes out. The stabilizer needs a
    single-pitch check (stable f0 over the minimum duration, with confidence), and the chord
    fixture is the test.
  - **35 ms double onsets** from pick contact (2.3): a refractory period of 50 ms or more covers
    them.
  - **Half/double tempo** (2.4): the beat tracker chose the 8th pulse at 92 BPM but the quarter
    at 120. Our estimator must pick the quarter both times.
- The power-chord octave trap (D7) is unmeasured. An optional recording command is in
  `tests/data/README.md`.

### 2026-10-09 — Step 2.2 — Analysis worker, pitch detection & stabilization (awaiting hardware test)

- **Config:** new `AnalysisConfig`, `JamMateConfig.analysis`.
  - Instrument and pitch range (D7): `instrument` (`bass` | `guitar`), with
    `pitch_min_hz` / `pitch_max_hz` overrides.
  - Detection: `hop_frames=480` (10 ms), `level_gate_dbfs=-45`, `yin_threshold=0.2`.
  - Stabilizer:
    - `min_note_seconds=0.07`, `stability_cents=35`, `tuning_tolerance_cents=40`;
    - `hysteresis_cents=70`, `octave_confirm_seconds=0.15`, `silence_hold_seconds=0.06`;
    - `new_note_gate_db=15`, `playing_level_release_db_per_second=3`.
  - Worker: `max_backlog_seconds=0.1`.
  - Validators: min < max pitch; hysteresis > tuning tolerance; max pitch < a quarter of the
    sample rate.
- **`analysis/pitch.py`:**
  - In-house NumPy YIN (D8): FFT difference function, cumulative mean normalized difference,
    first dip below the threshold, parabolic interpolation. Confidence = 1 − aperiodicity.
  - `PitchDetector`: a stateful 2nd-order band-pass from 0.5 × the min to 2 × the max pitch.
  - The window is twice the longest period plus 2 frames: guitar 1,602 frames (33 ms), bass
    2,746 (57 ms).
  - The gate uses the band-limited window RMS. It emits a `PitchFrame` per hop: frequency or
    None, confidence, RMS, peak.
- **`analysis/stabilizer.py` — `NoteStabilizer` (§26):** a new note needs 70 ms of contiguous
  voiced hops that meet all of the following:
  - spread ≤ 35 cents, which rejects glides, slides and attacks;
  - median within 40 cents of one note, so nothing is committed between notes;
  - no more than 15 dB below the recent playing level. That level follows the RMS up at once
    and falls back 3 dB/s.

  Other rules:
  - **Hysteresis:** the current note holds while the pitch stays within 70 cents.
  - **Octave jumps** need 150 ms.
  - **Re-picking** the current note emits nothing.
  - **Silence:** 60 ms unvoiced emits a `SilenceEvent`.
  - **Quiet sounds** never end the current note: harmonics left after muting, open strings
    ringing after a fretted note is released, touch noise.
  - `PitchEvent.frame_index` is the first hop of the evidence.
- **`analysis/worker.py`:**
  - `AnalysisWorker` is the ring's only consumer. It has explicit `start()` / `stop()` (joined,
    not a daemon) and cuts the input into hops.
  - Overruns and falling behind:
    - after a ring overrun, it restarts the hop at the gap;
    - with more than 0.1 s of unread input, it skips to the latest audio (§23) and counts the
      skip; the ring counts the dropped frames.
  - Error handling: errors are counted, and only the first is logged; the thread keeps running.
  - `stats()` reports:
    - hops, plus mean and max time per hop;
    - recent and maximum peak, and clipped hops;
    - the latest `PitchFrame` and the current note.
  - `analyze_offline()` runs the same path over an array, in 128-frame blocks through a ring,
    without a thread.
- `audio/levels.py`: dBFS helpers moved out of `headless.py`.
- **Headless runner:**
  - It wires the worker to the engine. Events reach the main thread through a bounded `deque`.
  - Every note change prints as a line within 50 ms; the status line keeps redrawing below.
  - The status line shows:
    - the current note and level;
    - xruns, ring fill and callback time;
    - analysis time per hop, plus `DROPPED` or `ANALYSIS ERRORS` when they happen.
  - Shutdown stops the worker, then the stream (§84). The summary adds an analysis line.
- CLI: `--instrument`. Makefile: `INSTRUMENT ?= guitar` for `run`, plus a new `analyze FILE=…`
  target.
- `scripts/analyze_wav.py PATH [--instrument --channel --block-size --hops --level-gate
  --yin-threshold]` prints the event timeline with the decision time per event, and a summary.
- **Tuning on the guitar fixtures** (`analyze_wav.py` plus a throwaway scoring script). Each
  default came from a measured case:

  | Change | Case it fixed |
  |---|---|
  | Relative new-note gate (15 dB, 3 dB/s) | 2 extra notes in the open strings and 4 in the 12th fret: harmonics left after muting (D3→D5, A2→A4), a touched low E (-37 dBFS), open strings ringing after a fretted note was released |
  | Applying that gate only to *new* notes | Inside the detector it would also cut off naturally decaying notes |
  | `min_note_seconds` 0.05 → 0.07 | The 92 BPM slide paused on G#2 for 60 ms |
  | `yin_threshold` 0.15 → 0.2 | Held riff notes no longer split into note/silence fragments |
  | Gate on the band-limited RMS | A DC step was reported as a confident A#5 |

- **Results:**

  | Fixture | Detected |
  |---|---|
  | Open strings, 12th fret | Exactly the 6 notes each, in order, every one followed by silence |
  | Both riffs | Exactly E2 G2 A2 E2 G2 A2, no G#2 |
  | Chords | One A2 (the root of the A chord), no wrong notes |
  | Synthetic bass E1–G3, guitar E2–E5 (guitar timbre) | All correct |
  | Synthetic riffs | Correct, with no repeated-note events |
  | Silence, noise | Nothing |

  The decision latency is about 61 ms after the first stable hop, and a note is reported about
  50–150 ms after its pick. Offline, an analysis step takes about 100 µs.
- **Realtime implications:**
  - The callback is unchanged. All analysis runs on the `jammate-analysis` thread, which only
    reads the ring.
  - The GIL is the risk. I measured it live on the built-in microphone and speakers, at 128
    frames with silent monitoring and YIN forced on every hop, 30 s per run.
  - With no worker, callback arrival gaps were p99 2.70 ms, max 3.12 ms.
  - With the worker waking every 5 ms: p99 3.44, p99.9 6.12, max 9.66 ms. Analysis steps took
    1.2 ms mean live against 0.12 ms offline.
  - **The cause:** after a 5 ms sleep, every NumPy/SciPy call ran 8–30× slower. Apple Silicon
    parks a mostly idle thread.
  - Things that didn't help:
    - `sys.setswitchinterval` (1 ms, 0.5 ms);
    - QoS `USER_INITIATED`;
    - QoS `USER_INTERACTIVE`, which once gave 11 input overflows.
  - **The fix:** wake every 1 ms (`POLL_SECONDS`). Result: p99 2.73, p99.9 3.92, max 7–9 ms
    (rare). There were no xruns in any worker run without QoS.
  - Remaining risk: rare 7–9 ms callback delays against the interface's 7.4 ms output latency.
    The hardware test must report xruns. If they appear, the fallbacks, in order:
    - `--block-size 256`;
    - a larger stream latency;
    - move analysis to a subprocess with a shared-memory ring (like the D1 fallback).
- **Live smoke test** (built-in mic, silent): the runner started, the MacBook mic picked up a
  room sound as C3 and printed it, then silence. A SIGINT sent directly to the app gave a clean
  shutdown in 0.31 s, exit 0. (Sent only to `uv run`, the signal isn't forwarded; Ctrl+C in a
  terminal reaches both processes.)
- **Tests** (267 pass, ruff clean):
  - `tests/analysis/`: `test_pitch.py`, `test_stabilizer.py` (every rule, on hand-made hop
    sequences), `test_worker.py` (hops, skip, overrun, thread lifecycle, error survival,
    synthetic end-to-end) and `test_recordings.py` (all five fixtures);
  - `tests/audio/test_levels.py`;
  - `tests/scripts/test_analyze_wav.py`;
  - the headless tests, rewritten: a real worker without a thread, plus `FakeClock`.
- **Done when** status:
  - [x] synthetic bass and guitar notes;
  - [x] the recorded open strings and 12th fret;
  - [x] silence returns `None`;
  - [x] the chord riff gives no wrong notes;
  - [ ] the note test (§68) on real hardware: **awaiting the developer**.
- **Known limitations:**
  - A new note more than 15 dB softer than the recent playing is ignored until the playing level
    falls (up to ~5 s), as for a sudden ff → pp. Revisit in Step 2.3: a strong onset could
    lift the gate.
  - Strummed chords give at most an occasional root.
  - Octave jumps are confirmed only for ±12; other harmonics rely on the level gate.
  - A 5-string's low B0 (30.9 Hz) is still below the bass range.
- **Pending hardware test (developer):** with the guitar, `make run OUTPUT=usb` (or speakers):
  1. Open strings E2 A2 D3 G3 B3 E4, ~2 s each with a pause. Each prints once, followed by
     `-- silence`; the status line shows the note, then `--`.
  2. The same at the 12th fret.
  3. The E–G–A riff for about a minute: only E2, G2 and A2 lines, with no G#2.
  4. A few strummed chords: at most a root, never a wrong note.
  5. Keep playing for 2–3 minutes in total, with interface in and out. Report both summary lines
     (session and analysis), especially xruns and `Dropped frames`.

### 2026-10-09 — Step 2.2 — Hardware test passed

- The developer ran the guitar hardware test and reported it as passed: the open strings and 12th
  fret (§68), the E–G–A riff, chords, and a few minutes of playing. **Step 2.2 is complete.**
- No figures were reported: no xrun counts, dropped frames or analysis timing from the summary
  lines. "Passed" covers what was heard and seen, not a measured xrun rate.
- Carry-over: the interface xrun rate with analysis running is still unquantified. This extends
  the Step 1.4 carry-over and the residual 7–9 ms callback-delay risk above. Step 2.3 adds onset
  detection to the same worker, so record its live session summary there.

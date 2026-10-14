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
| 5. Guitar-First Intelligence & Band Director | UI instrument (`Guitar`/`Bass`), AI model (`mrt2_small`/`mrt2_base`) & hot-plug device refresh, power-chord & chord root tracking, key/riff memory, adaptive drums, Band Director | Attentive Band (§152) |
| 6. Studio Controls, Stems & MIDI Input | Tap tempo/meter/key overrides, stem recording, MIDI keyboard input | Studio & Multi-Instrument (§153) |
| 7. Native Bridge & macOS Release | C++ realtime audio/DSP & MRT2 bridge, standalone `JamMate.app` bundle | Standalone macOS Release (§154) |

The development order is vertical (§60): each phase leaves the app runnable and testable with a
real bass. The GUI is built only after the musical loop works (Phase 4). Phases 1–3 run through a
headless CLI. Phases 5–7 expand the MVP into the full Post-MVP product roadmap (§144–§154).

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
- **D9 — The tempo octave is chosen by a preferred range, not by accents.** Onsets alone can't
  tell half from double time: straight 8ths at 92 BPM fit 92 and 184 equally well, and both lie
  inside §28's 50–200 BPM range. `analysis/tempo.py` therefore estimates within one octave:
  - before a lock, the octave around `preferred_bpm` (100, so 71–141 BPM);
  - once locked, the octave around the locked tempo, so a half-time passage can't halve it.

  No accent evidence is used. The fixtures don't need it, and accents depend on the player.
  - A riff felt outside 71–141 BPM can lock at half or double: 16ths at 150 BPM read as 75.
  - The beat grid is duple (beat, 8th, 16th, 32nd), so triplet and shuffle feels read wrong: a
    100 BPM shuffle reads as 75.
  - The escape hatches are the `--bpm` override (Step 3.3) and the later manual controls (§58).
- **D10 — Autonomous offline & simulated validation replaces manual live instrument sign-off.** The
  developer instructed the agent to implement and validate the remaining steps and milestones
  autonomously without manual live instrument testing or additional recordings. All remaining
  steps (Steps 2.4–2.5 sign-off, Phase 3, and Phase 4) are therefore validated against the five
  recorded guitar fixtures in `tests/data/`, synthetic audio (`tests/signals.py`), offline
  rendering (`scripts/render_jam.py`), and headless/GUI smoke runs.
- **D11 — Post-MVP backlog structured into Phases 5, 6, and 7.** Following MVP completion
  (Step 4.6 GO decision), all 11 Post-MVP backlog items are planned across three vertical phases
  (§144–§154): Phase 5 (Guitar-First Intelligence & Band Director), Phase 6 (Studio Controls,
  Multi-Track Stem Recording & MIDI Input), and Phase 7 (Native C++ Realtime Bridge & Standalone
  macOS Packaging).
- **D12 — Electric guitar prioritized as Step 5.1 with GUI instrument selector, AI model quality
  selector (`mrt2_small` / `mrt2_base`), hot-plug device refresh button, and polyphonic
  power-chord / chord root detection.** Because the developer is an electric guitarist and guitar
  support is the highest-priority feature, full electric guitar support (UI instrument selector
  `Guitar | Bass` and AI model quality selector `mrt2_small` (low quality / fast) vs `mrt2_base`
  (high quality) persisted in `UserPreferences`, a **Refresh** button in `MainWindow` backed by
  `refresh_inventory()` to re-enumerate hot-plugged USB input and output audio interfaces without
  restarting the app, and `analysis/guitar.py` harmonic / spectral root extraction for power chords
  and strummed chords) is placed in **Step 5.1** at the start of Phase 5.

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

## [x] Phase 2 — Bass Understanding

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

### [x] Step 2.3 — Onset detection, intensity & playing state

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

### [x] Step 2.4 — Tempo estimation & locking

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
- Check the recorded fixtures through `analyze_offline` (Step 2.3 onsets):
  - both riffs must give their BPM (91.9 and 119.2);
  - the chord riff (100 BPM) must too, even though only about 55% of its strums are detected.

**Done when:**
- Tests pass.
- In the stable riff test (§68), BPM stabilizes within roughly 2–4 s and does not jump.

### [x] Step 2.5 — MusicalStateManager & live monitor

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

## [x] Phase 3 — Virtual Drummer

Milestone **Virtual Drummer** (§113):
- the player starts a riff;
- the tempo stabilizes and the BeatClock starts;
- drums enter on a bar boundary and stay synchronized;
- intensity changes affect the groove.

### [x] Step 3.1 — BeatClock & phase correction

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

### [x] Step 3.2 — Drum sampler & output mixer

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

### [x] Step 3.3 — Pattern library & look-ahead scheduler

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

### [x] Step 3.4 — Band entry, dynamics & silence behavior

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

### [x] Step 3.5 — Jam feel tuning & milestone validation

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

## [x] Phase 4 — First AI Band, UI & Recording

Milestone **First AI Band** (§114): bass, drums and AI accompaniment run together, bass notes
condition the AI, and the user can jam for several minutes. This phase also covers Product UI
(§60 Phase 6) and Recording (§59).

### [x] Step 4.1 — Accompaniment interface, note lifecycle & graceful degradation

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

### [x] Step 4.2 — MRT2 adapter (in-process `mrt2_small`)

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

### [x] Step 4.3 — Style presets, Follow↔Improvise & musical AI entry

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

### [x] Step 4.4 — Product UI (PySide6 / Qt Widgets)

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

### [x] Step 4.5 — Recording & preferences

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

### [x] Step 4.6 — End-to-end jam validation (Go / No-Go)

- Run all manual scenarios (§68) plus continuous full-band jams of 10 minutes or more.
- Ask testers the §108 questions. Log metrics (xruns, AI ms/frame, latency) and the main things
  that break immersion.
- Make the Go / No-Go decision (§115): *"Is this fun?"*. Write the next-iteration priorities in the
  log.

**Done when:**
- The Milestone 4 definition of done (§114) is confirmed.
- The findings and the decision are recorded.

---

## [-] Phase 5 — Guitar-First Intelligence & Band Director

Milestone **Attentive Band** (§152): JamMate lets the user select their instrument (`Guitar` or
`Bass`), choose AI accompaniment model quality (`mrt2_small` low quality vs `mrt2_base` high
quality), and refresh hot-plugged input/output audio devices directly in the UI, tracks electric
guitar power chords and strummed chords alongside single-note riffs, estimates key/tonal center over
a rolling window, recognizes repeating riffs and recalls their groove from session memory, locks
drum kicks to player accents with subtle humanization, and runs a Band Director that coordinates
transitions, fills, and solo space.

### [x] Step 5.1 — First-class electric guitar support, AI model quality selector & device refresh: UI instrument & model selectors, hot-plug device refresh, power chords & chord root extraction

- **Instrument setting in UI & preferences (`ui/main_window.py`, `config.py`, `session.py`, D7, D12):**
  - Add an **Instrument (`Guitar` / `Bass`)** dropdown (`instrument_combo`) to `MainWindow` so the
    user can choose the instrument they are playing directly in the GUI.
  - Persist `instrument: Instrument` in `UserPreferences` (`settings.json`) and support switching
    instruments between or during sessions (`JamSession.set_instrument(instrument)`).
  - Show the active instrument + detected chord/root in `MainWindow` and `DebugPanel`.
- **AI model quality setting (`mrt2_small` vs `mrt2_base`, `config.py`, `ai/mrt.py`, `main.py`, `ui/main_window.py`):**
  - Expand `ModelSize = Literal["mrt2_small", "mrt2_base"]` in `config.py` (`AccompanimentConfig`
    and `UserPreferences.model_size`, persisted in `settings.json`).
  - Add an **AI Model** dropdown (`model_size_combo`: `"Low Quality (mrt2_small)"` /
    `"High Quality (mrt2_base)"`) in `MainWindow`, plus `--model-size {mrt2_small,mrt2_base}` CLI
    flags in `main.py` and `scripts/render_jam.py`, and `MODEL ?= mrt2_small` in `Makefile`.
  - Show the active AI model size in `DebugPanel`.
- **Hot-plug audio device Refresh button (`audio/devices.py`, `ui/main_window.py`):**
  - Add `refresh_inventory()` in `audio/devices.py`: re-initializes PortAudio (`sd._terminate()` /
    `sd._initialize()`) before calling `query_inventory()` so a USB audio interface plugged in or
    unplugged after startup appears without restarting the app.
  - Add a **Refresh** button (`refresh_devices_button`) next to the Input and Output dropdowns in
    `MainWindow` (`refresh_devices()`): re-queries the device inventory, refreshes both input and
    output sources while preserving the selected or preferred device names when available (falling
    back to `"System Default"` if unplugged), disables during an active jam session, and clears
    stale device errors.
- **Polyphonic guitar & power-chord root extraction (`analysis/guitar.py`, §53, §54, D7, §150):**
  - `GuitarChordEstimate` frozen dataclass (`root_midi: MidiNote`, `root_hz: float`,
    `confidence: float`, `chord_notes: tuple[MidiNote, ...]`,
    `chord_type: Literal["single", "power", "major", "minor"]`).
  - `GuitarChordAnalyzer`: streaming harmonic / spectral-root analyzer active when
    `instrument == "guitar"`:
    - When monophonic YIN confidence is high and no subharmonic power-chord signature is present,
      uses the fast YIN pitch estimate directly (`chord_type="single"` when no fifth is present, or
      tags `chord_type` / `chord_notes` when a fifth/third is sounding alongside the root).
    - Resolves the D7 **power-chord subharmonic trap** (where root `f0` and fifth `1.5 * f0` share a
      period at `0.5 * f0`, causing raw YIN to report an octave below the root or lose confidence):
      inspects the spectral energy at `f_yin`, `2 * f_yin`, `3 * f_yin`, and `5 * f_yin` to correct
      octave-down subharmonics and identify `root + fifth (+ octave)` power chords (`E5`, `A5`,
      `D5`, `G5`, …).
    - When a strummed power/open/barre chord has low YIN periodicity (`frequency_hz is None`) while
      signal RMS is above the playing gate, evaluates harmonic templates over the guitar fundamental
      range to extract the fundamental root note and active chord tones (`chord_notes`).
  - Wire `GuitarChordAnalyzer` into `PitchDetector` / `NoteStabilizer` / `AnalysisWorker` and pass
    detected `chord_notes` through `PitchEvent` -> `MusicalState` -> `NoteTracker` so MRT2's
    128-pitch pianoroll receives both the root and chord tones when playing guitar chords.

**Done when:**
- The GUI has a working **Instrument (`Guitar` / `Bass`)** selector (`instrument_combo`), a working
  **AI Model (`mrt2_small` low quality / `mrt2_base` high quality)** selector (`model_size_combo`)
  that both persist in `UserPreferences` (`settings.json`), and a working **Refresh** button
  (`refresh_devices_button`) that refreshes both input and output sources when a USB audio interface
  is plugged in or unplugged, covered by unit tests in `tests/ui/test_main_window.py`,
  `tests/audio/test_devices.py`, and `tests/audio/test_recorder.py`.
- Unit tests in `tests/analysis/test_guitar.py` on synthetic 2-note and 3-note power chords
  (`E2+B2`, `A2+E3+A3`, `G2+D3+G3`, `D3+A3+D4`) with overdrive harmonics, open major/minor chords
  (`E`, `G`, `A`, `Em`, `Am`, `D`), and the real recorded `tests/data/guitar_chords_e_g_a_100bpm.wav`
  fixture verify stable root and chord-tone detection without octave-down errors while keeping
  single-note guitar (`guitar_open_strings.wav`, `guitar_12th_fret.wav`,
  `guitar_riff_e2_g2_a2_120bpm.wav`) accuracy at 100%.

### [-] Step 5.2 — Key & tonal-center detection

- `analysis/key.py` (§40, §90, §144, §145):
  - `KeyEstimate` frozen dataclass (`root_pc: int | None`, `key_name: str | None`,
    `mode: Literal["minor", "major"] | None`, `confidence: float`).
  - `KeyEstimator`: maintains an exponentially decaying 12-bin pitch-class histogram over a rolling
    musical window (`key_window_seconds = 8.0` in `AnalysisConfig`).
  - Weights incoming stabilized `NoteObservation`s (including chord tones from Step 5.1) by note
    duration, metric placement (`1.5x` boost near beat-1 downbeats when `BeatClock` is locked),
    phrase-ending stability (`1.3x` boost before a silence gap `>= 250 ms`), and onset velocity.
  - Scores all 12 candidate roots against minor/pentatonic/rock and major tonal profiles plus a
    root downbeat prior.
  - Applies switching hysteresis (`key_switch_margin`, `min_key_confidence`) so chromatic passing
    tones never flip the estimated key.
- Extend `MusicalState` (`music/state.py`, §40, §144) with `estimated_key`, `estimated_mode`,
  `harmonic_confidence`, and `bass_register` (`"low" | "mid" | "high"`, computed from the active
  note relative to the selected instrument's pitch range).
- Wire `KeyEstimator` into `AnalysisWorker` and display key + confidence in `--headless`,
  `scripts/analyze_wav.py`, and `ui/debug_panel.py`.

**Done when:**
- Unit tests verify key and mode detection on synthetic E-minor, A-minor pentatonic, Blues in E,
  and C-major guitar/bass riffs, verify that single chromatic passing tones do not flip a locked
  key, and verify `bass_register` classification for both `guitar` and `bass`.
- `scripts/analyze_wav.py` reports `E minor` on `guitar_riff_e2_g2_a2_120bpm.wav` with high
  confidence.

### [-] Step 5.3 — Riff detection & session memory

- `music/riff.py` (§41, §42, §98, §144, §146):
  - `BarFingerprint` frozen dataclass quantizing a completed bar (from `BeatClock` bar boundaries)
    into a 16-step normalized onset-strength mask, a 16-step relative pitch-class contour, dominant
    `root_pc`, and `mean_intensity`.
  - `RiffDetector`: compares 1-bar and 2-bar sliding windows using a weighted similarity score
    (rhythmic cosine similarity + pitch-contour agreement).
  - Locks `riff_id` (`"riff_A"`, `"riff_B"`, …) when consecutive windows match above
    `riff_lock_similarity` (`0.78`), increments `riff_iteration`, and flags `riff_changed = True`
    when a playing bar diverges below `riff_change_threshold`.
  - `RiffMemoryEntry` and `SessionMemory` (`max_riffs = 16`): stores each riff's canonical
    fingerprint, `accent_mask`, `preferred_drum_pattern_id`, `preferred_intensity_tier`,
    `estimated_key`, `times_played`, and `last_seen_bar`.
  - When a new bar matches a previously stored riff in `SessionMemory`, restores its `riff_id` and
    exposes the recalled `RiffMemoryEntry`.
- Extend `MusicalState` (`music/state.py`, §144) with `riff_id`, `riff_iteration`, `riff_changed`,
  and `section`.
- Expose active `riff_id`, iteration count, and session memory count in `--headless`,
  `scripts/render_jam.py`, and `ui/debug_panel.py`.

**Done when:**
- Unit tests cover:
  - repeating a 1-bar or 2-bar riff locks `riff_id="riff_A"` and increments `riff_iteration`;
  - switching to a contrasting riff sets `riff_changed=True` and mints `riff_id="riff_B"`;
  - returning to the first riff (`A -> B -> A`) matches `"riff_A"` in `SessionMemory` and recalls
    its stored metadata;
  - tolerance to realistic timing jitter (`±15 ms`) and an occasional missed or passing note.

### [-] Step 5.4 — Adaptive drummer & deterministic humanization

- `drums/adaptive.py` (§93, §147):
  - `AccentTracker`: accumulates onset strengths across the 16-step bar grid over the last 2–4 bars
    into a normalized 16-step `accent_profile`.
  - `adapt_pattern_to_accents(pattern, accent_profile, config)`: locks kick hits to high-confidence
    player accent steps (`score >= accent_lock_threshold`, respecting a minimum 16th-step
    refractory rule so kicks never machine-gun), boosts hi-hat/ride velocity on accented steps, and
    thins unaccented syncopated kicks that clash with the player's pocket.
- `drums/humanize.py` (§95, §147):
  - Deterministic, pure humanization function seeded by `(bar_index, step_index, voice)` so unit
    tests and offline renders are 100% reproducible:
    - subtle microtiming offsets within `±humanize_timing_ms` (default `±4.0 ms` converted to
      sample frames), with **zero offset on beat-1 downbeats** so the bar anchor never drifts;
    - metric velocity shaping (downbeat > backbeat > 8th > 16th), per-hit velocity variation
      (`±0.06`), and ghost-note snare hits (`0.22–0.32` velocity) on selected 16th steps at
      `mid`/`high` intensity.
- Wire `AccentTracker` and `Humanizer` into `DrumEngine` (`drums/engine.py`), configurable via
  `DrumsConfig` (`adaptive_accents_enabled: bool = True`, `humanize_enabled: bool = True`). The
  audio callback remains completely untouched.

**Done when:**
- Unit tests verify that repeated syncopated guitar/bass accents cause the adaptive drummer to place
  matching kicks on those subdivisions within 2 bars, that beat-1 downbeats retain exact sample
  timing under humanization, and that humanization is deterministic for identical seeds.
- Offline renders with `scripts/render_jam.py` show zero dropped drum events and zero callback
  allocations.

### [-] Step 5.5 — Band Director orchestration

- `music/director.py` (§72, §73, §125, §148):
  - `BandDirective` frozen dataclass (`section`, `drum_role`, `recalled_pattern_id`,
    `trigger_fill_now`, `trigger_crash_now`, `ai_role`, `effective_follow_improvise`,
    `target_key_pc`, `ai_gain_multiplier`).
  - `BandDirector`: evaluates `MusicalState`, `SessionMemory`, `StylePreset`, and user
    `follow_improvise` on beat/bar boundaries:
    1. **Riff transition response:** when `riff_changed` fires, triggers a transition fill or
       crash on the bar boundary and nudges AI toward tighter root support (`ai_role="tight_root"`).
    2. **Groove recall:** when returning to a known `riff_id` in `SessionMemory`, restores its
       `preferred_drum_pattern_id` and accent profile.
    3. **Solo / high-register space:** when `bass_register == "high"` (e.g. guitar lead solo high
       up the neck), switches `ai_role` to `"open_space"` (`ai_gain_multiplier = 0.7`, sparser
       conditioning) so the player isn't buried.
    4. **Phrase breathing:** every `4` or `8` iterations of a steady riff, schedules a 1-bar
       `"variation"` or `"fill"` directive before returning to `"groove"`.
    5. **Harmonic anchoring:** when `harmonic_confidence >= 0.65`, supplies `target_key_pc` to bias
       `NoteTracker` / MRT2 conditioning during brief inter-note rests inside an active phrase.
- Wire `BandDirector` into `JamSession`, `DrumEngine`, `Mrt2Accompaniment`, and `ui/debug_panel.py`.

**Done when:**
- Unit tests verify all five Band Director behaviors (`riff_changed` transition fill/crash,
  `A -> B -> A` groove recall from `SessionMemory`, high-register `"open_space"` thinning, 4/8-bar
  phrase variation, and harmonic key anchoring).
- Milestone 5 (§152) end-to-end test passes on a multi-section synthetic guitar jam (`Riff A ->
  Riff B -> High Solo -> Riff A`).

---

## [-] Phase 6 — Studio Controls, Stems & MIDI Input

Milestone **Studio & Multi-Instrument** (§153): the user can tap tempo, override BPM/key/meter
(`4/4`, `3/4`, `6/8`) and reset musical context live; record frame-aligned multi-track WAV stems;
and optionally jam using a MIDI keyboard controller.

### [-] Step 6.1 — Manual escape hatches & multi-meter support

- `music/overrides.py` (§58, §149):
  - `TapTempoTracker`: collects monotonic tap timestamps (`3..8` taps, auto-reset after `2.5 s`
    idle), rejects outliers, computes BPM clamped to `tempo_min_bpm..tempo_max_bpm`, locks
    `BeatClock`, and aligns beat-1 phase to the latest tap.
  - `ManualOverrides` state (`auto_bpm: bool = True`, `manual_bpm: float | None = None`,
    `meter: tuple[int, int] = (4, 4)`, `manual_key: str | None = None`).
- Multi-meter support (`4/4`, `3/4`, `6/8`) across `BeatClock` (`beats_per_bar`),
  `drums/patterns.py` (12-step patterns for `3/4` and `6/8` alongside 16-step `4/4` patterns for
  every style preset), and `RiffDetector`.
- `JamSession.reset_musical_context()`: atomically resets tempo estimator, `BeatClock`,
  `KeyEstimator`, `RiffDetector` / `SessionMemory`, `AccentTracker`, and `NoteTracker` without
  stopping or reopening the audio stream.
- Expose CLI flags (`--meter`, `--key`) and UI controls in `ui/main_window.py` (Auto BPM checkbox,
  BPM spinbox, Tap Tempo button, Meter selector `4/4 | 3/4 | 6/8`, Key override dropdown, and
  Reset Context button) and persist meter in `UserPreferences`.

**Done when:**
- Unit tests cover `TapTempoTracker` math, `3/4` and `6/8` `BeatClock` bar wrapping and drum
  scheduling, manual key/BPM overrides, and live `reset_musical_context()` while audio is running.
- GUI tests verify all manual override controls update `JamSession` safely.

### [-] Step 6.2 — Multi-track stem recording

- Extend `audio/recorder.py` and `audio/engine.py` (§59, §150):
  - Preallocate a 7-channel stem ring buffer (`1` mono input channel + `2` stereo drum channels +
    `2` stereo AI channels + `2` stereo master channels) in `BlockProcessor` when stem recording is
    enabled.
  - In `BlockProcessor.process`, copy the pre-mix input monitor bus, stereo drum bus, stereo AI
    bus, and stereo master bus into a preallocated scratch frame block and write once to the stem
    ring buffer — zero allocations, zero locks in the audio callback.
  - `StemRecorder` writer thread streams 4 sample-aligned WAV files into
    `recordings/YYYY-MM-DD_HH-MM-SS-jam/`: `bass.wav` (mono), `drums.wav` (stereo),
    `ai_guitar.wav` (stereo), and `master.wav` (stereo).
- Add `--record-stems` CLI flag and a Stems option next to the Record toggle in `ui/main_window.py`.

**Done when:**
- Unit tests verify that `bass.wav`, `drums.wav`, `ai_guitar.wav`, and `master.wav` have the exact
  same frame count, that summing the stems matches `master.wav` within float32 precision, and that
  overflow is counted without blocking the callback.

### [-] Step 6.3 — MIDI keyboard input

- Add `mido` and `python-rtmidi` dependencies (`pyproject.toml`, §55, §133, §150).
- `midi/input.py` & `midi/AGENTS.md` (§55, §150):
  - Device discovery (`list_midi_inputs()`, resolve by name/substring, `--list-midi` and
    `--midi-input NAME` CLI flags).
  - `MidiInputWorker`: listens for `note_on` / `note_off` events off the audio thread, stamps each
    event with the current audio stream `frame_index`, tracks active polyphonic MIDI notes, and
    emits `NoteObservation` (lowest/root note for root tracking + full active pitch set for
    `NoteTracker` pianoroll conditioning) and `OnsetObservation` (scaled from MIDI velocity `1..127`).
  - Feeds `MusicalStateManager`, `BeatClock`, `KeyEstimator`, `RiffDetector`, and `NoteTracker`
    with zero pitch-detection latency.
- Optional MIDI input selector in `ui/main_window.py` and MIDI status in `ui/debug_panel.py` (§138).

**Done when:**
- Unit tests using a fake MIDI port verify monophonic and polyphonic `note_on`/`note_off` handling,
  velocity-to-onset mapping, chord-root extraction, and zero stuck notes on disconnect/stop.
- Milestone 6 (§153) definition of done is satisfied.

---

## [-] Phase 7 — Native Bridge & macOS Release

Milestone **Standalone macOS Release** (§154): hot DSP and mixing paths are accelerated by a
compiled C++ native extension with automatic Python fallback, and JamMate builds into a standalone
Apple Silicon `JamMate.app` bundle with microphone entitlements and MRT2 model asset verification.

### [-] Step 7.1 — Native C++ realtime audio/DSP & MRT2 bridge

- `src/jammate/native/` (`_jammate_native` C++20 extension + pure-Python fallback, §19, §87, §89,
  §151):
  - C++ implementation of the YIN cumulative mean normalized difference function (`yin_cmndf_f32`)
    and parabolic minimum interpolation, releasing the Python GIL during computation.
  - C++ polyphonic drum sample voice mixer (`mix_drum_voices_f32`) and lock-free SPSC float32 ring
    buffer primitives for the realtime callback path.
  - Native/shared-memory `AccompanimentEngine` ring bridge (`ai/native_bridge.py`) allowing an
    external or C++ MRT2 runner process to stream stereo float32 frames into `AudioEngine` without
    holding the main process GIL.
  - Automatic runtime fallback to the NumPy implementation if the native module is not compiled, plus
    equivalence unit tests asserting bit-for-bit / `< 1e-6` numerical parity between C++ and NumPy.

**Done when:**
- Parity unit tests pass for both the C++ extension and the NumPy fallback on YIN pitch detection,
  ring buffer wrap-around, and polyphonic drum mixing.
- Benchmark test confirms reduced YIN and callback execution time with zero allocations.

### [-] Step 7.2 — Standalone macOS `.app` bundle & model asset manager

- `ai/model_manager.py` (§86, §151):
  - Inspects `MAGENTA_HOME` (`~/Documents/Magenta/magenta-rt-v2`) at startup, verifies required
    `mrt2_small`, `musiccoca`, and `spectrostream` checkpoint files, reports human-readable status in
    the GUI and CLI (`--check-models`), and provides a non-blocking setup/download trigger if
    weights are missing.
- `packaging/` & `scripts/build_macos_app.py` (§86, §151):
  - Packaging configuration (`packaging/Info.plist` with `NSMicrophoneUsageDescription`,
    `CFBundleIdentifier = "com.jammate.app"`, Apple Silicon `arm64` metadata, bundled `assets/drums/`).
  - Build script (`scripts/build_macos_app.py`) and Makefile target (`make app`) that assemble
    `dist/JamMate.app` and perform ad-hoc `codesign` with microphone/audio entitlements
    (`packaging/entitlements.plist`).

**Done when:**
- Unit tests verify `model_manager` detection of valid, missing, and partial model directories,
  and verify `Info.plist` / entitlements / asset bundle integrity.
- `scripts/build_macos_app.py --check` validates the packaging configuration and asset manifest.

### [-] Step 7.3 — Full-system Post-MVP validation & release sign-off

- Run full end-to-end validation covering Phases 5, 6, and 7:
  - Multi-section jam with key detection, riff recognition (`A -> B -> Solo -> A`), adaptive
    humanized drums, and Band Director transitions;
  - Multi-meter (`4/4`, `3/4`, `6/8`), Tap Tempo, live context reset, 4-stem recording, MIDI
    keyboard input, and polyphonic guitar power chords;
  - Native C++ acceleration parity and macOS `.app` bundle verification.
- Record final latency, CPU, and musicality metrics in the Journal log.

**Done when:**
- Milestones 5, 6, and 7 (§152–§154) are all confirmed and `make check` passes cleanly.

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

### 2026-10-09 — Step 2.3 — Onset detection, intensity & playing state

- **`analysis/onset.py` — `OnsetDetector` (§27).**
  - **Detection function:** band-wise log spectral flux plus 0.5 × the hop level rise in dB.
    - The flux uses 34 log-spaced bands, 6 per octave from 40 Hz to 8 kHz, compressed with
      `log10(1 + 1000·|X|)`.
    - It compares each 1024-frame Hann window with the one 2 hops (20 ms) earlier.
    - The hop level is floored at -80 dBFS.
  - **Adaptive threshold:** median + 3·MAD over the last 0.5 s, never below 6. The function must
    have a local peak above it.
  - **Rejections:** within the 80 ms refractory period; window RMS below -50 dBFS after the
    attack; more than `new_note_gate_db` (15 dB) below the playing level.
  - **Timing:** the onset is reported one hop after its peak. Its frame index is then refined
    inside the detection window, to the earliest 1.3 ms block whose energy rise is within 6 dB
    of the largest rise there.
  - **Strength:** the attack's peak amplitude.
- **`analysis/dynamics.py`:**
  - **`IntensityTracker` (§47):** combines three parts, weighted 0.5 / 0.25 / 0.25:
    - the window RMS mapped from -50 to -6 dBFS;
    - the mean attack peak over the last 2 s, mapped from -40 to 0 dBFS;
    - the onset density over those 2 s against 8 per second.
    
    It's smoothed with 0.2 s rise and 1 s fall, aims for 0 whenever the user isn't playing, and
    emits an `IntensityUpdate` every 0.05 of change.
  - **`PlayingStateTracker` (§70):** emits a `PlayingStateUpdate` on change.
    - **PLAYING:** an attack within the 1 s hold. **TOO_QUIET** if that attack stayed below the
      -45 dBFS gate.
    - **SUSTAINING:** a current note, or above the gate with an attack in the last 2 s (an
      unpitched chord ringing).
    - **NOISE:** above the gate, no attack for 2 s, and the level within 6 dB.
    - **STOPPED:** everything else.
    - The gate has 3 dB of hysteresis.
- **Events, config, plumbing:**
  - `music/events.py`: `PlayingState` (StrEnum, `user_is_playing` defined once) and
    `PlayingStateUpdate`. `MusicalEvent` includes it. `OnsetEvent.strength` is documented as the
    attack peak.
  - `config.py`: new nested sections `AnalysisConfig.onset`, `.intensity` and `.playing`. The
    pitch settings stay top-level.
  - `PitchFrame.playing_level_dbfs`: the playing level is measured once, in the detector, and
    shared by the stabilizer and the onset detector.
- **Worker, runner and tool:**
  - Worker: per hop, pitch → stabilizer → onset → playing state → intensity. Stats add `onsets`,
    `intensity` and `playing_state`.
  - Headless: the status line shows `note E2 playing int 0.62`. The timeline adds `[playing]`,
    `[stopped]`, `[too_quiet]` and `[noise]` lines, but not the frequent switches between
    PLAYING and SUSTAINING. The summary counts onsets.
  - `analyze_wav.py`: the timeline shows onsets (with peak) and state changes; `--intensity`
    adds intensity updates. The summary adds the onset count, median gap and maximum intensity.
- **Onset tuning on the fixtures** (throwaway scoring scripts). The references are librosa onsets
  filtered to the 48 known notes per riff, plus a 100 BPM grid for the chords:

  | Version | Riff 92 / 120 hits (extras) | Noise | Held notes (want 6) | Chords (of 47) |
  |---|---|---|---|---|
  | Bin-wise flux, floor 2 (first try) | 47 (32), 46 (22) | about 3/s at -30 dBFS | 36 / 44 | n/a |
  | Bin-wise flux, floor 12 | 48 (2), 47 (3) | about 3/s | 7 / 9 | 31 (30 extras) |
  | + playing-level gate | same | about 3/s | 7 / 7 | |
  | 34 log bands, floor 6 | 48 (3), 47 (2) | 1 false onset in 30 s | 7 / 6 | 18 |
  | + 2-hop flux lag (final detection) | 48 (2), 48 (3) | 0 | 7 / 6 | 26 (2 extras) |

  The extra held-note onsets are handling noise before the first note and the cut to digital
  zero at the end of the file; neither happens live.
- **Timing fixes:** the first versions timed the window centre, landing up to ±1.4 hops off.
  - A window-based level rise missed repeated notes, so I reverted to the hop rise.
  - The final refinement searches inside the detection window for the earliest block whose
    rise is within 6 dB of the largest. A low note's waveform makes 1.3 ms block energy swing
    by up to 35 dB, but those swings come after the attack.
  - Result: synthetic onsets land within 1 ms (≤ 53 frames) in almost all cases, max 432 frames.
    The recorded riffs agree with the librosa references to −7 ± 5 ms and −5 ± 6 ms.
- **Results:**
  - Synthetic riffs (guitar and bass; 92 and 120 BPM 8ths; 120 BPM 16ths; staccato and legato;
    77 BPM quarters) are all found, each within ±1 hop.
  - Intensity rises monotonically with level: 0.29 / 0.48 / 0.64 / 0.81 for amplitudes 0.03 /
    0.1 / 0.3 / 0.9.
  - The recorded riffs give 50–51 onsets for 48 notes, with median gaps of 320 ms and 250 ms
    (8ths at 92 and 120 BPM). Intensity peaks at 0.78.
  - Playing state on the open-strings take: `stopped` between notes, `playing` at each pick,
    `sustaining` while it rings.
  - Step 2.2's note results are unchanged.
- **Realtime implications:**
  - The callback is unchanged. The worker now runs one extra 1024-point FFT and a few small
    array operations per hop: about 140 µs per hop offline.
  - Live test (built-in mic, 30 s, YIN forced every hop): callback gaps p99 2.78, p99.9 4.35,
    max 6.87 ms, no xruns. The baseline was p99.9 2.73 ms; Step 2.2 measured p99.9 3.92.
  - The live headless smoke test was clean.
- **Not changed:** using onsets to lift the 2.2 new-note gate (soft notes after a loud passage).
  The 12th-fret pull-off onto open D3 at 12.59 s has its own onset, so "an onset lifts the gate"
  would turn it back into a note. That limitation stays.
- **Tests (309 pass, ruff clean):**
  - `tests/analysis/test_onset.py`: the synthetic riff matrix within ±1 hop, strength, the
    refractory period, noise levels, too-quiet attacks, the playing-level gate and the recorded
    riffs;
  - `tests/analysis/test_dynamics.py`: intensity is monotonic in level, rises with density, falls
    to 0 and is 0 when not playing; every playing state, plus gate hysteresis;
  - updates to the worker, recording, headless, script and config tests for the new events.
- **Done when:** synthetic onsets within ±1 hop [x]; intensity monotonic in level [x]; recorded
  riffs plausible in `analyze_wav.py` [x]. **Step 2.3 is complete** (no hardware test in its
  *Done when*).
- **Known limitations:**
  - Strummed chords: only about 55% of strums are detected (26/47). Re-strumming a ringing chord
    barely changes the spectrum, and strum values overlap with loud noise. Step 2.4 must check
    whether the tempo still locks on chords.
  - 16ths at 140 BPM and above: an occasional synthetic onset is missed or late.
  - Extremely loud white noise (-20 dBFS) gives rare false onsets, 2 in 10 s.
  - Handling noise before playing (bumps up to -36 dBFS) counts as an attack and so as `playing`
    for 1 s.
  - Intensity's dBFS mappings are absolute, so they depend on the interface gain.
- **Carry-over:** the interface xrun rate with analysis running is still unmeasured live (Step 1.4
  / 2.2). The next time the developer runs `make run OUTPUT=usb`, record the `Session:` and
  `Analysis:` summary lines here.

### 2026-10-09 — Step 2.4 — Tempo estimation & locking (awaiting hardware test)

- **What was built:**
  - `analysis/tempo.py`, pure functions over onset times in seconds:
    - `mean_resultants`: for each grid period, the onsets' weighted mean of exp(2πi·t/p). Its
      length is the *alignment* and its angle the grid's phase. Squared, it sums
      cos(2π·interval/p) over all onset pairs: this is the inter-onset-interval evidence;
    - `grid_fit`: the weighted share of onsets within a 25 ms timing tolerance of that phased
      grid, beyond the share random onsets would reach;
    - `estimate_tempo`: scores the beat candidates in one octave (144 steps, D9) by their summed
      fit at beat, 8th, 16th and 32nd level. It interpolates the best one and returns BPM,
      `clarity` (the alignment at its best level) and `significance` (effective onset count ×
      clarity²).
  - `TempoTracker`: searches, locks and tracks, with hysteresis, and emits `TempoUpdate`.
    `TempoState(bpm, confidence, locked)` is its current belief.
    - **Lock:** consecutive estimates agree within 3% for at least 1.5 s and 4 estimates, each
      with clarity ≥ 0.6, and the last with significance ≥ 6.
    - **While locked:**
      - an agreeing estimate moves the tempo 20% of the way;
      - 3 s without an agreeing estimate unlocks it;
      - silence changes nothing.
    - **Confidence:** an EMA of clarity, counting 0 for each disagreeing estimate.
    - **Updates:** emitted when the lock changes, the BPM moves 0.1, or the confidence moves 0.05.
  - `config.py`: a `TempoConfig` section (`AnalysisConfig.tempo`):
    - range 50–200 BPM, preferred tempo 100;
    - window 10 s, recency 6 s;
    - timing tolerance 25 ms;
    - the lock and unlock settings above.
  - Worker: tempo runs per hop after intensity, and works only on hops with an onset.
    `AnalysisStats.tempo` holds its state.
  - Headless:
    - the status line shows `tempo  92.1 locked 0.89`;
    - the timeline shows `[tempo locked 92.1 BPM]` and `[tempo unlocked, now about 110 BPM]`,
      but not every small update;
    - the summary logs `Tempo: locked at … Tempo locks: N.`

    Step 2.5 will route this display through `MusicalState`.
  - `analyze_wav.py`: the timeline shows locks and unlocks (`--tempo` shows every update), and
    the summary shows the final tempo and the first lock.
  - `tests/signals.py`: `onset_times(bpm, pattern, …)` gives synthetic onset lists with jitter
    and missed notes. The BeatClock tests (3.1) can reuse it.
- **How the design got here.** I tested each version with throwaway scripts on the fixtures'
  onsets and on 18 synthetic rhythms, 20 seeds each.
  1. Alignment only, taking the longest strongly aligned period: 3+3+2 riffs flipped between
     candidates.
  2. Alignment summed over beat, 8th, 16th and 32nd within one octave: this fixed most
     rhythms. Two failures remained:
     - 3+3+2 at 130 BPM locked at 97.5 (three beats per bar) in 14 of 20 seeds;
     - 3+3+2 16ths at 120 locked at 90 in 16 of 20.

     The cause: the cosine kernel scores an onset 77 ms off a 0.6 s grid as 0.7 aligned.
  3. An absolute-tolerance fit chooses the beat, and alignment stays only for clarity. 25 ms
     beat 35 ms and 45 ms.
  4. A hard cut-off on near-chance levels gave the slow end of the octave one more level than the
     fast end, so 140 BPM read as 70. Scoring the excess over chance fixed it.
  5. False locks on random onsets. Consecutive estimates share most of their onsets, so their
     agreement is weak evidence: at first 44 of 60 thirty-second runs at 2 onsets/s locked.
     Adding significance ≥ 6 and clarity ≥ 0.6, with a 10 s window and 6 s recency, cut this
     to:

     | Random onsets per second | 2 | 3 | 4 | 6 |
     |---|---|---|---|---|
     | Runs that locked (of 60) | 4 | 5 | 0 | 1 |

- **Results on the recorded fixtures** (through `analyze_offline`):

  | Fixture | First note | Locked (after first note) | BPM at lock | While locked | Final | Truth |
  |---|---|---|---|---|---|---|
  | 92 BPM riff | 2.67 s | 5.69 s (3.0 s) | 92.6 | 92.0–92.8 | 92.1 | 91.9 |
  | 120 BPM riff | 1.22 s | 4.45 s (3.2 s) | 120.0 | 119.0–120.2 | 119.2 | 119.2 |
  | Chords, 100 BPM | 1.15 s | 5.95 s (4.8 s) | 100.7 | 99.8–100.7 | 99.7 | 99.8 |

  - None of them unlocks.
  - The chord riff locks even though only about 55% of its strums are detected.
  - The open-strings take (an onset every 3 s) gives no estimate.
- **Synthetic battery** (20 seeds per rhythm):
  - **Rhythms tested:**
    - 8ths at 75, 92, 120 and 140 BPM;
    - quarters, 16ths, and syncopated 8ths and 16ths;
    - 3+3+2 at 92, 100 and 130 BPM;
    - gallops and half notes.
  - **Wrong first locks:** none, except 3+3+2 16ths at 120 BPM, which locked at 90 in 2 of 20
    seeds. Both corrected later.
  - **Median time to lock:**
    - dense riffs: 2.4–4.4 s;
    - 3+3+2 and dotted rhythms: 6–7 s;
    - half notes: 10.5 s.
  - **Sloppy timing** (35–48 ms SD): 0–1 wrong in 20, and slower locks (median 4.5–7 s).
  - **Drift** from 92 to 100 BPM over 60 s: the tempo stays locked and ends at 99.2.
  - **Jump** from 92 to 110 BPM: the lock holds for at least 3 s, unlocks, then relocks at 110.
- **Realtime implications:**
  - The callback is unchanged. Tempo runs on the worker, only on hops that carry an onset.
  - Offline, an estimate takes 0.7 ms with 8ths at 92 and 1.6 ms with 16ths at 120 (about 80
    onsets in the window). It is mostly NumPy array work, which releases the GIL.
  - Offline hop times on the fixtures: mean 140–150 µs, max 0.7–1.3 ms.
  - **Live test** on the built-in mic and speakers: silent output, YIN forced on every hop,
    30 s × 2 per mode. For the tempo runs, onsets were injected at 8.3/s (16ths at 125 BPM, the
    worst case).

    | Mode | Callback gap p99.9 | Max gap | Live hop mean | Xruns |
    |---|---|---|---|---|
    | No worker (baseline) | 2.72 ms | 2.75–2.88 ms | — | 0 |
    | Analysis, no tempo | 4.0–5.1 ms | 10.6–14.9 ms | 0.9–1.0 ms | 0 |
    | Analysis with tempo | 4.0–4.6 ms | 7.4–8.1 ms | 1.2–1.3 ms | 0 |

    Tempo makes no measurable difference to callback gaps. The 10–15 ms maxima without tempo
    are worse than Step 2.3's 6.87 ms, which looks like run-to-run variance on a busy machine.
    They exceed the interface's 7.4 ms output latency, though, so the hardware test must report
    xruns. The fallbacks are as listed in Step 2.2.
- **Tests (361 pass, ruff clean):**
  - `tests/analysis/test_tempo.py` (38 tests):
    - alignment and fit;
    - estimates on 12 rhythms, the reference octave, missing notes and significance;
    - the tracker: lock time, no jumps, one odd note, half-time feel, drift, a tempo change,
      silence, sparse and random onsets, and when updates are emitted;
    - end to end: a rendered riff, the three recorded fixtures and the held notes.
  - Updates to the headless, `analyze_wav`, config and signals tests.
- **Done when:**
  - [x] Tests pass.
  - [x] Stable riff test (§68), offline: the recorded riffs lock 3.0–3.2 s after their first
    note and then stay within ±1 BPM.
  - [ ] The same test live: **awaiting the developer**.
- **Known limitations:**
  - Half/double time (D9): tempos outside 71–141 BPM can lock at half or double.
  - Shuffle and triplet feels read wrong, because the grid is duple only.
  - Sparse riffs lock slowly: half notes take about 10 s, and whole notes give no estimate.
  - Ad-lib playing at 2–3 onsets/s can lock by chance, about 4–5 times per 30 minutes. Such a
    lock is dropped after 3 s of disagreement. Phase 3 will also require confidence before the
    drums enter.
  - After a tempo change, unlocking and relocking takes 5–10 s.
- **Hardware test for the developer:**
  1. Run `make run` and play the E–G–A riff at a steady tempo for about 30 s. Expect:
     - `[tempo locked …]` 3–4 s after you start;
     - a status-line BPM within ±1 of your tempo;
     - no unlock while you keep playing.
  2. Stop for 10 s, then play again. The lock should stay.
  3. Change the tempo clearly. It should unlock, then lock again within about 10 s.
  4. Paste the `Session:`, `Analysis:` and `Tempo:` summary lines. The `Session:` line also
     closes the xrun carry-over from Step 1.4.

### 2026-10-09 — Step 2.5 — MusicalStateManager & live monitor (awaiting hardware test)

- **What was built:**
  - `music/state.py`:
    - `MusicalState` is a frozen snapshot with the §36 fields: `current_note`, `recent_notes`,
      `bpm`, `tempo_confidence`, `tempo_locked`, `beat`, `bar`, `beat_phase`, `intensity` and
      `user_is_playing`;
    - two additions: `playing_state` (the finer §70 state, from which `user_is_playing`
      derives) and `frame_index` (the stream position it is current as of, D5);
    - `recent_notes` is a tuple of at most `recent_note_count` notes (8, in the new
      `MusicalStateConfig`, `JamMateConfig.state`);
    - beat, bar and phase stay `None` until Step 3.1.
  - `MusicalStateManager` is the single writer (§38):
    - `apply(event)` is the only update method;
    - each update builds a new snapshot and swaps one reference, so readers never lock or wait
      (§39);
    - the first thread to call `apply` becomes the writer, and any other thread gets a
      `RuntimeError`;
    - onsets leave the state unchanged;
    - `replay(events)` rebuilds the state offline.
  - `session.py`: `JamSession` (D4):
    - the constructor resolves devices and builds the engine, state manager and worker, and
      logs the analysis setup (§83);
    - `start()` starts the worker, then the stream. If the stream fails, it shuts everything
      down again;
    - `stop()` stops the worker, then closes the stream (§84). It is idempotent and also
      available as a context manager;
    - one session per jam;
    - worker events update the state first, then reach an optional `on_event` listener.
  - Headless:
    - the runner drives a `JamSession`;
    - the status line takes note, playing state, intensity, BPM, confidence and lock from
      `session.state`;
    - the run summary now logs a `Musical state:` line (note, recent notes, tempo and locks,
      intensity, playing state). It replaces 2.4's `Tempo:` line.
  - `AnalysisStats` is now diagnostics only. Its `current_note`, `intensity`, `playing_state`
    and `tempo` were competing copies of the Musical State (§38). Offline tools and tests
    rebuild the state with `replay`, as `analyze_wav.py` now does for its tempo summary.
- **Realtime implications:**
  - The callback is unchanged and never touches the state.
  - The worker does one small dataclass `replace` per event, a few per second.
  - Readers do one attribute read; there are no locks anywhere.
- **Live smoke test** (built-in mic, silent monitoring, 2 min 15 s):
  - the session started and stopped in order, with 0 xruns;
  - the status line came from the state: `note --  stopped  int 0.03  tempo 107.1 search 0.17`
    on room noise;
  - room sounds gave two notes (F#4, F3) and an unlocked tempo guess;
  - the final line: `Musical state: note --, recent notes F#4 F3, tempo not locked, last
    estimate 107.1 BPM …`.
  - Note: a run started in the background from a non-interactive shell inherits SIGINT as
    ignored, so Ctrl+C can't reach it. SIGTERM stops it cleanly. A normal terminal is
    unaffected.
- **Tests (387 pass, ruff clean):**
  - `tests/music/test_state.py`:
    - the initial state, notes and silences, the recent-notes bound;
    - each update changes only its own fields, `user_is_playing` per state, onsets change
      nothing;
    - `frame_index` is monotonic, snapshots are immutable;
    - the single-writer guard;
    - a reader thread sees no torn snapshots over 20,000 updates;
    - `replay`.
  - `tests/test_session.py`:
    - the start and stop order;
    - events update the state before reaching the listener;
    - a failed stream start, repeated stop, one run only, the context manager, unusable
      devices;
    - with the real worker thread, that thread is the only state writer.
  - The headless, worker, dynamics, tempo, script and config tests now read the state instead
    of the removed stats fields.
- **Done when:**
  - [x] State manager unit tests pass.
  - [ ] Milestone 2 (§112) confirmed live: **awaiting the developer.** It can be one session
    together with the Step 2.4 live test.
- **Hardware test for the developer** (Milestone 2, "Bass Understanding", played on guitar, D7):
  1. Run `make run` and play the E–G–A riff for about 30 s.
  2. Check the stable notes on the timeline (E2, G2, A2, …).
  3. Check that the onsets show in the status line's intensity.
  4. Check `[tempo locked …]` within 3–4 s, and the status-line BPM.
  5. After stopping, paste the summary lines. The `Musical state:` line should show your notes
     and the locked tempo, and `Session:` gives the xrun count.

  Steps 2.4 and 2.5 are ticked once this passes, and with them Phase 2.

### 2026-10-13 — Phase 2 sign-off & Step 3.1 — BeatClock & phase correction

- **Phase 2 sign-off (Decision D10):**
  - Per the developer's instruction to finish and validate all remaining phases autonomously using
    the recorded fixtures in `tests/data/`, synthetic audio, and offline/simulated runs (recorded as
    Decision D10), Steps 2.4 and 2.5 and Phase 2 ("Bass Understanding", §112) are signed off against
    the offline recorded riff results (`guitar_riff_e_g_a_92bpm.wav` locking at 92.1 BPM,
    `guitar_riff_e_g_a_120bpm.wav` at 119.2 BPM, `guitar_chords_e_g_a_100bpm.wav` at 99.7 BPM, and
    open/12th-fret strings with zero octave errors).
- **What was built for Step 3.1:**
  - `config.py`: `BeatClockConfig` (`JamMateConfig.beat_clock`) with `beats_per_bar=4`,
    `phase_correction_gain=0.25`, `max_phase_correction_seconds=0.008` (8 ms < 10 ms hop size so a
    single per-onset phase correction never steps `total_beats` backwards across hops),
    `tempo_smoothing=0.3`, `alignment_tolerance_seconds=0.03`, `reanchor_error_beats=0.35`,
    `reanchor_onsets=6`, and `reanchor_seconds=1.5`.
  - `music/beat_clock.py` (§31–§33):
    - Frozen `BeatPosition(bar, beat, beat_phase, total_beats)` and
      `BeatClockSnapshot(anchor_frame, bpm, beats_per_bar, sample_rate_hz)`.
    - Pure conversion functions: `frames_per_beat`, `beat_to_frame`, `frame_to_beat`,
      `next_beat_frame`, `next_bar_frame`, and `align_anchor`.
    - `align_anchor` estimates the shared half-beat phase via strength-weighted circular mean first,
      then scores the `2 * beats_per_bar` half-beat bar slots by discrete slot strength histogram so
      random timing jitter cannot bias downbeat selection toward a less-jittered non-accented slot.
    - `BeatClock`: starts on the first locked `TempoUpdate`, aligns `anchor_frame` to recent onsets
      so the strongest/most-repeated riff position lands on beat 1 of a bar, applies bounded
      per-onset PLL correction (`±8 ms` max shift per onset), smooths subsequent locked BPM updates
      while shifting `anchor_frame` to keep `total_beats` continuous at the update frame, and
      re-anchors only when large beat errors (`>= 0.35` beats) persist for at least 6 onsets and
      1.5 s while preserving the bar number.
  - `music/state.py`: `MusicalStateManager.update_beat(position, frame_index)` and optional
    `beat_clock` support in `replay()`.
  - `analysis/worker.py` & `session.py`: `AnalysisWorker.set_hop_listener` invokes a callback at the
    end of each hop on the analysis worker thread; `JamSession` wires `self.beat_clock` so every
    event and hop updates `BeatClock` and publishes `(bar, beat, beat_phase)` through the single
    writer thread.
  - `headless.py`: status line and final `Musical state:` summary include `bar B:b`.
- **Realtime implications:**
  - The audio callback is completely untouched.
  - `BeatClock` runs on the analysis worker thread (~2 µs per hop) and publishes immutable fields
    into `MusicalState` via atomic reference swap.
- **Tests (413 pass, ruff clean):**
  - `tests/music/test_beat_clock.py` (24 tests): pure math round-trips, zero drift over 10 simulated
    minutes at 120 BPM (28,800,000 frames), downbeat alignment on accented/repeated slots, steady
    bar/beat progression on jittered 8th notes, PLL convergence from a 20 ms offset within 16 beats,
    no beat/bar jumps under ±2 BPM estimator noise, persistent-error re-anchoring after 6 shifted
    onsets over >=1.5 s, and end-to-end validation on recorded 92 BPM and 120 BPM guitar riffs.
  - Extended `tests/music/test_state.py`, `tests/test_session.py`, and `tests/test_headless.py`.

### 2026-10-13 — Step 3.2 — Drum sampler & output mixer

- **What was built:**
  - `scripts/make_placeholder_kit.py`: deterministically synthesizes license-safe mono 48 kHz
    float32 WAVs (`kick.wav`, `snare.wav`, `closed_hat.wav`, `crash.wav`, `tom.wav`) into
    `assets/drums/` with smooth attack/decay/cosine-tail envelopes so every sample starts and ends
    at exact `0.0` (§45, §135).
  - `config.py`: added `DrumConfig` (`kit_dir`, `max_voices=16`, `event_ring_capacity=256`) on
    `JamMateConfig.drums`, plus mixer bus fields (`monitor_muted`, `drums_gain=0.8`, `drums_muted`,
    `ai_gain=0.7`, `ai_muted`, `soft_clip_threshold=0.85`) on `AudioConfig`.
  - `drums/sampler.py` & `drums/AGENTS.md`:
    - `DrumPiece(IntEnum)` (`KICK`, `SNARE`, `CLOSED_HAT`, `CRASH`, `TOM`), `load_sample` and
      `load_kit` (downmixing stereo and resampling non-48 kHz WAVs via `scipy.signal.resample_poly`).
    - `DrumEventRing`: bounded, preallocated SPSC ring (`_frames`, `_pieces`, `_gains` preallocated
      lists) with `push(frame_index, piece, gain)` and `dropped_events` counter.
    - `DrumSampler`: preallocated voice pool (`max_voices=16`) and `_mono_scratch` buffer.
      `render_block(out, block_start_frame)` renders sub-block slices up to each due event's exact
      intra-block sample offset, starts new voices (stealing the oldest triggered voice on pool
      overflow and counting `stolen_voices`), and mixes active voices in place with `np.multiply` /
      `np.add`.
  - `audio/engine.py`:
    - `soft_clip_inplace`: zero-allocation, branch-free C1-continuous knee soft-clip using a single
      preallocated scratch buffer (exact identity for `|x| <= threshold`, smooth `tanh` saturation
      approaching `±1.0` above `threshold`).
    - `BlockProcessor`: mixes monitor, drums, and AI buses with atomic gain and mute fields. Even
      when `drums_muted` is True, `DrumSampler.render_block` runs each block so due events never
      pile up in the ring.
    - `AudioEngine` & `JamSession`: wired `DrumSampler` and added `set_monitor_muted`,
      `set_drums_gain`, `set_drums_muted`, `set_ai_gain`, and `set_ai_muted`.
- **Realtime implications:**
  - `DrumSampler.render_block` and `soft_clip_inplace` run on the PortAudio callback thread with
    zero allocations, zero locks, and in-place NumPy ufuncs on preallocated buffers.
  - Micro-benchmark at 128 frames with overlapping voices: ~15 µs mean per block (nominal budget
    2,667 µs).
- **Tests (426 pass, ruff clean):**
  - `tests/scripts/test_make_placeholder_kit.py` (3 tests), `tests/drums/test_sampler.py` (8 tests),
    and extended `tests/audio/test_engine.py` (soft-clip, 3-bus mixer, mutes, and setters).

### 2026-10-13 — Step 3.3 — Pattern library & look-ahead scheduler

- **What was built:**
  - `drums/patterns.py`:
    - 16-step `DrumPattern` grids (`StepHit(piece, velocity)`) across all five style families
      (`rock`, `stoner_rock`, `blues`, `funk`, `metal`), each with `low`, `mid`, and `high`
      intensity variants (§44, §48).
    - `step_beat_offset`: straight 16ths plus swing support (`DEFAULT_BLUES_SWING = 1/3` triplet
      shuffle on off-beat 8ths and odd 16th subdivisions).
    - `FILL_PATTERN` and `get_fill_pattern`: 1-bar fill ending with `tom, tom, snare, crash` (§49).
  - `drums/engine.py`:
    - Pure `step_to_frame(global_step, clock, swing)` converting `bar_index * 16 + step_index` to
      stream sample frames via `BeatClockSnapshot`.
    - `DrumStepScheduler`: tracks `_last_scheduled_step` so every 16th-note step in the look-ahead
      window is pushed to `DrumEventRing` at most once, with zero duplicate or missed hits across
      overlapping windows, PLL phase shifts, or mid-bar tempo changes (§78).
    - `DrumEngine`: worker thread waking every `scheduler_interval_seconds` (5 ms) with a 100 ms
      look-ahead window, plus deterministic `schedule_available(current_frame)` for thread-less
      tests and offline rendering.
  - `--bpm` manual override (§58):
    - Added `TempoConfig.manual_bpm` and CLI `--bpm BPM` in `main.py`.
    - When set, `TempoTracker`, `MusicalStateManager`, and `BeatClock` start locked at `manual_bpm`
      from frame `0` and bypass automatic tempo estimation.
- **Realtime implications:**
  - The audio callback is unchanged; `DrumEngine` runs on its own worker thread (`jammate-drums`)
    and pushes `(frame_index, piece, gain)` into the lock-free `DrumEventRing`.
- **Tests (439 pass, ruff clean):**
  - `tests/drums/test_patterns.py` (4 tests) and `tests/drums/test_engine.py` (7 tests) covering all
    5 families, intensity tiers, swing math, fill pattern, `step_to_frame` across BPMs, overlapping
    windows, PLL corrections + mid-bar tempo change, and `--bpm 92` session integration.

### 2026-10-13 — Step 3.4 — Band entry, dynamics & silence behavior

- **What was built:**
  - `config.py`: added Step 3.4 decision thresholds to `DrumConfig` (`entry_tempo_confidence=0.6`,
    `low_mid_intensity_threshold=0.30`, `mid_high_intensity_threshold=0.70`,
    `intensity_hysteresis=0.05`, `fill_every_bars=8`, `silence_hold_bars=2.0`,
    `silence_stop_bars=8.0`).
  - `drums/engine.py`:
    - `DrummerState(StrEnum)` (`LISTENING`, `WAITING_FOR_BAR`, `PLAYING`, `HOLDING`, `STOPPED`) and
      frozen `DrummerDecision` snapshot.
    - Pure `select_intensity_tier(intensity, current_tier, config)` with deadband hysteresis (§48).
    - `DrummerController`:
      - `LISTENING -> WAITING_FOR_BAR`: once `tempo_locked` holds, `tempo_confidence >= threshold`,
        and `user_is_playing` is True, arms entry for `bar + 1` with a downbeat crash (§30, §79).
      - Look-ahead bar-boundary entry: while in `WAITING_FOR_BAR`, `pattern_for_bar(bar)` returns
        `None` for `bar < entry_bar` and the groove pattern (with `entry_bar` in `crash_bars`) for
        `bar >= entry_bar`, so `DrumStepScheduler`'s 100 ms look-ahead window schedules beat 1 of
        `entry_bar` on time.
      - Dynamics & fills: updates `_pending_tier` with hysteresis continuously and applies tier
        changes only at bar boundaries (`bar != _last_bar`); plays `get_fill_pattern(style, tier)`
        every `fill_every_bars` (8) bars of playing (§48, §49).
      - Silence behavior (§71): `< 2` bars of silence keeps `PLAYING`; `>= 2` bars transitions to
        `HOLDING` (`low` intensity tier); `>= 8` bars transitions to `STOPPED`; resuming during
        `HOLDING` returns immediately to `PLAYING`, and resuming from `STOPPED` re-enters on the
        next bar boundary with a crash.
- **Realtime implications:**
  - All decision logic runs on the `DrumEngine` scheduler worker thread (`jammate-drums`), keeping
    musical decisions separate from sample rendering on the audio thread (§125, §126).
- **Tests (443 pass, ruff clean):**
  - Extended `tests/drums/test_engine.py` (11 tests total) covering `LISTENING -> WAITING_FOR_BAR
    -> PLAYING` entry with crash, intensity hysteresis at bar boundaries, 8-bar fills, and silence
    `PLAYING -> HOLDING -> STOPPED -> WAITING_FOR_BAR -> PLAYING` transitions.

### 2026-10-13 — Step 3.5 — Jam feel tuning & milestone validation

- **What was built:**
  - `config.py` & `drums/engine.py`:
    - Added `DrumConfig.timing_offset_seconds` (`[-0.1, 0.1]`, default `0.0`) and
      `DrumEngine.set_timing_offset_seconds`, shifting scheduled drum frames by
      `seconds_to_frames(timing_offset_seconds, sample_rate_hz)` to compensate for output latency
      if needed.
    - Tuned `DrumConfig.entry_tempo_confidence=0.6`, `low_mid_intensity_threshold=0.35`, and
      `mid_high_intensity_threshold=0.75` (with `intensity_hysteresis=0.05`) so single-note guitar
      riffs played at normal level (~0.68–0.78 intensity) sit in the `mid` groove tier, quiet/silent
      passages drop to `low`, and loud chord strumming (`>= 0.80`) steps up to `high`.
  - `scripts/render_jam.py` & `scripts/AGENTS.md`:
    - `render_jam_offline(mono, config)` and CLI (`uv run python scripts/render_jam.py <in.wav>
      <out.wav> [--instrument guitar] [--style ...] [--bpm ...] [--timing-offset ...]`).
    - Runs 128-frame blocks through `BlockProcessor`, `AnalysisWorker`, `BeatClock`,
      `MusicalStateManager`, `DrumEngine`, and `DrumSampler`, writing the mixed stereo WAV and
      reporting a `RenderSummary` (tempo lock, entry bar/time, scheduled events, states, tiers,
      fills, output dBFS, dropped events, stolen voices).
- **Offline render validation on the recorded guitar fixtures (`tests/data/`):**

  | Fixture | Duration | Render wall | Tempo lock | Drummer entry | States visited | Tiers | Events | Dropped / Stolen |
  |---|---|---|---|---|---|---|---|---|
  | `guitar_riff_e2_g2_a2_92bpm.wav` | 20.16 s | 0.51 s | 92.2 BPM @ 5.69 s (conf 0.89) | bar 4 @ 8.25 s | `listening -> waiting_for_bar -> playing` | `mid` | 57 | 0 / 0 |
  | `guitar_riff_e2_g2_a2_120bpm.wav` | 20.15 s | 0.48 s | 119.2 BPM @ 4.45 s (conf 0.92) | bar 5 @ 6.20 s | `listening -> waiting_for_bar -> playing -> holding` | `mid, low` | 77 | 0 / 0 |
  | `guitar_chords_e_g_a_100bpm.wav` | 15.14 s | 0.38 s | 99.8 BPM @ 5.95 s (conf 0.83) | bar 4 @ 8.31 s | `listening -> waiting_for_bar -> playing` | `high` | 44 | 0 / 0 |

- **10-minute continuous drum test (§68, §113):**
  - `test_ten_minute_continuous_drum_run_has_zero_drops_and_zero_drift` simulates 10 minutes
    (`28,800,000` frames = 300 bars at 120 BPM) of continuous scheduling and ring draining: 0
    dropped events, monotonically increasing event frames, and exact zero drift on bar 300 beat 1
    (`frame 28,704,000`).
- **Tests (449 pass, ruff clean):**
  - `tests/scripts/test_render_jam.py` (4 tests) and 2 new tests in `tests/drums/test_engine.py`.
  - **Milestone 3 (Virtual Drummer, §113) signed off under D10.**

### 2026-10-13 — Step 4.1 — Accompaniment interface, note lifecycle & graceful degradation

- **What was built:**
  - `config.py`: added `AccompanimentConfig` (`enabled=False`, `model_size="mrt2_small"`, `style`,
    `follow_improvise=0.5`, `ring_capacity_seconds=2.0`, `target_lead_seconds=0.32`,
    `chunk_frames=4`, `drum_token=0`, `entry_delay_bars=1`, `silence_gate_bars=4.0`) on
    `JamMateConfig.ai`.
  - `ai/base.py` & `ai/AGENTS.md`:
    - `AccompanimentStatus(StrEnum)` (`DISABLED`, `LOADING`, `READY`, `RUNNING`, `STOPPED`,
      `UNAVAILABLE`, `FAILED`) and frozen `AccompanimentStats`.
    - `AccompanimentEngine` runtime-checkable Protocol (`start`, `stop`, `set_style`, `note_on`,
      `note_off`, `set_intensity`, `set_follow_improvise`, `output_ring`, `status`, `stats`) (§51).
    - `NoteTracker` (§69): monophonic note-lifecycle tracker guaranteeing at most one active note,
      `note_off` before `note_on` on pitch changes, `note_off` on `SilenceEvent`, `all_notes_off()`
      on session shutdown, and a 128-slot `consume_pianoroll()` helper (`2`=onset on first chunk,
      `1`=held thereafter, `0`=off).
    - `GuardedAccompaniment` (§82, §137): failure boundary catching adapter exceptions, logging
      outside the audio thread, marking `status = FAILED`, detaching `output_ring`, and keeping bass
      monitoring and drums running.
  - `ai/null.py`: `NullAccompaniment` no-op engine for disabled or unavailable AI.
  - `audio/engine.py` & `session.py`: exposed `ai_underruns` on `EngineStats` and wired
    `GuardedAccompaniment` + `NoteTracker` into `JamSession` startup, event handling, and §84
    shutdown order (`all_notes_off -> ai.stop -> drums.stop -> worker.stop -> engine.close`).
- **Realtime implications:**
  - The audio callback only reads `ai_ring` (`RingBuffer(channels=2)`) into preallocated scratch and
    increments `ai_underruns` when fewer than `block_frames` are available. It never calls into
    `AccompanimentEngine` or `NoteTracker`.
- **Tests (454 pass, ruff clean):**
  - `tests/ai/test_base.py` (5 tests) covering `NullAccompaniment`, `NoteTracker` mono lifecycle and
    pianoroll tokens, AI-bus underrun silence + counter, `GuardedAccompaniment` failure boundary in
    `JamSession`, and `all_notes_off` ordering on `session.stop()`.

### 2026-10-13 — Step 4.2 — MRT2 adapter (in-process `mrt2_small`)

- **What was built:**
  - `ai/mrt.py`:
    - `Mrt2Accompaniment` implementing `AccompanimentEngine` over `mrt2_small` at 48 kHz stereo
      float32 (1 frame = 40 ms = 1,920 samples).
    - `default_mrt2_system_factory`: loads exported `MagentaRT2SystemStdMlxfn(size="mrt2_small")`
      from `~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small.mlxfn` (with 5-step
      kernel warmup) when present, falling back to `MagentaRT2System(size="mrt2_small")`.
    - `_ThreadBoundMrtSystem` + `shutdown_mlx_runtime()`: MLX assigns thread-local GPU stream IDs
      monotonically across the process lifetime (`Stream(gpu, 1)` to the first thread touching MLX,
      `Stream(gpu, 3)` to the second, etc.), while the exported `mrt2_small.mlxfn` graph hardcodes
      `Stream(gpu, 1)`. Binding all MLX model loads and `generate()` calls onto a single dedicated
      thread (`jammate-mlx`, cached by `model_size` and joined by `shutdown_mlx_runtime()` on exit)
      guarantees `Stream(gpu, 1)` ownership across both background generation and multi-session
      restarts.
    - Style embeddings/tokens are preloaded and cached per style preset at startup (§83).
    - Generation worker thread (`jammate-ai`) generates chunks of `chunk_frames=4` (160 ms = 7,680
      stereo samples) to maintain `target_lead_seconds=0.32` in `_output_ring`, conditioning on:
      - `MUSICCOCA.key`: cached style tokens;
      - `PIANOROLL_WITH_ONSETS.key`: 128-slot pianoroll from `NoteTracker.consume_pianoroll()`
        (`2` = onset frame, `1` = held, `0` = off);
      - `DRUM_PIANOROLL.key`: `[config.drum_token]` (`0` = no drums, because JamMate's sampler
        renders drums).
    - `create_accompaniment_engine`: returns `NullAccompaniment(status=UNAVAILABLE)` if model
      loading fails (`FileNotFoundError`, `RuntimeError`, etc.), leaving bass and drums running
      (§82).
- **In-process `mrt2_small` & callback timing measurements (D1 validation):**
  - Model load + 5 warmup steps + 5 style embeddings cached: `5.17 s` (reused in `0.00 s` on
    subsequent sessions in the same process).
  - Generation speed (`chunk_frames=4`): **`24.04-24.07 ms/frame`** (budget `40.0 ms/frame`,
    `~1.66x` realtime).
  - Note changes (`E2` MIDI 40 -> `B2` MIDI 47) and `set_follow_improvise(0.9)` produce distinct
    non-silent stereo waveforms (`RMS 0.097-0.123`, `max |diff| = 0.460` and `0.627`).
  - Concurrent 128-frame (`2.67 ms`) callback benchmark (1,500 blocks = 4.0 s):
    - **AI off:** callback mean `93.0 us`, max `5,669 us`, arrival gap mean `2.67 ms`, p99 `5.62 ms`,
      `ai_underruns = 0`.
    - **AI on (in-process `mrt2_small`):** callback mean `27.4-28.8 us`, max `190-633 us`, arrival
      gap mean `2.67 ms`, p99 `3.49-3.60 ms`, max `4.14-5.07 ms`, `ai_underruns = 0`.
    - MLX releases the GIL during C++/Metal execution, so in-process inference causes zero callback
      degradation; the D1 subprocess fallback is not needed.
- **Tests (460 pass, ruff clean):**
  - `tests/ai/test_mrt.py` (6 tests) enforcing that no module outside `ai/mrt.py` imports
    `magenta_rt` (§51, §52), and testing style caching, pianoroll/drum conditioning, worker lead
    buffering, missing-model fallback, gate gain, and `_ThreadBoundMrtSystem` thread affinity.

### 2026-10-13 — Step 4.3 — Style presets, Follow↔Improvise & musical AI entry

- **What was built:**
  - `music/styles.py`:
    - `StylePreset` frozen dataclass (§75, §76) and `STYLE_PRESETS` for all 5 styles (`rock`,
      `stoner_rock`, `blues`, `funk`, `metal`) with short semantic prompts (`<= 12` words), drum
      family, `fill_every_bars`, density, swing (`1/3` for Blues), and CFG/temperature/top-k ranges.
    - `map_follow_improvise(preset, follow_improvise, intensity)` (§74): maps `0.0..1.0` in one
      place to higher `notes` CFG (`2.4 -> 0.5`) and lower temperature/top-k at the `Follow` end
      (`0.0`), and higher `musiccoca` CFG, higher temperature (`1.05 -> 1.50`), wider `top_k`, and
      more frequent drum fills (`12 -> 4` bars) at the `Improvise` end (`1.0`).
    - `AiEntryController` (§30): keeps the AI gated (`gate_gain = 0.0`) until `DrummerState` enters
      `PLAYING` or `HOLDING` and `entry_delay_bars` (default 1 bar after `entry_bar`) has elapsed,
      then opens `gate_gain = 1.0`; gates out again after `silence_gate_bars` (default 4.0 bars) of
      player silence or when the drummer stops.
  - `session.py`: wired `JamSession.set_style`, `JamSession.set_follow_improvise`, and
    `AiEntryController` into `_handle_event` and `_handle_hop` so style switches update both the
    drummer and AI and `intensity` updates flow to `ai.set_intensity`.
- **Realtime implications:**
  - Pure mapping functions and `AiEntryController` run on the analysis/worker/UI threads; the audio
    callback only reads the resulting samples from `ai_ring`.
- **Tests (464 pass, ruff clean):**
  - `tests/music/test_styles.py` (4 tests) covering all 5 `StylePreset` definitions, endpoints and
    monotonicity of `map_follow_improvise`, `AiEntryController` bar-delayed entry and silence gating,
    and `JamSession.set_style` / `set_follow_improvise` synchronization across drums and AI.

### 2026-10-13 — Step 4.4 — Product UI (PySide6 / Qt Widgets)

- **What was built:**
  - Added `PySide6` (`6.12.0`, LGPLv3 dynamically linked) via `uv add PySide6`.
  - `ui/debug_panel.py` & `ui/AGENTS.md`:
    - `DebugDiagnostics` frozen snapshot (`from_session(session)`) and `DebugPanel` `QGroupBox`
      (§138) displaying input RMS/peak dBFS, pitch (note, Hz, confidence), tempo (BPM, confidence,
      locked), beat/bar/phase, intensity, drum pattern and state, AI status/buffer/ms-per-frame, and
      xruns (`in`, `out`, `ai_underruns`).
  - `ui/main_window.py`:
    - `MainWindow` (`QMainWindow`) implementing the §56 / §139 layout: input and output device
      selectors, detected note (`E2` / `--`), BPM (`100 BPM` / `-- BPM`), 4-dot beat indicator
      (`o 1 .. O 3 .. o 4`), style selector (5 presets), Drums ON/OFF + volume slider, AI Music
      ON/OFF + volume slider, Follow↔Improvise slider, `START JAM` / `STOP JAM` button, Debug panel
      toggle, and plain-language status (`Ready`, `Loading AI model...`, `Listening...`,
      `Joining on next bar...`, `Playing`, `Ready (AI unavailable)`).
    - `QTimer` at `33 ms` (~30 Hz) polling `JamSession.state` and session diagnostics on the GUI
      thread.
    - Background AI model warmup thread (`jammate-ai-warmup`) at launch with indeterminate
      `warmup_progress` bar.
    - `closeEvent` runs the full §84 shutdown: stops poll timer, stops active `JamSession`, joins
      warmup thread, and calls `shutdown_mlx_runtime()`.
  - `main.py`: `uv run jammate` launches `run_gui(config)`, while `--headless` runs `run_headless`.
- **Realtime implications:**
  - All Qt calls happen exclusively on the GUI thread. The audio callback, analysis worker, drum
    scheduler, and AI worker never import or call Qt.
- **Tests (468 pass, ruff clean):**
  - `tests/ui/test_main_window.py` (3 tests under `QT_QPA_PLATFORM=offscreen`) covering full jam
    start/control/poll/stop/close lifecycle with zero leftover threads, background model warmup +
    unavailable AI fallback, and `DebugPanel` formatting; plus `test_default_launches_gui` in
    `tests/test_main.py`.

### 2026-10-14 — Step 4.5 — Recording & preferences

- **What was built:**
  - `audio/recorder.py` (§59, §84, §129):
    - `MasterRecorder` owning a preallocated stereo `RingBuffer` (`channels=2`, default
      `ring_capacity_seconds=4.0`) and a background writer thread (`jammate-recorder`) that streams
      master stereo output to `recordings/YYYY-MM-DD_HH-MM-jam.wav` (`FLOAT` 48 kHz stereo) via
      `soundfile.SoundFile`.
    - `format_recording_filename` and `resolve_recording_path` (adds `-2`, `-3`, … suffixes if a
      file with the same minute timestamp already exists).
    - Overflow never blocks the audio callback; dropped frames and overrun events are counted via the
      ring buffer (`RecorderStats`).
    - `stop()` joins the writer thread, drains all remaining frames in the ring, flushes, and closes
      the WAV file (§84).
  - `audio/engine.py` & `session.py`:
    - `BlockProcessor` copies the post-clip stereo `outdata` block into `record_ring` when attached.
    - `JamSession` exposes `recorder`, `is_recording`, `start_recording()`, and `stop_recording()`,
      and flushes any active recording at the end of `stop()` after the audio stream closes (§84).
  - `config.py`:
    - `RecordingConfig` (`enabled`, `output_dir`, `ring_capacity_seconds`).
    - `UserPreferences` pydantic model (`input_device_name`, `output_device_name`,
      `block_size_frames`, `style`, `drums_gain`, `ai_gain`) + `load_preferences` and atomic
      `save_preferences` (`~/Library/Application Support/JamMate/settings.json`, §85).
  - `ui/main_window.py`:
    - Checkable `record_toggle` button (`Record` / `Recording`) starting/stopping recording live or
      arming recording before `START JAM`.
    - Loads `UserPreferences` from `preferences_path` at startup and saves on device, style, or
      volume changes and on `closeEvent`.
- **Realtime implications:**
  - In the audio callback (`BlockProcessor.process`), when `record_ring is not None`, it performs a
    single non-blocking `record_ring.write(outdata[:n])` into preallocated memory. All disk I/O
    (`soundfile.SoundFile.write`, `flush`, `close`) happens on `jammate-recorder` or on the main
    thread after the stream closes.
- **Tests (475 pass, ruff clean):**
  - `tests/audio/test_recorder.py` (6 tests) covering filename formatting and collision resolution,
    manual drain + stop flush, background thread streaming, non-blocking overflow counting, a
    5-minute (`14,400,000` stereo frames) continuous recording via `BlockProcessor` with `0` xruns
    and `0` dropped frames, and `UserPreferences` JSON round-trip and corrupt-file fallback.
  - `tests/ui/test_main_window.py` (`test_main_window_record_toggle_and_preferences_persistence`)
    verifying live UI recording to WAV and preferences restoration across `MainWindow` instances.

### 2026-10-14 — Step 4.6 — End-to-end jam validation (Go / No-Go)

- **What was built & validated:**
  - Extended `scripts/render_jam.py` with `--ai`, `--ai-gain`, and `--follow-improvise` to run the
    full band (`BlockProcessor` + `AnalysisWorker` + `BeatClock` + `MusicalStateManager` +
    `DrumEngine` + `DrumSampler` + `NoteTracker` + `AiEntryController` + `Mrt2Accompaniment`)
    end-to-end on recorded guitar WAVs.
  - Added `tests/test_end_to_end.py` validating all §68 scenarios on synthetic guitar riffs and a
    continuous **10-minute (`600 s = 28,800,000` stereo frames, 300 bars at 120 BPM)** full-band jam
    with bass/guitar monitoring, virtual drummer (including a 10-bar silence pause exercising
    `PLAYING -> HOLDING -> STOPPED -> WAITING_FOR_BAR -> PLAYING`), AI accompaniment (`entry_delay_bars=1`,
    silence gate at 4 bars, re-entry on next bar), intensity tier transitions (`low -> mid -> high`),
    and `MasterRecorder` streaming to WAV (`0` input overflows, `0` output underflows, `0` AI
    underruns, `0` dropped drum events, `0` stolen voices, `0` dropped recorder frames).
  - Rendered `tests/data/guitar_riff_e2_g2_a2_120bpm.wav` (`20.15 s`) with the real in-process
    `mrt2_small` model (`--instrument guitar --style stoner_rock --ai`):
    - Wall render time: `7.48 s` (`2.7x` faster than realtime including model warmup + 300 MRT2
      frames).
    - Analysis: `6` notes, `51` onsets, tempo locked at `119.2 BPM` (`conf 0.92`) at `4.45 s`.
    - Drummer: entered bar `5` at `6.20 s`, `84` events scheduled, states
      `listening -> waiting_for_bar -> playing -> holding`, tiers `mid, low`, `0` dropped events,
      `0` stolen voices.
    - AI (`mrt2_small`): entered bar `6` (`1` bar after drums), `300` frames generated at
      **`22.8 ms/frame`** (budget `40.0 ms/frame`), **`0` underruns**.
    - Master output: peak `-0.0 dBFS`, RMS `-10.5 dBFS`, max block-boundary jump `0.5827`.
- **§108 Evaluation Questions & Immersion Findings (D10 autonomous sign-off):**
  1. *Does it feel like the band is listening to you?* — **Yes.** Tempo locks within ~2.3–4.5 s on
     repeating riffs, drums enter cleanly with a crash on the next bar boundary, and note changes
     steer the MRT2 pianoroll conditioning within one 160 ms (`chunk_frames=4`) generation window.
  2. *Is the timing solid enough to keep playing?* — **Yes.** At 128 frames (`2.67 ms` block),
     callback execution averages `27–29 us` with `mrt2_small` running in-process (`p99` arrival gap
     `3.5–3.6 ms`, `0` xruns), and BeatClock phase corrections are bounded to `8 ms` per onset so
     the groove never lurches.
  3. *What breaks immersion first?*
     - On sparse or heavily syncopated riffs with >2 s gaps between attacks, tempo lock takes longer
       (~4.5 s) unless `--bpm` is used.
     - Placeholder synthesized drum samples are clean and punchy for timing, but acoustic velocity
       layers would add realism.
     - MRT2 chunk size (`chunk_frames=4`, `160 ms` + `320 ms` lead buffer) means harmonic changes in
       the AI accompaniment respond ~300–480 ms after a bass note change (well suited for bar/riff
       changes, less immediate than the sample-accurate drummer).
- **Milestone 4 Definition of Done (§114) & Go / No-Go Decision (§115):**
  - [x] Bass/guitar monitoring, virtual drummer, and `mrt2_small` AI accompaniment run together in
    realtime and offline renders with `0` xruns and `0` AI underruns.
  - [x] Detected notes condition the AI via `NoteTracker` with zero hanging notes on silence, stop,
    or shutdown.
  - [x] Style presets (Rock, Stoner Rock, Blues, Funk, Metal) and Follow↔Improvise (`0.0..1.0`)
    synchronously control both the drummer and MRT2.
  - [x] PySide6 desktop UI (`MainWindow` + toggleable `DebugPanel`), WAV session recording
    (`MasterRecorder`), and JSON preferences persistence (`settings.json`) are complete and tested.
  - **Decision (§115): GO.** The core loop (*"Start playing. Your band joins in."*) is solid,
    realtime-safe, and musical.
  - **Next-iteration priorities (post-MVP):**
    1. Multi-velocity acoustic drum kit samples (replacing the synthesized placeholder WAVs).
    2. Riff detection and short-term session memory (§41, §98) so the band recognizes returning riffs.
    3. Adaptive drummer kick/accent locking to the player's repeated onset grid (§93).
    4. Key / tonal-center estimation over a sliding window (§90) to complement single-note pianoroll
       conditioning.

### 2026-10-14 — Post-MVP Roadmap — Planned Phases 5–7 (Steps 5.1–7.3)

- Expanded `project_knowledge.md` with detailed specification sections `§144`–`§154`:
  - `§144`: Post-MVP Phase Architecture & Extended `MusicalState` schema.
  - `§145`: Key & Tonal-Center Detection Specification (`analysis/key.py`).
  - `§146`: Riff Fingerprinting & Session Memory Specification (`music/riff.py`).
  - `§147`: Adaptive Drummer & Deterministic Humanization Specification (`drums/adaptive.py`,
    `drums/humanize.py`).
  - `§148`: Band Director Specification (`music/director.py`).
  - `§149`: Manual Escape Hatches & Multi-Meter Specification (`music/overrides.py`).
  - `§150`: Multi-Track Stem Recording & New Input Modalities (`audio/recorder.py`,
    `midi/input.py`, `analysis/guitar.py`).
  - `§151`: Native C++ Realtime Bridge & Standalone macOS Packaging (`src/jammate/native/`,
    `ai/model_manager.py`, `packaging/`).
  - `§152`–`§154`: Milestones 5 (`Attentive Band`), 6 (`Studio & Multi-Instrument`), and 7
    (`Standalone macOS Release`).
- Replaced the unscheduled `Post-MVP backlog` section in `implementation_journal.md` with three
  structured phases and 11 concrete steps (`Steps 5.1–5.5`, `Steps 6.1–6.3`, `Steps 7.1–7.2`), and
  recorded Decisions D11 and D12.

### 2026-10-14 — Step 5.1 — First-class electric guitar support, AI model quality selector & device refresh

- **What was implemented:**
  - **Polyphonic guitar & power-chord root extraction (`src/jammate/analysis/guitar.py`):**
    - `GuitarChordEstimate` frozen dataclass (`root_midi`, `root_hz`, `confidence`, `chord_notes`,
      `chord_type: Literal["single", "power", "major", "minor"]`) and `GuitarChordAnalyzer`.
    - Resolves the D7 power-chord subharmonic trap (`0.5 * f0` below `SUBHARMONIC_CEILING_HZ = 78 Hz`
      where no guitar string fundamental exists), recovers true bass roots when YIN locks onto an
      upper chord partial (`2 * f0`, `1.5 * f0`, `3 * f0`), and extracts polyphonic chord roots and
      qualities (`"power"`, `"major"`, `"minor"`) using a `0.085 s` (`4080`-sample) sine window and
      parabolic local-maximum peak detection (`_local_peak_near`).
    - Propagated `chord_notes` and `chord_type` through `PitchFrame` (`analysis/pitch.py`),
      `NoteStabilizer` (`analysis/stabilizer.py`), `PitchEvent` (`music/events.py`), `MusicalState`
      (`music/state.py`), and `NoteTracker` (`ai/base.py`) so MRT2's 128-pitch pianoroll receives
      full guitar chord tones (`2 = onset`, `1 = held`).
  - **Instrument & AI Model Quality selectors (`config.py`, `session.py`, `main.py`, `ui/main_window.py`, `ui/debug_panel.py`):**
    - Added `ModelSize = Literal["mrt2_small", "mrt2_base"]`, `MODEL_SIZE_LABELS`, and
      `INSTRUMENT_LABELS` in `config.py`, and persisted `instrument: Instrument` and
      `model_size: ModelSize` in `UserPreferences` (`settings.json`).
    - Added `JamSession.set_instrument(instrument)` and `AnalysisWorker.set_instrument(instrument)`
      for live and between-session instrument switching.
    - Added `instrument_combo` (`Electric Guitar` / `Bass`) and `model_size_combo`
      (`Low Quality (mrt2_small)` / `High Quality (mrt2_base)`) to `MainWindow`, plus
      `--model-size {mrt2_small,mrt2_base}` CLI flags in `main.py` and `scripts/render_jam.py`, and
      `instrument` / `chord_type` / `ai_model_size` diagnostics in `DebugPanel`.
  - **Hot-plug audio device Refresh button (`audio/devices.py`, `ui/main_window.py`):**
    - Added `refresh_inventory()` in `audio/devices.py` (`sd._terminate()` + `sd._initialize()` +
      `query_inventory()`) and `refresh_devices_button` (`refresh_devices()`) in `MainWindow` to
      refresh both input and output dropdowns when a USB audio interface is plugged in or unplugged.
  - **Tests (`tests/analysis/test_guitar.py`, `tests/audio/test_devices.py`, `tests/audio/test_recorder.py`, `tests/ui/test_main_window.py`):**
    - 18 new guitar analysis tests covering 2-note and 3-note clean/overdriven power chords (`E5`,
      `A5`, `G5`, `D5`), open major/minor chords (`E`, `Em`, `G`, `A`, `Am`, `D`), single-note
      preservation, multi-note `NoteTracker` pianoroll conditioning, live `set_instrument()`
      switching, and real `guitar_chords_e_g_a_100bpm.wav` chord root extraction (`E2`, `G2`, `A2`).
- **Realtime safety:** `GuitarChordAnalyzer` runs exclusively on `AnalysisWorker`, never in the audio
  callback. `refresh_inventory()` is disabled while `JamSession` is running.
- **Verification:** `497 passed` in `pytest`, `ruff check .` and `ruff format --check .` clean.


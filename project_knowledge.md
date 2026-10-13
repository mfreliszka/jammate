# Project Knowledge — JamMate

## 1. Project Summary

JamMate is a local desktop application for musicians who want to practice or jam but do not have other musicians available.

The core idea is simple:

> Connect a bass guitar or electric guitar to the computer, start playing a riff, and have an AI-assisted virtual band listen to the player and join the jam.

The application is not primarily a music generator.

It is not intended to behave like Suno, Udio, or a text-to-song application.

It is also not intended to be a DAW such as Ableton Live, Logic Pro, or Reaper.

JamMate should behave like **other musicians in a rehearsal room**.

The target experience is:

1. The musician connects an instrument.
2. The musician presses `Jam`.
3. The musician starts playing without configuring BPM, key, chords, or a backing track.
4. JamMate listens.
5. JamMate determines enough musical information to follow the player.
6. A virtual drummer joins.
7. An AI-generated accompaniment joins.
8. The accompaniment reacts to changes in the player's performance.
9. The user can continue improvising naturally.

A useful product statement is:

> **Start playing. Your band joins in.**

Another possible product positioning is:

> **Never practice alone.**

The application should feel closer to playing with musicians than playing over a static backing track.

---

# 2. Core User Problem

The initial problem came from a bass player.

The bass player can practice riffs alone, but practicing alone becomes boring and does not simulate playing in a band.

A typical situation is:

```text
Bass player:
"I have this riff."

E E E G A E
E E G D E

But there is:
- no drummer,
- no guitarist,
- no band,
- no musical response.
```

Traditional solutions are insufficient.

## 2.1 Metronome

A metronome provides timing but no musical context.

```text
Bass
+
click click click click
```

It helps technique but does not feel like a jam.

---

## 2.2 Drum loops

Drum loops are better than a metronome but remain static.

They do not respond to:

- changes in intensity,
- riff changes,
- pauses,
- accents,
- transitions,
- improvisation.

---

## 2.3 Backing tracks

Backing tracks force the musician to follow an existing song structure.

JamMate should do the opposite:

> The virtual band follows the musician.

---

## 2.4 DAW workflows

A DAW can technically provide drums, loops, plugins, MIDI instruments, and recording.

However, that requires the musician to configure:

- tempo,
- tracks,
- routing,
- plugins,
- loops,
- key,
- song structure.

That is not the intended JamMate experience.

JamMate should require almost no setup.

---

# 3. Product Philosophy

The primary design principle is:

> **The musician should think about music, not software.**

The user should not need to understand:

- MIDI,
- audio routing,
- AI models,
- machine learning,
- buffer sizes,
- sample rates,
- chord detection,
- beat phase,
- DSP,
- prompt engineering.

The system may use all of those internally, but they should remain implementation details.

---

# 4. Product Category

JamMate should initially be considered a:

**local AI-assisted desktop practice and jamming application**

rather than:

- a website,
- a cloud service,
- a DAW,
- a social network,
- a music production suite,
- a mobile-first application.

The long-term product may eventually include some of those elements, but they should not influence the initial architecture unnecessarily.

---

# 5. Initial Target Platform

The first supported platform is:

```text
macOS
Apple Silicon
```

Examples:

```text
M1
M2
M3
M4
M5
and future Apple Silicon Macs
```

The initial project should deliberately avoid cross-platform complexity.

Do not initially support:

```text
Windows
Linux
iPhone
iPad
Android
Web browsers
VST
AU plugins
```

These can be considered later.

---

# 6. Why Desktop Instead of Web

Realtime audio applications have very different requirements from traditional web applications.

JamMate needs:

- low-latency audio input,
- low-latency audio output,
- reliable access to USB audio interfaces,
- stable timing,
- realtime audio callbacks,
- local AI inference,
- MIDI routing,
- predictable resource usage.

A browser introduces unnecessary complexity and constraints.

Therefore the first version should be a native/local desktop application.

There is no backend requirement for the prototype.

Do not introduce:

```text
FastAPI
Django
PostgreSQL
Redis
Celery
Kafka
Docker
Kubernetes
WebSockets
REST APIs
authentication
cloud infrastructure
```

unless a future requirement explicitly makes them necessary.

The application should initially work entirely offline.

---

# 7. Initial User

The first target user is a bass guitarist.

This is an intentional technical choice.

Bass guitar is usually monophonic:

```text
one dominant note at a time
```

This makes realtime pitch detection significantly easier than guitar chord recognition.

Example:

```text
82.41 Hz
→ E2
→ MIDI 40
```

This is a much easier signal interpretation problem than determining a distorted guitar chord containing several simultaneous pitches.

Therefore:

```text
v0.x:
Bass input

future:
Electric guitar
Keyboard/MIDI
Other instruments
```

---

# 8. MVP Scope

The first meaningful JamMate version should do only the following.

## Input

A bass guitar connected using a USB audio interface.

Example:

```text
Bass
  |
  v
Focusrite / Behringer / another audio interface
  |
  USB
  |
  v
Mac
```

---

## Detection

JamMate should estimate:

```text
current note
note onset
playing intensity
tempo
beat position
bar position
whether the player is currently playing
```

---

## Accompaniment

JamMate should provide:

```text
drums
AI-generated musical accompaniment
```

Initially the AI accompaniment can broadly act as the guitarist or supporting ensemble.

---

## Controls

Initial controls should be minimal:

```text
Input Device

Output Device

Instrument:
Bass

Style:
Rock
Stoner Rock
Blues
Funk
Metal

Drums:
ON/OFF
Volume

AI Music:
ON/OFF
Volume

Follow <------------> Improvise

START JAM
STOP JAM
```

Recording can be added shortly after the fundamental realtime pipeline works.

---

# 9. Non-Goals for the MVP

Do not implement the following unless explicitly requested.

```text
user accounts
cloud synchronization
social profiles
multiplayer
online jams
song sharing
community feed
marketplace
music publishing
mobile applications
Windows support
guitar chord recognition
full song generation
stem mastering
professional DAW features
plugin hosting
VST hosting
VST export
music notation
tabs
automatic song composition
complex project files
collaboration
cloud inference
custom model training
```

The project should remain deliberately small until the central musical experience has been validated.

---

# 10. MVP Success Criterion

The most important success criterion is not accuracy on a benchmark.

It is this test:

> A bass player connects the instrument, plays an arbitrary riff, and after several seconds feels that another musician is actually playing with them.

The critical reaction is something equivalent to:

> "This actually feels like jamming."

A technically impressive system that does not create this feeling is not successful.

---

# 11. High-Level System Architecture

The intended architecture is hybrid.

Not every musical task should use AI.

```text
                     BASS
                      |
                      v
             +----------------+
             | AUDIO INPUT    |
             | CoreAudio      |
             +-------+--------+
                     |
                     v
             +----------------+
             | AUDIO ANALYSIS |
             |                |
             | pitch          |
             | onset          |
             | intensity      |
             | tempo          |
             +-------+--------+
                     |
                     v
              +--------------+
              | MUSICAL      |
              | STATE        |
              +------+-------+
                     |
          +----------+-----------+
          |                      |
          v                      v
 +----------------+       +------------------+
 | RHYTHM ENGINE  |       | AI ACCOMPANIMENT |
 |                |       |                  |
 | BeatClock      |       | Magenta RT2      |
 | Drum patterns  |       | or future model  |
 +-------+--------+       +--------+---------+
         |                         |
         v                         v
 +----------------+       +------------------+
 | DRUM SAMPLER   |       | GENERATED AUDIO  |
 +-------+--------+       +--------+---------+
         |                         |
         +-----------+-------------+
                     |
                     v
               +-----------+
               | AUDIO MIX |
               +-----+-----+
                     |
                     v
                HEADPHONES
```

The architecture deliberately separates:

- audio transport,
- musical analysis,
- musical state,
- deterministic timing,
- drum scheduling,
- generative AI,
- GUI.

---

# 12. Hybrid AI Philosophy

The application should **not** attempt to solve every problem with generative AI.

Classical DSP and deterministic algorithms are often superior for low-latency realtime tasks.

Use AI where musical creativity is valuable.

Use deterministic systems where timing and reliability are critical.

For example:

| Problem | Preferred approach |
|---|---|
| Audio input | CoreAudio / PortAudio |
| Pitch detection | DSP / YIN |
| Note conversion | Mathematics |
| Signal level | RMS |
| Onset detection | DSP |
| Tempo estimation | Statistical/DSP |
| Beat clock | Deterministic timing |
| Drum sample playback | Deterministic |
| Drum event scheduling | Deterministic |
| Musical variation | AI / heuristics |
| Supporting guitar generation | Generative model |
| Long-term arrangement decisions | AI / musical rules |

This distinction is fundamental.

---

# 13. Why Drums Should Not Initially Be Pure Generative Audio

Realtime drums are extremely sensitive to timing.

A kick drum that occurs even noticeably late can make the entire jam feel wrong.

A generative model may have:

- inference latency,
- buffering,
- scheduling delay,
- jitter.

Therefore the first drummer should work approximately like this:

```text
MusicalState
      |
      v
Drum decision logic
      |
      v
scheduled MIDI-like drum events
      |
      v
sample playback
```

For example:

```text
16-step bar

HH  X-X-X-X-X-X-X-X-
SN  ----X-------X---
KD  X-----X-X-------
```

The audio engine then plays samples exactly at scheduled positions.

This provides stable timing.

AI can later decide:

- which pattern to use,
- when to add a fill,
- when to become more aggressive,
- when to simplify,
- when to switch groove.

But AI should not necessarily determine the timing of individual sample playback operations.

---

# 14. Primary Technology Stack

## Operating system

```text
macOS
Apple Silicon
```

---

## Language

```text
Python 3.12
```

Python is intentionally used for the prototype because:

- the developer is an experienced Python engineer,
- development velocity matters,
- AI coding agents work very effectively with Python,
- DSP experimentation is easier,
- the application architecture can later move critical pieces to C++.

Python should not be treated as a permanent requirement for every subsystem.

---

## Package / environment management

```text
uv
```

Use `uv` for:

- Python environment,
- dependency installation,
- lockfile,
- developer tooling.

Do not introduce Poetry unless there is a strong reason.

---

## GUI

```text
PySide6
Qt Widgets
```

Do not initially use:

```text
QML
Electron
React
webviews
browser UI
```

Qt Widgets are sufficient for a prototype desktop application.

---

## Audio

```text
sounddevice
PortAudio
CoreAudio backend
```

Expected configuration:

```text
Sample rate:
48,000 Hz

Input:
mono

Output:
stereo

Initial block size:
128 frames

Fallback:
256 frames
```

All audio-related values should be configurable rather than unnecessarily hard-coded, but 48 kHz should be the expected system sample rate.

---

# 15. DSP Stack

Use:

```text
NumPy
SciPy
librosa where useful
```

Initial bass pitch detector:

```text
YIN
```

Potential implementation:

```text
librosa.yin
```

Target bass frequency range approximately:

```text
35 Hz - 400 Hz
```

Exact thresholds should be configurable.

---

# 16. MIDI

Use:

```text
mido
python-rtmidi
```

MIDI is primarily used during the prototype as a bridge between:

```text
JamMate
and
the AI music engine
```

The pitch detector converts detected bass notes to MIDI notes.

Example:

```text
82.4 Hz
→ E2
→ MIDI 40
```

JamMate then produces events similar to:

```text
note_on(40)

note_off(40)
note_on(43)
```

MIDI operations must not occur in the realtime audio callback.

---

# 17. Initial AI Model

The first generative music engine should be:

```text
Google Magenta RealTime 2
```

For the earliest prototype, prefer the smaller realtime-capable model variant.

The purpose of MRT2 is not to analyze the bass.

The Python application performs most of the analysis.

MRT2 receives higher-level musical control such as:

```text
detected notes
musical prompt/style
potentially drum-related controls
other conditioning
```

and produces musical accompaniment.

---

# 18. Initial MRT2 Integration Strategy

Do not immediately embed the native Magenta C++ engine into JamMate.

The first integration should minimize engineering risk.

Use this architecture:

```text
Bass
 |
 v
JamMate Python
 |
 +----> drum engine
 |
 +----> MIDI
          |
          v
    MRT2 standalone host
          |
          v
   generated accompaniment
```

The standalone model host is treated as a separate process during experimentation.

JamMate should discover the appropriate MIDI output dynamically.

Do not rely unnecessarily on a fixed device/port identifier.

This architecture allows the team to validate the musical idea before investing in native integration.

---

# 19. Future Native MRT2 Integration

If the prototype proves valuable, remove the separate standalone process.

Future architecture:

```text
JamMate
  |
  +-- Python application/controller
  |
  +-- Native C++ bridge
          |
          v
      MRT2 realtime engine
```

The preferred native abstraction is the model's realtime runner rather than building a custom inference loop from scratch.

Native integration should eventually provide:

- an inference thread,
- output ring buffers,
- realtime-safe generated audio access,
- model parameter controls,
- MIDI conditioning,
- prompt controls.

Python can remain responsible for:

- product logic,
- configuration,
- Musical State,
- GUI,
- high-level orchestration.

Performance-critical components can progressively move to C++.

---

# 20. Suggested Repository Layout

```text
jammate/
|
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- AGENTS.md
|-- project_knowledge.md
|
|-- src/
|   `-- jammate/
|       |
|       |-- __init__.py
|       |-- AGENTS.md
|       |-- main.py
|       |
|       |-- audio/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   |-- devices.py
|       |   |-- engine.py
|       |   |-- ring_buffer.py
|       |   `-- recorder.py
|       |
|       |-- analysis/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   |-- pitch.py
|       |   |-- onset.py
|       |   |-- tempo.py
|       |   `-- notes.py
|       |
|       |-- music/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   |-- state.py
|       |   |-- events.py
|       |   `-- beat_clock.py
|       |
|       |-- drums/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   |-- engine.py
|       |   |-- patterns.py
|       |   `-- sampler.py
|       |
|       |-- ai/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   `-- mrt.py
|       |
|       |-- midi/
|       |   |-- __init__.py
|       |   |-- AGENTS.md
|       |   `-- magenta.py
|       |
|       `-- ui/
|           |-- __init__.py
|       |   |-- AGENTS.md
|           `-- main_window.py
|
|-- assets/
|   `-- drums/
|       |-- kick.wav
|       |-- snare.wav
|       |-- closed_hat.wav
|       |-- crash.wav
|       `-- tom.wav
|
|-- scripts/
|
`-- tests/
```

This structure may evolve, but avoid adding layers without concrete need.

---

# 21. Realtime Audio Rules

Realtime safety is one of the most important architectural requirements.

The audio callback must remain extremely small.

Conceptually:

```python
def audio_callback(input_buffer, output_buffer):
    # copy input into realtime-safe buffer
    # read already prepared output
    # mix already prepared samples
    # return
```

The callback must **not** perform:

```text
AI inference
librosa analysis
logging
database operations
disk I/O
network operations
GUI updates
Qt calls
MIDI operations
large allocations
slow Python operations
blocking locks
sleep()
waiting for other threads
```

---

# 22. Threading Model

Conceptually use separate execution contexts.

```text
AUDIO THREAD
    |
    +-- audio input
    +-- audio output
    +-- minimal sample mixing
    |
    v
RING BUFFER
    |
    v
ANALYSIS WORKER
    |
    +-- pitch
    +-- onset
    +-- intensity
    +-- tempo evidence
    |
    v
MUSICAL STATE
    |
    +------------------+
    |                  |
    v                  v
RHYTHM WORKER       AI CONTROL
    |                  |
    v                  v
DRUM EVENTS          MIDI/model
    |
    v
AUDIO OUTPUT

GUI THREAD
    |
    `-- reads snapshots of Musical State
```

The GUI should display state.

It should not own the authoritative realtime state.

---

# 23. Buffers and Data Flow

Audio data should travel using bounded buffers.

Never use an unbounded queue for audio.

Preferred model:

```text
Audio callback
     |
     v
fixed-size ring buffer
     |
     v
analysis worker
```

If the consumer falls behind, the system should have a deliberate strategy such as:

- dropping old analysis frames,
- incrementing an overrun counter,
- preserving current realtime behavior.

The system should never accumulate seconds or minutes of stale audio waiting to be processed.

Realtime behavior is more important than analyzing every sample.

---

# 24. Musical Events

The analysis layer should convert audio into lightweight semantic events.

Examples:

```python
PitchEvent
OnsetEvent
TempoUpdate
IntensityUpdate
SilenceEvent
```

Possible definitions:

```python
@dataclass(frozen=True)
class PitchEvent:
    timestamp: float
    frequency_hz: float
    midi_note: int
    note_name: str
    confidence: float
```

```python
@dataclass(frozen=True)
class OnsetEvent:
    timestamp: float
    strength: float
```

The precise schema may evolve.

Avoid passing raw audio everywhere.

Raw audio belongs primarily to the audio and analysis layers.

Musical decisions should increasingly operate on semantic events and Musical State.

---

# 25. Pitch Detection

Initial pitch detection targets monophonic bass.

Pipeline:

```text
audio
 |
 v
basic filtering / level validation
 |
 v
YIN pitch detection
 |
 v
frequency
 |
 v
MIDI note
 |
 v
note stabilization
 |
 v
PitchEvent
```

Example:

```text
82.36 Hz
→ MIDI 40
→ E2
```

---

# 26. Pitch Stabilization

Realtime pitch algorithms can produce temporary incorrect values.

Example:

```text
E2
E2
E2
F2
E2
E2
```

when the musician is actually holding E2.

Therefore raw pitch output must not directly become a musical note event.

Introduce stabilization rules such as:

- minimum note duration,
- hysteresis,
- confidence threshold,
- silence threshold,
- octave-error suppression,
- repeated-note suppression.

Do not emit a new `PitchEvent` every analysis frame if nothing musically changed.

---

# 27. Onset Detection

Pitch answers:

> What note is being played?

Onset detection answers:

> When was a new note attacked?

For rhythm analysis, onset timestamps are often more useful than continuous pitch frames.

Possible first implementation:

```text
RMS change
+
spectral flux
+
adaptive threshold
```

Do not require a neural network for initial onset detection.

---

# 28. Tempo Estimation

Tempo should be estimated from recent onset events.

Potential pipeline:

```text
Onset timestamps
     |
     v
inter-onset intervals
     |
     v
tempo candidates
     |
     v
half/double tempo normalization
     |
     v
rolling evidence
     |
     v
TempoState
```

Target initial tempo range:

```text
50 - 200 BPM
```

The system should expose something like:

```python
TempoState(
    bpm=93.2,
    confidence=0.84,
    locked=True,
)
```

Do not instantly change BPM because of one unusual note.

Tempo should adapt gradually.

---

# 29. Tempo Locking

Before enough evidence exists:

```text
tempo_locked = False
```

After the player establishes a consistent pulse:

```text
tempo_locked = True
```

The system can then start the virtual band at an appropriate musical boundary.

This behavior is desirable.

A real drummer does not necessarily start playing on the first sound.

The app may listen for several beats first.

---

# 30. Natural Band Entry

Do not consider a short startup delay automatically a failure.

A natural interaction can be:

```text
0.0 s
player starts riff

0-2 s
JamMate listens

2-4 s
tempo/rhythm confidence increases

next bar boundary
drummer joins

next suitable boundary
AI accompaniment joins
```

This can feel more natural than forcing an immediate response.

Use musical boundaries whenever possible.

---

# 31. BeatClock

Once the tempo is sufficiently stable, JamMate creates an internal musical clock.

Conceptually:

```python
BeatClock(
    bpm=92.0,
    beat=3,
    bar=5,
    phase=0.42,
)
```

For 4/4:

```text
BAR 1

beat 1
beat 2
beat 3
beat 4

BAR 2

beat 1
...
```

The BeatClock is crucial because downstream components should schedule events against a common time reference.

---

# 32. Beat Phase

Beat phase describes progress through the current beat.

Example:

```text
0.0 = beat begins
0.5 = halfway through beat
1.0 = next beat begins
```

This is useful for:

- scheduling drums,
- UI beat indicators,
- synchronization,
- correcting timing.

---

# 33. BeatClock Correction

Do not constantly destroy and recreate the BeatClock when the tempo estimator moves slightly.

Bad behavior:

```text
92 BPM
94 BPM
91 BPM
93 BPM

reset clock each time
```

This produces unstable accompaniment.

Instead use gradual synchronization.

Conceptually:

```text
internal beat clock
+
observed onsets
=
small phase correction
```

This behaves more like synchronization or a lightweight phase-locked loop.

---

# 34. The Musical State

The **Musical State** is one of the most important architectural concepts in JamMate.

It is the shared representation of:

> What does JamMate currently believe is happening musically?

Without Musical State, every subsystem would interpret the raw audio independently.

That would create duplication and inconsistent decisions.

Instead:

```text
audio
 |
 v
analysis
 |
 v
MUSICAL STATE
 |
 +---------------------+
 |          |          |
 v          v          v
drummer     AI         UI
```

---

# 35. Why Musical State Exists

Consider a bass riff.

The raw signal is simply thousands of floating-point audio samples.

The drummer does not need those samples.

The GUI does not need them.

The band director does not necessarily need them.

Those components need information like:

```text
tempo = 93 BPM

current beat = 2

bar = 6

current note = E2

recent notes = E E G A E

intensity = 0.72

player is currently playing

tempo confidence = 0.91
```

Musical State converts realtime signal processing into musical meaning.

---

# 36. Initial Musical State Schema

The first version should remain intentionally simple.

Example:

```python
class MusicalState:
    current_note: int | None
    recent_notes: list[int]

    bpm: float | None
    tempo_confidence: float
    tempo_locked: bool

    beat: int | None
    bar: int | None
    beat_phase: float

    intensity: float

    user_is_playing: bool
```

Potential snapshot:

```text
current_note       = 40
recent_notes       = [40, 40, 43, 45, 40]

bpm                = 92.7
tempo_confidence   = 0.88
tempo_locked       = true

beat               = 3
bar                = 7
beat_phase         = 0.32

intensity          = 0.68
user_is_playing    = true
```

---

# 37. What Should NOT Be in Musical State Initially

Do not prematurely add:

```text
song title
verse
chorus
bridge
complex chord progression
genre probabilities
emotion
musical intention
scale mode
arrangement tree
song structure graph
AI prompt history
```

These may appear later.

Keep the first state model observable and understandable.

---

# 38. Musical State Ownership

There should be a clear state owner.

Avoid having:

```text
UI state
drum state
analysis state
AI state
```

all independently claiming what the current tempo is.

Instead:

```text
analysis produces updates
      |
      v
MusicalStateManager
      |
      v
immutable/read-only snapshots
```

Other components consume snapshots.

---

# 39. Musical State Thread Safety

Because multiple workers may consume the state, avoid exposing a mutable object that every thread edits directly.

Prefer one of:

- atomic fields for simple values,
- controlled update methods,
- immutable snapshots,
- copy-on-read snapshots.

The GUI should not hold locks needed by the realtime audio system.

The audio callback should not depend on Musical State locks.

---

# 40. Future Musical State

As JamMate becomes more intelligent, Musical State can grow.

Possible future structure:

```python
MusicalState(
    bpm=93.2,
    meter="4/4",

    root="E",
    scale="minor",

    current_note="E2",

    recent_notes=[
        "E2", "E2", "G2", "A2", "E2"
    ],

    riff_id="riff_a",
    riff_iteration=4,

    section="riff_a",

    intensity=0.71,
    density=0.62,

    user_is_playing=True,

    predicted_next_boundary="bar",

    harmonic_confidence=0.82,
    rhythmic_confidence=0.91,
)
```

This should only be introduced incrementally.

---

# 41. Riff Detection

One future high-value feature is riff recognition.

Example:

```text
Bass plays:

E E E G A E
E E E G A E
E E E G A E
```

JamMate recognizes:

```text
same pattern repeated
→ riff A
```

Then the player changes:

```text
G G A C
G G A C
```

JamMate recognizes:

```text
new pattern
→ riff B
```

This enables the virtual band to make much more musical decisions.

---

# 42. Why Riff Detection Matters

A basic system knows:

```text
E2 is currently playing.
```

A better system knows:

```text
The user is playing riff A for the fourth time.
```

That enables behaviors like:

```text
riff repetition 4
→ add fill

riff change
→ crash cymbal

new riff becomes quieter
→ simplify arrangement

return to riff A
→ restore previous groove
```

This is where JamMate can begin to feel like an attentive band.

---

# 43. Drummer Architecture

The drummer should initially consist of:

```text
BeatClock
+
Pattern Library
+
Drum Scheduler
+
Sample Player
```

Example directory:

```text
drums/
    patterns.py
    engine.py
    sampler.py
```

---

# 44. Drum Patterns

Use simple 16-step grids.

Example rock groove:

```text
1 e & a 2 e & a 3 e & a 4 e & a

HH  X-X-X-X-X-X-X-X-
SN  ----X-------X---
KD  X-------X-------
```

Example stoner groove:

```text
HH  X-X-X-X-X-X-X-X-
SN  ----X-------X---
KD  X-----X-X-------
```

Example heavier groove:

```text
HH  X-X-X-X-X-X-X-X-
SN  ----X-------X---
KD  X-X---X-X-X-----
```

Exact patterns are not permanent product decisions.

They are scaffolding for validating the experience.

---

# 45. Drum Samples

The prototype needs only a small kit:

```text
kick
snare
closed hi-hat
crash
tom
```

All samples must be loaded before realtime playback begins.

Never perform:

```text
open("kick.wav")
```

inside the audio callback.

Decode/load samples into memory during initialization.

---

# 46. Drum Scheduling

Pattern decisions should happen before the exact playback time.

The realtime audio system should receive events equivalent to:

```text
sample 153820:
kick

sample 177820:
snare
```

or another deterministic scheduling representation.

The callback's responsibility is playback/mixing, not deciding whether a kick drum is musically appropriate.

---

# 47. Intensity

JamMate should estimate how intensely the user is playing.

Initial intensity may use:

```text
RMS / loudness
+
onset strength
+
note density
```

Normalize to:

```text
0.0 - 1.0
```

Example:

```text
0.15
very quiet playing

0.45
normal groove

0.90
aggressive playing
```

---

# 48. Using Intensity

The drummer can respond immediately without sophisticated AI.

Example:

```text
intensity < 0.3

closed hat
simple kick
no crash
```

```text
0.3 <= intensity < 0.7

standard groove
```

```text
intensity >= 0.7

stronger kick pattern
open hat / crash
higher drum density
```

Even simple behavior can create the illusion that the virtual drummer is listening.

---

# 49. Drum Fills

The initial drummer may add simple deterministic fills.

For example:

```text
every 8 bars
```

or later:

```text
before detected riff transition
```

Example:

```text
tom
tom
snare
crash
```

Do not over-invest in sophisticated fills before the main synchronization feels good.

---

# 50. AI Accompaniment

The initial generative AI layer should primarily provide harmonic/textural accompaniment.

The first use case is approximately:

```text
Bass player
+
deterministic drummer
+
AI rhythm guitar / ensemble
```

Example conditioning concept:

```text
Style:
fuzzy stoner rock

Role:
supportive rhythm guitarist

Mood:
heavy, dark

Detected notes:
E G A

Intensity:
medium
```

---

# 51. AI Should Be Abstracted

Do not couple the entire application directly to one model.

Provide an interface conceptually similar to:

```python
class AccompanimentEngine(Protocol):

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def set_style(self, style: str) -> None:
        ...

    def note_on(self, midi_note: int) -> None:
        ...

    def note_off(self, midi_note: int) -> None:
        ...

    def set_intensity(self, intensity: float) -> None:
        ...
```

MRT2 can implement this interface.

A future model can replace it.

---

# 52. Model Independence

Future alternatives may include:

- another local realtime music model,
- a commercial realtime API,
- MIDI-generating models,
- specialized guitar generation,
- custom JamMate models.

Application code should not contain MRT2-specific assumptions everywhere.

Keep model-specific logic inside:

```text
ai/
```

or an equivalent adapter.

---

# 53. Guitar Input

Guitar input is a future feature.

It is much more difficult than bass because a guitar often contains simultaneous notes.

Example power chord:

```text
E
B
E
```

Distortion introduces additional harmonic content.

Other complications include:

```text
bends
slides
hammer-ons
pull-offs
harmonics
string noise
feedback
```

Therefore guitar support should not block the bass MVP.

---

# 54. Future Guitar Transcription

A future implementation may investigate models such as Basic Pitch or a specialized realtime guitar-to-MIDI model.

However, do not assume that an offline transcription model can automatically be used as a low-latency streaming model.

Potential future pipeline:

```text
guitar input
   |
   v
streaming polyphonic transcription
   |
   v
note set
   |
   v
chord/root interpretation
   |
   v
Musical State
```

This should be treated as a separate R&D effort.

---

# 55. Keyboard/MIDI Support

MIDI keyboard support should eventually be relatively straightforward.

A keyboard already gives JamMate:

```text
exact note
exact note-on time
exact note-off time
velocity
multiple simultaneous notes
```

No pitch transcription is required.

This could become an attractive second instrument after bass.

---

# 56. Initial GUI Philosophy

The interface should be exceptionally simple.

Possible main screen:

```text
+--------------------------------------+
|              JAMMATE                 |
|                                      |
| Input                                |
| [ Scarlett 2i2                 v ]   |
|                                      |
| Detected                             |
|                                      |
| E2                 92 BPM            |
| o  o  O  o                           |
| 1  2  3  4                           |
|                                      |
| Style                                |
| [ Stoner Rock                  v ]   |
|                                      |
| Drums       [ ON ]  ========---      |
| AI Music    [ ON ]  ======-----      |
|                                      |
| Follow --------o--------- Improvise  |
|                                      |
|           [ STOP JAM ]               |
+--------------------------------------+
```

The GUI is not the priority.

A crude interface with excellent musical interaction is better than a beautiful interface with poor timing.

---

# 57. UX Principle: No Music Theory Required

Avoid requiring users to select:

```text
Key: E minor
Scale: Aeolian
Tempo: 93 BPM
Time signature: 4/4
Chord progression: Em-G-A
```

The default behavior should infer these where possible.

Advanced users may eventually override them.

The default interaction remains:

```text
connect
press Jam
play
```

---

# 58. Manual Overrides

Automatic detection will sometimes fail.

A good product should not insist that automation is always correct.

Future controls may include:

```text
Auto BPM: ON/OFF

Tap Tempo

BPM override

Key override

4/4 / 3/4 / 6/8

Reset musical context
```

These are escape hatches, not primary workflow requirements.

---

# 59. Recording

A useful early feature is recording a complete jam.

Initial version:

```text
Record master stereo output
```

Example output:

```text
recordings/
    2026-10-09_21-45-jam.wav
```

Later versions can provide stems:

```text
bass.wav
drums.wav
ai_guitar.wav
master.wav
```

Recording must not compromise realtime audio performance.

---

# 60. Prototype Development Phases

The initial development should progress vertically rather than building many infrastructure components.

## Phase 1 — Audio

Goal:

```text
Bass
→ Python
→ headphones
```

Requirements:

- enumerate devices,
- select input/output,
- stable passthrough,
- 48 kHz,
- no repeated underruns.

---

## Phase 2 — Pitch

Goal:

```text
Bass E2
→ application displays E2
```

Requirements:

- monophonic pitch,
- note stabilization,
- silence handling.

---

## Phase 3 — Rhythm

Goal:

```text
riff
→ approximate tempo
→ stable BeatClock
```

Requirements:

- onsets,
- tempo evidence,
- tempo confidence,
- beat phase.

---

## Phase 4 — Drums

Goal:

```text
play riff
→ drummer joins
```

Requirements:

- pattern scheduler,
- sample playback,
- beat synchronization,
- basic dynamic response.

---

## Phase 5 — AI

Goal:

```text
bass
+
drums
+
AI accompaniment
```

The first AI integration may remain external through MIDI.

---

## Phase 6 — Product UI

Only after the fundamental loop works:

```text
connect
listen
analyze
follow
play
```

should significant GUI work begin.

---

# 61. Development Strategy With AI Coding Agents

AI agents should receive small, bounded tasks.

Bad prompt:

```text
Build an AI app that listens to bass and generates a band.
```

This will encourage:

- premature architecture,
- unnecessary frameworks,
- huge changes,
- low confidence code.

Better:

```text
Implement only audio device discovery.

Do not implement DSP, UI, AI, recording, or networking.
```

Then test manually.

---

# 62. Recommended Agent Workflow

For each feature:

```text
1. Agent inspects repository.

2. Agent proposes a small implementation.

3. Agent implements one subsystem.

4. Agent runs unit tests.

5. Agent runs ruff.

6. Developer runs real instrument test.

7. Fix actual observed problems.

8. Commit.

9. Move to next subsystem.
```

Realtime audio requires real-world testing.

Automated tests cannot replace:

```text
plug bass into interface
put on headphones
play for ten minutes
```

---

# 63. Agent Global Instructions

AI agents working on this repository should follow these principles.

```text
JamMate is an experimental realtime audio application.

Prefer simple implementations.

Do not introduce distributed systems.

Do not introduce web services unless explicitly requested.

Protect the realtime audio callback.

Prefer deterministic DSP over AI when AI does not add value.

Do not over-engineer.

Do not implement speculative future features.

Do not change architecture globally to solve a small local problem.

Keep dependencies minimal.

Test mathematical and scheduling logic.

Use real hardware testing for audio behavior.

Keep model-specific integrations behind adapters.

Musical quality and latency are more important than conventional CRUD application architecture.
```

---

# 64. Realtime Safety Checklist for Agents

Before merging audio-related code, inspect specifically for:

```text
[ ] logging inside audio callback
[ ] file I/O inside audio callback
[ ] MIDI operations inside audio callback
[ ] GUI calls inside audio callback
[ ] network calls inside audio callback
[ ] model inference inside audio callback
[ ] blocking locks
[ ] unbounded queues
[ ] excessive memory allocations
[ ] expensive NumPy operations
[ ] librosa calls
[ ] sleeps/waits
[ ] thread shutdown bugs
[ ] output underrun handling
[ ] input overflow handling
```

Any of these inside the callback should be treated as suspicious.

---

# 65. Testing Strategy

Different parts require different tests.

## Unit tests

Use unit tests for:

```text
Hz → MIDI
MIDI → note name
pitch stabilization
tempo candidate calculation
half/double tempo normalization
BeatClock math
drum grid scheduling
Musical State updates
MIDI note lifecycle
```

---

# 66. Fake Clock

Beat-related tests should use a controllable fake clock.

Avoid tests that literally sleep for musical durations.

Example:

```python
clock.advance(0.5)
```

is preferable to:

```python
time.sleep(0.5)
```

This makes tests deterministic and fast.

---

# 67. Audio Integration Tests

Some audio systems are difficult to test in CI.

Do not create brittle CI requirements dependent on physical hardware.

Separate:

```text
audio transport interface
```

from:

```text
music logic
```

so most of the application can be tested without a Scarlett interface attached.

---

# 68. Manual Test Scenarios

Maintain a small manual checklist.

## Bass note test

Play:

```text
E2
A2
D3
G3
```

Expected:

- correct approximate note,
- no wild octave jumps,
- silence returns `None`.

---

## Stable riff test

Play a repeated simple riff.

Expected:

- BPM stabilizes,
- BeatClock does not jump constantly.

---

## Drum test

Play continuously for ten minutes.

Expected:

- no clicks,
- no major drift,
- no crashes,
- drums remain usable.

---

## Silence test

Stop playing.

Expected:

- `user_is_playing=False`,
- active MIDI note receives `note_off`,
- no stuck generated note.

---

# 69. MIDI Lifecycle

MIDI note lifecycle must be correct.

When pitch changes:

```text
E2
→ note_on 40

G2
→ note_off 40
→ note_on 43
```

When playing stops:

```text
note_off 43
```

During shutdown:

```text
All Notes Off
```

Never leave hanging notes.

---

# 70. Silence Detection

Silence is musically meaningful.

The system needs to distinguish:

```text
user is sustaining note

user stopped

signal is too quiet

noise only
```

A silence state can influence:

- MIDI note-off,
- AI gating,
- drum continuation,
- future band behavior.

---

# 71. Future Silence Behavior

Eventually silence could become contextual.

Examples:

```text
bass stops for 1 beat
→ band continues
```

```text
bass stops for 2 bars
→ drummer continues lightly
```

```text
bass stops for 8 bars
→ band fades / waits
```

This should become musical behavior rather than only signal detection.

---

# 72. Band Director

A future major subsystem should be the **Band Director**.

This may become one of JamMate's most valuable pieces of product intelligence.

Its role is not raw audio generation.

Its role is deciding:

> What should the virtual musicians do next?

Architecture:

```text
Musical State
      |
      v
Band Director
      |
      +------------+
      |            |
      v            v
Drummer         AI guitarist
```

---

# 73. Band Director Decisions

Possible future decisions:

```text
player repeated riff four times
→ introduce variation

player becomes louder
→ increase arrangement intensity

player changes riff
→ crash + transition

player pauses
→ maintain groove

new riff is sparse
→ simplify guitar

8 bars completed
→ drum fill

player starts playing high register
→ reduce competing guitar density
```

This orchestration intelligence may become more important than the specific generative model.

---

# 74. Follow vs Improvise

One useful product control is:

```text
FOLLOW <----------------> IMPROVISE
```

At high `Follow`:

- accompaniment closely follows detected notes,
- fewer harmonic surprises,
- simpler parts,
- more predictable rhythm.

At high `Improvise`:

- greater generative freedom,
- more variation,
- more independent musical ideas.

Internally this can influence:

- AI conditioning strength,
- pattern variation,
- harmonic freedom,
- density,
- temperature/sampling parameters.

Do not hard-code the implementation prematurely.

Treat this as a product-level semantic parameter.

---

# 75. Style Presets

Initial styles:

```text
Rock
Stoner Rock
Blues
Funk
Metal
```

These should eventually be presets configuring several systems simultaneously.

A style should not merely be a text string.

For example:

```text
STONER ROCK

AI prompt:
fuzzy, heavy, slow, dark rhythm guitar

drum family:
heavy rock

fill frequency:
medium

humanization:
medium

density:
medium
```

This can evolve into structured style configuration.

---

# 76. Avoid Prompt-Only Architecture

Do not build the product around enormous natural-language prompts.

Prompting is one component.

A robust system should combine:

```text
structured musical state
+
deterministic timing
+
model control parameters
+
small semantic prompts
```

rather than:

```text
"Please play a guitar part that sounds good with what the bassist is playing..."
```

on every interaction.

---

# 77. Latency Philosophy

There are several kinds of latency.

They should not be confused.

## Monitoring latency

Time between:

```text
bass string attack
→ hearing own bass
```

This must be very low.

---

## Analysis latency

Time between:

```text
note attack
→ JamMate recognizing the note
```

It can be somewhat larger but should remain responsive.

---

## Musical reaction latency

Time between:

```text
musical change
→ band reacting
```

This does not always need to be minimal.

Sometimes reacting on the next beat or next bar is more musical.

---

## Generative latency

AI generation may have larger latency.

This should be hidden using:

- buffering,
- prediction,
- musical scheduling,
- continuous generation.

The product should optimize **perceived musical responsiveness**, not simply minimize every numerical latency value.

---

# 78. Timing Ahead Instead of Reacting Late

Musical accompaniment often knows future timing because the tempo is stable.

If BPM is known:

```text
92 BPM
```

the system knows approximately when the next:

```text
beat
bar
8-bar boundary
```

will occur.

Therefore it can schedule future events.

This is far superior to waiting until after a beat has occurred and then deciding what should have happened.

---

# 79. Error Philosophy

JamMate will frequently be uncertain.

Do not hide this fact internally.

Store confidence.

Examples:

```text
pitch confidence
tempo confidence
harmonic confidence
riff confidence
```

Future decisions can use uncertainty.

Example:

```text
tempo confidence < 0.4
→ don't start drums yet
```

This is better than pretending every estimate is equally reliable.

---

# 80. Logging

Logging should be useful for debugging but must remain outside realtime code.

Useful diagnostic information:

```text
audio device configuration
sample rate
buffer size
xrun counters
pitch events
tempo lock/unlock
BeatClock corrections
MIDI connection
MIDI note events
AI lifecycle
model load errors
```

Never log every audio frame.

---

# 81. Metrics for Development

Useful runtime diagnostics:

```text
input overflow count
output underflow count
analysis queue fill percentage
analysis processing time
pitch confidence
tempo confidence
detected BPM
BeatClock phase correction
AI buffer level
AI inference duration
MIDI connection status
```

These can initially appear in a developer/debug panel.

---

# 82. Graceful Degradation

The app should remain usable if AI fails.

Example:

```text
MRT2 unavailable
```

JamMate should still potentially provide:

```text
bass monitoring
+
drums
```

Do not make the entire audio engine crash because the generative model disconnected.

Subsystems should have clear failure boundaries.

---

# 83. Startup Flow

Possible startup sequence:

```text
Launch application

↓

Load configuration

↓

Enumerate audio devices

↓

Initialize samples

↓

Initialize analysis workers

↓

Attempt AI/MIDI connection

↓

Display ready state

↓

User selects input/output

↓

User presses START JAM

↓

Open audio stream

↓

Begin listening
```

Avoid doing expensive initialization after realtime audio starts whenever possible.

---

# 84. Shutdown Flow

Shutdown must be explicit.

Example:

```text
Stop GUI timer

Stop musical generation

Send MIDI note-off / All Notes Off

Stop analysis workers

Stop audio stream

Flush recording if active

Release devices

Exit
```

Threads must terminate cleanly.

Avoid daemon threads used as a substitute for lifecycle management.

---

# 85. Configuration

Initial configuration can remain in memory.

Later store simple preferences:

```text
last input device
last output device
buffer size
selected style
drums volume
AI volume
```

Do not add a database for these settings.

Use a simple local config mechanism.

---

# 86. Packaging

Packaging is not an early milestone.

Development should initially run as:

```bash
uv run ...
```

Once the musical prototype works, package it as a normal macOS application.

Potential packaging tools can be evaluated later.

Do not let packaging complexity delay musical experimentation.

---

# 87. Architecture Migration Strategy

The project should allow gradual native optimization.

## Stage 1

```text
mostly Python
external AI host
```

## Stage 2

```text
Python app
+
small C++ audio/model bridge
```

## Stage 3

```text
native realtime engine
+
Python high-level control or fully native application if justified
```

Do not rewrite everything in C++ before there is performance evidence requiring it.

---

# 88. What Python Is Good For Here

Python is suitable for:

```text
prototype orchestration
state management
DSP experimentation
tempo algorithms
musical rules
MIDI
GUI prototype
configuration
recording coordination
tests
```

---

# 89. What May Eventually Move to C++

Potential native components:

```text
audio callback
sample mixer
sample-accurate scheduler
lock-free buffers
embedded model inference
critical DSP
```

Move them only when necessary.

---

# 90. Future Key Detection

Eventually JamMate should infer likely tonal center and scale.

Possible information:

```text
recent pitch-class histogram
note duration
note emphasis
phrase ending notes
repetition
```

Output:

```text
root = E
mode ≈ minor
confidence = 0.76
```

This should remain probabilistic.

A bass riff may not uniquely determine a key.

---

# 91. Future Chord Understanding

Bass alone does not always uniquely define chords.

Example:

```text
bass note E
```

could belong to many harmonic contexts.

Therefore avoid pretending harmonic interpretation is certain.

Future guitar/keyboard input will provide much stronger harmonic evidence.

AI accompaniment should tolerate ambiguity.

---

# 92. Future Section Detection

Long-term JamMate can infer sections such as:

```text
riff A
riff B
break
solo section
build-up
outro
```

Avoid assuming conventional:

```text
verse
chorus
bridge
```

for every jam.

Instrumental jams often have different structures.

---

# 93. Future Adaptive Drummer

A better drummer can analyze riff rhythm.

Example bass rhythm:

```text
X--X-X---X--X---
```

The kick could mirror important bass accents.

Future architecture:

```text
bass onset pattern
     |
     v
rhythm representation
     |
     v
drum pattern generator
     |
     v
kick alignment
```

This could create much stronger perceived musical interaction.

---

# 94. Future Drum AI

Eventually AI may generate drum patterns.

However the output should ideally remain symbolic:

```text
kick at step 0
snare at step 4
kick at step 7
...
```

rather than directly generating realtime drum audio.

Symbolic generation preserves precise scheduling.

---

# 95. Future Humanization

Perfect quantization may sound artificial.

The drummer may eventually support:

```text
microtiming
velocity variation
swing
ghost notes
dynamic accents
```

Humanization must never introduce noticeable instability.

---

# 96. Future AI Roles

Eventually the user could select virtual band members:

```text
Drummer
Rhythm Guitar
Lead Guitar
Keyboard
Synth
Additional Bass
Percussion
```

But avoid this complexity initially.

The first convincing experience matters more than quantity of virtual instruments.

---

# 97. Future "Band Personalities"

Potential presets:

```text
Tight drummer
Loose drummer
Aggressive drummer
Minimal drummer

Supportive guitarist
Experimental guitarist
Heavy guitarist
Ambient guitarist
```

These can map to structured behavior and model parameters.

---

# 98. Future Song Memory

JamMate could remember patterns within the current session.

Example:

```text
riff A
→ groove A

riff B
→ groove B

user returns to riff A
→ drummer recalls groove A
```

This would feel significantly more intelligent than independently generating every bar.

---

# 99. Future Session Timeline

A jam could eventually be represented as:

```text
00:00 intro
00:13 riff A
00:42 riff B
01:10 riff A
01:38 break
01:54 riff C
```

This enables:

- navigation,
- export,
- editing,
- automatic highlights.

Not required for MVP.

---

# 100. Future Practice Features

After the jamming experience is strong, JamMate could add practice tools.

Examples:

```text
tempo consistency
timing accuracy
note detection
riff looping
slow practice
progressive BPM
recording comparison
practice history
```

These should remain secondary to the core "bandmate" experience.

---

# 101. Future Mobile Version

A future iOS/iPadOS version could be interesting because Apple devices provide strong local processing and good audio APIs.

However mobile should not influence early implementation choices.

Potential eventual setup:

```text
bass
→ mobile audio interface
→ iPhone/iPad
→ JamMate
```

Android would require separate evaluation because audio latency varies across hardware.

---

# 102. Future Windows Version

Windows support may eventually be commercially important.

It will introduce different audio paths such as:

```text
WASAPI
ASIO
```

Cross-platform audio architecture should be evaluated only after the macOS product proves valuable.

Do not introduce Windows abstractions prematurely.

---

# 103. Future Plugin Version

A future version could run inside a DAW as:

```text
VST3
AU
```

Possible use:

```text
Logic Pro track
→ JamMate plugin
```

However this serves a somewhat different user persona from the simple standalone practice application.

Standalone remains the primary initial product.

---

# 104. Future Cloud Features

Potential future cloud functionality:

```text
account
preset synchronization
jam backup
sharing
collaboration
community riffs
AI processing
```

None of these are required to validate the core value proposition.

Avoid backend development until a concrete product requirement exists.

---

# 105. Future Business Model Possibilities

Potential models include:

```text
paid desktop application
freemium application
subscription for advanced models
premium style packs
premium virtual musicians
cloud collaboration subscription
```

The technical architecture should not currently optimize around any one business model.

---

# 106. Intellectual Property / Product Moat

The generative model itself is unlikely to be JamMate's strongest moat.

Models will improve and can be replaced.

More defensible product intelligence can come from:

```text
realtime musical state estimation
riff detection
band interaction logic
Band Director behavior
low-latency orchestration
musical synchronization
quality presets
great UX
session memory
```

The key experience is:

> "The band understands what I am doing."

not:

> "The application uses model X."

---

# 107. Avoid Model Lock-In

Assume the AI model will eventually change.

A future model may be:

- faster,
- better sounding,
- more controllable,
- easier to license,
- cross-platform.

Therefore the architecture should make model replacement possible.

---

# 108. Product Validation Strategy

Do not spend months polishing before musician testing.

Recommended validation loop:

```text
build feature
↓
give bass to musician
↓
play
↓
observe reaction
↓
identify what breaks immersion
↓
fix that
```

Questions to ask testers:

```text
Did the drummer feel late?

Did the band understand your riff?

Did you feel forced to follow the software?

Was it more enjoyable than a backing track?

Would you practice with this?

When did the illusion of "playing with musicians" break?
```

---

# 109. The Most Important Metric

An internal product metric could eventually be:

> **Continuous jam duration**

If users voluntarily play for:

```text
10
20
30
60 minutes
```

that indicates genuine engagement.

A flashy 30-second demo is less valuable than something musicians enjoy using repeatedly.

---

# 110. First Four-Day Prototype Goal

After approximately the first four focused development days, the desired technical slice is:

```text
bass
 |
 v
audio interface
 |
 v
JamMate
 |
 +-- live monitoring
 |
 +-- note detection
 |
 +-- onset detection
 |
 +-- approximate BPM
 |
 +-- BeatClock
 |
 +-- automatic drums
 |
 +-- MIDI control
 |
 v
AI accompaniment
```

A polished app is not expected.

A convincing experiment is.

---

# 111. First Development Milestone

Milestone:

```text
Bass Audio Pipeline
```

Definition of done:

```text
JamMate detects the audio interface.

Bass can be monitored through the app.

48 kHz operation is stable.

No major clicks/dropouts for a sustained test.

Audio callback remains realtime-safe.
```

---

# 112. Second Development Milestone

Milestone:

```text
Bass Understanding
```

Definition of done:

```text
JamMate displays stable bass notes.

Example:

E2
E2
G2
A2
E2

Onsets are detected.

Approximate tempo is estimated.

Musical State receives updates.
```

---

# 113. Third Development Milestone

Milestone:

```text
Virtual Drummer
```

Definition of done:

```text
Player begins riff.

Tempo stabilizes.

BeatClock starts.

Drums enter at an appropriate beat/bar boundary.

Drums remain synchronized enough to be enjoyable.

Basic intensity changes affect groove.
```

---

# 114. Fourth Development Milestone

Milestone:

```text
First AI Band
```

Definition of done:

```text
Bass
+
Drums
+
AI accompaniment

can run simultaneously.

Bass note events control/condition the AI system.

The user can jam for several minutes.
```

---

# 115. Go / No-Go Decision

After this vertical slice, ask:

> Is this fun?

Not:

```text
Is the architecture elegant?
Is test coverage 100%?
Is the UI beautiful?
```

If the jam feels compelling, continue.

If it feels like unrelated generated music playing beside the bassist, focus on musical interaction before adding features.

---

# 116. Key Technical Risks

## Risk 1 — Audio latency

Symptoms:

```text
player hears delay
drums feel late
playing feels uncomfortable
```

Mitigation:

```text
small buffers
minimal callback
native optimization where necessary
```

---

## Risk 2 — Tempo instability

Symptoms:

```text
drums accelerate
slow down
jump
```

Mitigation:

```text
confidence
tempo locking
gradual correction
BeatClock
```

---

## Risk 3 — Pitch errors

Symptoms:

```text
AI receives incorrect notes
octave jumps
harmonic confusion
```

Mitigation:

```text
bass-first MVP
stabilization
confidence
frequency constraints
```

---

## Risk 4 — AI feels unrelated

Symptoms:

```text
generated music sounds good
but doesn't sound like it responds to the player
```

This is likely one of the most important risks.

Mitigation:

```text
stronger note conditioning
Musical State
riff tracking
Band Director
session memory
better timing of musical changes
```

---

# 117. Risk 5 — AI latency

Mitigation:

```text
continuous generation
buffering
predictable musical boundaries
native inference
separate deterministic drums
```

Do not require generative AI to solve events with millisecond-level timing.

---

# 118. Risk 6 — Over-Engineering

This is especially important because the primary developer has professional backend experience.

Do not treat JamMate like a backend SaaS product.

Avoid instinctively adding:

```text
repositories
services
controllers
API gateways
ORM
database migrations
message brokers
microservices
cloud deployment
```

This application is primarily:

```text
audio
DSP
musical state
scheduling
AI
desktop UI
```

Architecture should reflect the actual problem.

---

# 119. Code Style

Prefer:

```text
small modules
typed interfaces
dataclasses
clear ownership
explicit lifecycle
dependency injection where actually useful
pure functions for math
tests for deterministic logic
```

Avoid unnecessary:

```text
abstract factories
deep class inheritance
framework-like internal systems
enterprise patterns
```

---

# 120. Python Concurrency

Do not assume `asyncio` is the answer to realtime audio.

The realtime path is not an HTTP application.

Potential mechanisms:

```text
audio callback
worker threads
ring buffers
bounded queues
native threads inside model engine
```

Use `asyncio` only if a future non-realtime subsystem clearly benefits from it.

---

# 121. Performance Measurement

Do not optimize based only on assumptions.

Measure:

```text
callback execution time
analysis time
queue lag
pitch detection duration
AI inference time
underruns
buffer occupancy
```

Use profiling before rewriting components.

---

# 122. Audio Monitoring

The user may want to hear their raw bass.

The system should eventually support:

```text
direct input monitoring
+
drums
+
AI accompaniment
```

Input monitoring volume may be configurable.

Avoid unnecessary processing of the player's bass in the initial version.

---

# 123. Future Bass Effects

Future optional processing:

```text
compressor
EQ
amp simulation
distortion
noise gate
```

This is not part of the core product.

Integrating full guitar/bass effects would move the product toward a DAW/plugin host and should be approached carefully.

---

# 124. Long-Term Architecture Concept

A mature JamMate might look like:

```text
                        LIVE AUDIO
                            |
                            v
                    SIGNAL ANALYSIS
                            |
            +---------------+---------------+
            |                               |
            v                               v
      RHYTHM MODEL                    HARMONIC MODEL
            |                               |
            +---------------+---------------+
                            |
                            v
                    MUSICAL STATE
                            |
                            v
                     BAND DIRECTOR
                            |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
      DRUMMER            GUITARIST          KEYBOARD
         |                  |                  |
         v                  v                  v
  symbolic events       generative AI     generative AI
         |                  |                  |
         +------------------+------------------+
                            |
                            v
                         MIXER
                            |
                            v
                         OUTPUT
```

The MVP implements only a small subset of this architecture.

---

# 125. Architectural Boundary: Analysis vs Decision

Keep these concepts separate.

## Analysis

Answers:

```text
What note?
When?
How loud?
What tempo?
Where is the beat?
```

## Decision

Answers:

```text
Should drums become busier?
Should guitar simplify?
Should there be a fill?
Should the band transition?
```

This distinction will become increasingly important.

---

# 126. Architectural Boundary: Decision vs Rendering

Another important separation:

## Decision

```text
play kick on beat 1
```

## Rendering

```text
mix kick.wav into audio samples at frame 102400
```

Similarly:

## Decision

```text
support current riff with heavy rhythm guitar
```

## Rendering

```text
AI model generates audio
```

This separation allows musical intelligence to evolve independently from low-level audio rendering.

---

# 127. Future Event Bus

A lightweight internal event mechanism may eventually be useful.

Possible events:

```text
NoteStarted
NoteEnded
TempoLocked
BeatStarted
BarStarted
IntensityChanged
RiffDetected
RiffChanged
PlayerStopped
```

Do not introduce a complex event framework prematurely.

Simple typed callbacks or bounded queues are enough initially.

---

# 128. Musical Time vs Wall Time

The system needs to distinguish:

```text
wall-clock time
```

from:

```text
musical time
```

Wall time:

```text
12.392 seconds since stream start
```

Musical time:

```text
bar 8
beat 3
phase 0.41
```

Musical decisions should increasingly operate in musical time.

---

# 129. Monotonic Clock

Timing logic should use a monotonic clock.

Do not rely on system wall-clock timestamps that can change due to:

```text
NTP correction
timezone
manual clock changes
```

Recording filenames can use wall time.

Realtime scheduling should use monotonic time or audio sample positions.

---

# 130. Sample Position

Where possible, the most precise audio scheduling reference is:

```text
audio sample frame index
```

Example:

```text
kick scheduled at sample 384000
```

This avoids accumulated timing errors from repeatedly converting wall time.

The prototype may begin simpler and become more sample-oriented as needed.

---

# 131. Data Types

Prefer explicit types.

Examples:

```python
type MidiNote = int
type SampleFrame = int
```

or small dataclasses if useful.

Be careful with units.

Avoid ambiguous variables like:

```python
time
duration
position
```

when the unit is unclear.

Prefer:

```python
timestamp_seconds
duration_samples
frequency_hz
bpm
```

---

# 132. Configuration Values

Important parameters should not be magic numbers scattered across modules.

Examples:

```text
sample_rate
block_size
pitch_min_hz
pitch_max_hz
silence_threshold
tempo_min_bpm
tempo_max_bpm
tempo_lock_confidence
pitch_hysteresis
analysis_window
```

Use centralized typed configuration.

Do not turn configuration into a giant framework.

---

# 133. Dependency Policy

Before adding a dependency, ask:

```text
Does the standard library already solve this?

Does NumPy/SciPy solve it?

Is the dependency actively maintained?

Is the license appropriate?

Does it introduce huge transitive dependencies?

Can we remove it later?
```

Music/ML Python ecosystems can easily accumulate heavy dependencies.

Keep the environment manageable.

---

# 134. Licensing

The application may eventually become commercial.

Therefore agents should consider library/model licenses before introducing dependencies.

Do not casually introduce GPL-only dependencies into a closed-source commercial architecture without explicit approval.

When evaluating new music models, verify:

```text
code license
model-weight license
commercial-use restrictions
output restrictions
redistribution requirements
attribution requirements
```

Licensing is especially important for generative music models.

---

# 135. Asset Licensing

Drum samples must also have a usable license.

Development placeholders can be temporary.

Before distribution, replace them with:

- owned samples,
- properly licensed samples,
- commissioned sounds.

Do not assume random internet WAV files can be redistributed.

---

# 136. Privacy

The initial offline architecture has a major privacy advantage.

Instrument audio does not need to leave the user's computer.

If cloud models are ever added, the application should clearly distinguish:

```text
local processing
vs
cloud processing
```

and explain when audio is transmitted.

---

# 137. Crash Resilience

A failure in the AI engine should not leave:

```text
stuck MIDI notes
locked audio devices
broken recording files
zombie worker threads
```

Always implement cleanup using explicit lifecycle management.

---

# 138. Developer Debug Mode

A debug screen can expose:

```text
Input device
Output device

Input RMS

Pitch:
82.4 Hz
E2
confidence 0.92

Tempo:
92.7
confidence 0.84
locked yes

Beat:
3 / 4
phase 0.61
bar 14

Intensity:
0.73

Drum pattern:
STONER_01

MIDI:
connected

AI:
running

Audio xruns:
0
```

This is much more valuable during development than sophisticated visual design.

---

# 139. User-Facing UI Should Hide Debug Complexity

The final normal view should remain simple.

Developer mode may contain diagnostics.

Normal mode should approximately be:

```text
Instrument connected

Ready

[JAM]

Drums
AI Guitar

Style

Follow / Improvise
```

Do not expose internal engineering terms unnecessarily.

---

# 140. First AI Agent Bootstrap Prompt

An AI coding agent starting work on this project should understand the following:

```text
You are working on JamMate, an experimental realtime AI music accompaniment
desktop application for macOS Apple Silicon.

The user connects a bass guitar through a USB audio interface. JamMate listens
to the live bass signal, detects played notes and rhythm, estimates tempo,
maintains a Musical State, plays synchronized drum accompaniment, and controls
a realtime generative music engine for additional accompaniment.

This is a realtime audio application, not a web backend.

Primary technology:

- Python 3.12
- uv
- macOS Apple Silicon
- PySide6 / Qt Widgets
- sounddevice / PortAudio
- 48 kHz audio
- 128 frame default buffer, 256 fallback
- NumPy / SciPy
- YIN-based monophonic bass pitch detection
- soundfile
- mido + python-rtmidi
- pytest
- ruff
- Magenta RealTime 2 as the initial generative music engine

Critical architecture rules:

1. Never run AI inference inside the PortAudio callback.
2. Never run heavy DSP inside the PortAudio callback.
3. Never perform file I/O inside the PortAudio callback.
4. Never perform logging inside the PortAudio callback.
5. Never update GUI widgets inside the PortAudio callback.
6. Never perform MIDI operations inside the PortAudio callback.
7. Avoid blocking locks in realtime audio code.
8. Use bounded/preallocated buffers where practical.
9. Run analysis outside the audio callback.
10. Keep model-specific integration behind an adapter.
11. Do not introduce databases, HTTP APIs, cloud infrastructure, Docker,
    Electron, React, or microservices unless explicitly requested.
12. Do not implement speculative future features.
13. Prefer simple, testable implementations.
14. Musical quality and realtime stability matter more than architectural
    sophistication.
15. Inspect the existing repository before changing architecture.

Before implementation:
- understand the current module boundaries,
- keep changes focused,
- state any realtime implications,
- implement,
- run tests,
- run ruff,
- report limitations.
```

---

# 141. Rules for Future AI Agents

When an agent encounters uncertainty:

1. Prefer the simplest solution compatible with realtime safety.
2. Do not redesign unrelated systems.
3. Preserve existing module boundaries unless there is clear technical justification.
4. Do not silently introduce background services.
5. Do not silently change sample rate or audio format.
6. Do not place convenience code in the audio callback.
7. Do not optimize prematurely.
8. Do not substitute an AI model for a deterministic algorithm without demonstrating value.
9. Treat real instrument testing as required for realtime features.
10. Maintain the distinction between raw audio, musical observations, Musical State, musical decisions, and rendering.

---

# 142. Core Mental Model

Every contributor should understand the following transformation:

```text
RAW AUDIO
   |
   v
MUSICAL OBSERVATIONS
   |
   v
MUSICAL STATE
   |
   v
MUSICAL DECISIONS
   |
   v
AUDIO / MIDI OUTPUT
```

More concretely:

```text
waveform
   |
   v
E2 played at t=4.21
strong onset
92 BPM
   |
   v
bar 5
beat 1
riff active
medium intensity
   |
   v
kick now
snare on beat 2
guitar should support E
   |
   v
samples + generated audio
```

This layered model is central to JamMate.

---

# 143. Ultimate Product Vision

The mature application should make a musician forget that they are interacting with software.

A bass player should be able to:

```text
plug in
start riffing
hear drummer join
hear guitarist respond
change riff
hear band transition
play harder
hear band intensify
stop briefly
hear band hold groove
return to old riff
hear band recognize it
record the jam
```

The user should feel:

> "These musicians are listening to me."

That feeling is the central product goal.

Everything else — AI models, DSP algorithms, MIDI, C++, Python, buffer sizes, and architecture — exists only to create that experience.

---

# 144. Post-MVP Phase Architecture & Extended Musical State

Once the MVP vertical slice (Phases 1–4) is validated, the Post-MVP roadmap (Phases 5–7) evolves JamMate from a reactive accompaniment loop into an attentive virtual band and standalone macOS studio tool.

All Post-MVP extensions preserve the core layered architecture (§142) and the immutable `MusicalState` snapshot contract (§38, §39). `MusicalState` expands cleanly with optional/defaulted fields (§40):

```python
@dataclass(frozen=True, slots=True)
class MusicalState:
    # --- MVP core (§36) ---
    frame_index: SampleFrame = 0
    timestamp_seconds: float = 0.0
    is_playing: bool = False
    silence_duration_seconds: float = 0.0
    current_note: MidiNote | None = None
    note_name: str | None = None
    pitch_frequency_hz: float | None = None
    pitch_confidence: float = 0.0
    last_onset_frame: SampleFrame | None = None
    onset_strength: float = 0.0
    onset_density_per_second: float = 0.0
    bpm: float | None = None
    tempo_confidence: float = 0.0
    tempo_locked: bool = False
    beat_index: int = 0
    bar_index: int = 0
    beat_phase: float = 0.0
    intensity: float = 0.0
    # --- Post-MVP extensions (§40) ---
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4
    estimated_key: str | None = None
    estimated_mode: Literal["minor", "major"] | None = None
    harmonic_confidence: float = 0.0
    bass_register: Literal["low", "mid", "high"] = "low"
    riff_id: str | None = None
    riff_iteration: int = 0
    riff_changed: bool = False
    section: Literal["listening", "intro", "groove", "variation", "fill", "breakdown"] = "listening"
```

---

# 145. Key & Tonal-Center Detection Specification

Single-note conditioning tells the AI what pitch is sounding right now, while key/tonal-center estimation (§90) gives the band harmonic memory over several bars.

## Algorithm (`analysis/key.py`)

- Maintain a rolling 12-bin pitch-class histogram (`C, C#, ..., B`) with exponential decay over a configurable musical window (`key_window_seconds = 8.0`).
- Accumulate evidence from stabilized `NoteObservation`s weighted by:
  - **note duration** (sustained notes carry more weight than brief chromatic passing tones);
  - **metric weight** (notes landing near beat 1 or downbeats receive a `1.5x` weight boost);
  - **phrase-ending weight** (the last stable note before a gap `>= 250 ms` receives a `1.3x` tonic/resolution boost);
  - **onset velocity / RMS**.
- Score all 12 pitch classes against minor/pentatonic/rock and major tonal profiles combined with a bass-root prior (bass riffs strongly emphasize the tonic on beat 1).
- Apply hysteresis (`key_switch_margin`, minimum `harmonic_confidence`) before switching `estimated_key` so transient chromatic notes never flip the key.
- Expose `reset()` and `manual_key` override support (§58).

---

# 146. Riff Fingerprinting & Session Memory Specification

To feel like real bandmates, JamMate must recognize when the player is repeating a riff, when they move to a B-section, and when they return to an earlier riff (§41, §42, §98).

## Bar-Quantized Riff Fingerprinting (`music/riff.py`)

- At each bar boundary from `BeatClock`, quantize the bar's `OnsetObservation`s and `NoteObservation`s onto a 16-step (or meter-adapted) subdivision grid:
  - `onset_mask`: 16-element float vector of normalized subdivision onset strengths;
  - `pitch_classes`: 16-element integer vector of relative pitch classes (`-1` for rest, `0..11` relative to bar root);
  - `root_pc`: dominant pitch class of the bar;
  - `mean_intensity`: average intensity across the bar.
- Support both 1-bar and 2-bar riff windows. Compare the latest window against recent windows using a weighted similarity metric combining rhythmic cosine similarity and pitch-contour agreement.
- State transitions:
  - When consecutive bars match above `riff_lock_similarity` (`>= 0.78`), lock or keep the active `riff_id` (e.g. `"riff_A"`) and increment `riff_iteration`.
  - When a new bar diverges below `riff_change_threshold` for `>= 1` full bar while playing, flag `riff_changed = True` on the boundary and either match an existing riff in `SessionMemory` or mint a new `riff_id` (`"riff_B"`, `"riff_C"`, ...).

## Session Memory (`music/riff.py`)

- Bounded in-memory table (`max_riffs = 16`) mapping `riff_id -> RiffMemoryEntry`:
  - canonical `RiffFingerprint`;
  - `preferred_drum_pattern_id` and `preferred_intensity_tier`;
  - `accent_mask` (16-step average onset accent profile for adaptive drumming);
  - `estimated_key` and `bass_register`;
  - `times_played` and `last_seen_bar`.
- When the player returns to a previously stored `riff_id`, `SessionMemory` surfaces the stored entry so the Band Director and Drummer can immediately recall the groove that worked for that riff.

---

# 147. Adaptive Drummer & Deterministic Humanization Specification

## Adaptive Accent Locking (`drums/adaptive.py`)

- Expand on §93: track an exponentially weighted 16-step onset accent grid (`AccentProfile`) across the last 2–4 bars.
- When a subdivision step has high accent consistency (`accent_score >= accent_lock_threshold`):
  - reinforce that step with a synchronized kick hit (respecting a minimum refractory gap so the kick never machine-guns);
  - boost hi-hat / ride velocity on accented steps;
  - optionally thin unaccented syncopated kicks from the base pattern so the kick drum locks tightly to the bassist's pocket.
- All adaptations produce standard `PatternStep` events ahead of time in `DrumEngine` — the audio callback remains unchanged.

## Deterministic Humanization (`drums/humanize.py`)

- Expand on §95: pure, deterministic function seeded by `(bar_index, step_index, drum_voice)` so offline renders and unit tests are 100% reproducible:
  - **Microtiming:** subtle sample-frame offset within `±humanize_timing_ms` (default `±4.0 ms`), with `0` offset on beat-1 downbeats so the bar anchor never drifts relative to `BeatClock`.
  - **Velocity contour:** metric accent shaping (downbeat > backbeat > 8th > 16th) plus subtle per-hit variation (`±0.06`) and ghost-note snare scaling (`0.22–0.32` velocity on selected 16th subdivisions at `mid`/`high` intensity).

---

# 148. Band Director Specification

The `BandDirector` (§72, §73) is the decision layer (§125) between `MusicalState` / `SessionMemory` and the rendering engines (`DrumEngine`, `AccompanimentEngine`).

## Responsibilities & Outputs (`music/director.py`)

- Consumes `MusicalState`, `SessionMemory`, active `StylePreset`, and user `follow_improvise`.
- Evaluates on beat/bar boundaries and publishes an immutable `BandDirective`:
  - `section`: `"listening" | "intro" | "groove" | "variation" | "fill" | "breakdown"`;
  - `drum_role`: `"silent" | "groove" | "accent_lock" | "fill" | "crash_transition" | "hold_sparse"`;
  - `recalled_pattern_id: str | None` (from `SessionMemory` when a known riff returns);
  - `ai_role`: `"silent" | "tight_root" | "rhythm_support" | "open_space" | "lead_fill"`;
  - `effective_follow_improvise: float` (nudged toward `Follow` on new riffs and toward `Improvise` on long repeated riffs or high-register solo space);
  - `target_key_pc: int | None` (biases AI pianoroll conditioning with tonic stability when `harmonic_confidence` is high).
- Key musical behaviors:
  1. **Riff transition response:** when `riff_changed` fires, schedule a short drum fill or crash transition on the bar boundary and temporarily tighten AI conditioning (`tight_root`).
  2. **Riff recall:** when `riff_id` matches an earlier riff in `SessionMemory`, restore its `preferred_drum_pattern_id` and accent profile.
  3. **Register awareness:** when `bass_register == "high"` (e.g. MIDI `>= 55` for bass or `>= 64` for guitar), switch `ai_role` to `"open_space"` (reducing AI density/gain slightly) so the player has room to solo.
  4. **Phrase breathing:** after every `4` or `8` iterations of the same riff, trigger a 1-bar `"variation"` or `"fill"` directive before returning to `"groove"`.

---

# 149. Manual Escape Hatches & Multi-Meter Specification

Realtime estimators can occasionally misinterpret sparse or polyrhythmic playing. Manual escape hatches (§58) let the musician take immediate control without stopping the jam.

## Controls & Engine Behavior (`music/overrides.py`)

- **Auto BPM vs Manual BPM:** toggling Auto BPM off freezes `BeatClock` at the user's chosen BPM while keeping phase locked.
- **Tap Tempo:** `TapTempoTracker` records monotonic tap timestamps (`3..8` taps, resetting after `2.5 s` idle), computes the robust mean inter-tap interval, clamps to `tempo_min_bpm..tempo_max_bpm`, sets manual BPM, and aligns `BeatClock` beat-1 phase to the latest tap.
- **Time Signature / Meter (`4/4`, `3/4`, `6/8`):**
  - `BeatClock` generalizes `beats_per_bar` (`4` for `4/4`, `3` for `3/4`, `6` for `6/8`).
  - `drums/patterns.py` includes meter-aware patterns (`16` steps for `4/4`, `12` steps for `3/4` and `6/8`).
- **Key Override:** dropdown (`Auto`, `E`, `F`, `F#`, `G`, `G#`, `A`, `A#`, `B`, `C`, `C#`, `D`, `D#` × `minor`/`major`) that pins `estimated_key` and `harmonic_confidence = 1.0`.
- **Reset Musical Context:** single button / CLI action that clears tempo history, BeatClock lock, key histogram, riff session memory, and active notes without closing the audio stream.

---

# 150. Multi-Track Stem Recording & New Input Modalities

## Stem Recording (`audio/recorder.py`)

- Expand on §59: support both single-file master recording (`master.wav`) and multi-track stem recording (`recordings/YYYY-MM-DD_HH-MM-SS-jam/` containing frame-aligned `bass.wav`, `drums.wav`, `ai_guitar.wav`, and `master.wav`).
- Realtime safety:
  - `BlockProcessor` copies the pre-mix buses (mono input monitor, stereo drum bus, stereo AI bus, stereo master bus) into a single preallocated multi-channel `RingBuffer` (`7` channels per frame) using allocation-free slice copies.
  - The background writer thread de-interleaves chunks and writes the 4 WAV files via `soundfile.SoundFile` so all stems have the exact same sample frame count and zero phase offset.

## MIDI Keyboard Input (`midi/input.py`)

- Expand on §55: optional `MidiInputWorker` using `mido` / `python-rtmidi`.
- Listens to a selected MIDI input port off the audio thread, translates `note_on` / `note_off` messages into `NoteObservation` and `OnsetObservation` events stamped with the current stream `frame_index`, and extracts the lowest active note (or root of a played triad) for `MusicalState` and `NoteTracker`.
- Bypasses YIN pitch detection latency completely while driving the exact same downstream `BeatClock`, `MusicalStateManager`, `DrumEngine`, and `AccompanimentEngine`.

## Polyphonic Guitar & Power-Chord Input (`analysis/guitar.py`)

- Expand on §53, §54, and D7: electric guitarists frequently play 2-note and 3-note power chords (`root + fifth`, `root + fifth + octave`) or double-stops where monophonic YIN may lose confidence or jump to a harmonic.
- Add a lightweight streaming harmonic-product / spectral-root estimator (`GuitarChordAnalyzer`) running in `AnalysisWorker` when `instrument == "guitar"`:
  - When YIN is confident (`>= confidence_threshold`), use YIN directly for fast monophonic single-note tracking.
  - When an onset/active signal has low YIN periodicity (characteristic of dyads/power chords or heavy overdrive), compute a Windowed Harmonic Product Spectrum / 12-bin chroma over the guitar fundamental range (`75–660 Hz`), identify power-chord `(f0, 1.5*f0, 2*f0)` pairs, and emit a stabilized root `NoteObservation` so the drummer and AI stay locked to the guitarist's root movement.

---

# 151. Native C++ Realtime Bridge & Standalone macOS Packaging

## Native C++ Audio/DSP Bridge (`src/jammate/native/`)

- Expand on §19, §87, §89: isolate hot inner loops behind a compiled C++20 / `nanobind` extension (`_jammate_native`) with a seamless pure-Python fallback if the extension is not compiled:
  - Lock-free single-producer/single-consumer audio/event ring buffers and sample-accurate polyphonic drum voice mixer (`mix_voices_into_stereo`) that execute with zero Python object traffic.
  - Vectorized cumulative mean normalized difference function (CMNDF) for YIN pitch detection, cutting analysis hop CPU time by `4–8x` and releasing the GIL during DSP computation.
  - Native `AccompanimentEngine` C-ABI / shared-memory ring adapter so an external or C++ inference runner can feed stereo float32 frames directly into the audio engine.

## Standalone macOS Application Packaging (`packaging/`)

- Expand on §86: package JamMate as a standalone Apple Silicon (`arm64`) `JamMate.app` bundle:
  - Build specification (`packaging/jammate.spec` / `scripts/build_macos_app.py`) bundling Python, PySide6, PortAudio, and `assets/drums/`.
  - `packaging/Info.plist` declaring `NSMicrophoneUsageDescription` (required by macOS CoreAudio for USB audio interface input), bundle identifier `com.jammate.app`, and high-DPI retina support.
  - Model asset manager (`ai/model_manager.py` + UI status dialog) that checks `~/Documents/Magenta/magenta-rt-v2` on startup, verifies model weight checksums/files, and guides or triggers background initialization if missing.
  - Ad-hoc and Developer ID signing + DMG creation script (`scripts/package_dmg.sh`).

---

# 152. Fifth Development Milestone

Milestone:

```text
Attentive Band (Musical Intelligence & Band Director)
```

Definition of done:

```text
JamMate estimates key/tonal center over a rolling window.

Riff detector recognizes repeating 1- and 2-bar riffs and detects transitions.

Session memory recalls the groove when the player returns to an earlier riff.

Drummer locks kicks to consistent bass accents and applies subtle humanization.

Band Director coordinates fills on transitions and opens space for high-register solos.
```

---

# 153. Sixth Development Milestone

Milestone:

```text
Studio & Multi-Instrument
```

Definition of done:

```text
User can tap tempo, lock/override BPM and key, switch meter (4/4, 3/4, 6/8), and reset context live.

Multi-track stem recording writes sample-aligned bass.wav, drums.wav, ai_guitar.wav, and master.wav.

MIDI keyboard input drives notes, onsets, BeatClock, drums, and AI with zero pitch latency.

Guitar input handles both single-note riffs and power chords without losing root tracking.
```

---

# 154. Seventh Development Milestone

Milestone:

```text
Standalone macOS Release
```

Definition of done:

```text
Compiled C++ native extension accelerates YIN CMNDF and voice mixing with a clean Python fallback.

JamMate builds into a standalone Apple Silicon JamMate.app bundle with microphone permissions.

Startup verifies MRT2 model assets and degrades cleanly if weights are absent.

Full automated test suite, lint, format, and packaged app smoke checks pass.
```
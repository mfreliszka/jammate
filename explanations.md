# JamMate — Feature & Packaging Guide (`explanations.md`)

This document explains:
1. **Why the `MIDI In` selector exists** (`None (Audio only)` by default) and when you would use it.
2. **What the `Stems` checkbox does** when recording a jam.
3. **How to build the macOS `JamMate.app` bundle** step by step.
4. **How your friend can install and run JamMate on his personal MacBook Pro** (what works when sending the `.app` directly, what doesn't, and the exact steps to set him up).

---

## 1. The `MIDI In` Selector (`None (Audio only)`) — Why It Exists & When to Use It

### Why does it say `None (Audio only)` by default?
When you play **electric guitar** or **bass** plugged into your USB audio interface (`USB Audio CODEC`), JamMate listens to the **raw analog/digital audio waveform** from your instrument cable. It runs real-time Digital Signal Processing (DSP) on that audio to detect:
- When you pluck a string (**Onset detection**);
- Which note or power-chord/chord root you are playing (**YIN pitch + harmonic root analysis**);
- How hard you are playing (**RMS intensity**);
- Your **tempo (BPM)** and **musical key**.

In that normal guitar/bass setup, you are **not** using a MIDI device at all—so **`None (Audio only)` is the right setting for 100% of normal guitar or bass jamming.**

---

### Why do we have `MIDI In` at all?
The `MIDI In` selector (implemented in [`src/jammate/midi/input.py`](src/jammate/midi/input.py)) lets JamMate accept **live MIDI note events** (`note_on`, `note_off`, and key velocity) from any USB or virtual MIDI controller **in addition to (or instead of) audio input**.

Instead of estimating pitch from an audio waveform with DSP, `MidiInputWorker` translates incoming MIDI messages directly into the exact same `MusicalEvent` stream (`OnsetEvent`, `PitchEvent`, `IntensityUpdate`, `TempoUpdate`, `KeyUpdate`) that the guitar audio analyzer produces.

### In which situations would you select something in `MIDI In`?

1. **Jamming without a guitar in hand (e.g., on the couch or traveling with a small USB MIDI keyboard)**
   - If you don't have your guitar or audio interface plugged in, you can connect any small USB MIDI keyboard (like an Akai MPK Mini, Arturia MiniLab, Novation Launchkey, etc.) or a virtual macOS IAC MIDI port.
   - When you play basslines, riffs, or chords on the keys, JamMate's **BeatClock**, **Drummer**, **Band Director**, and **Magenta RT2 AI** lock onto your MIDI timing, velocity, and chords just as if you were playing guitar.

2. **Feeding exact chord changes / harmonic roots to the AI while you solo on guitar**
   - Audio chord recognition on a heavily distorted or fast-soloing electric guitar is naturally harder than reading discrete MIDI keys.
   - You can plug your **guitar into the audio interface** *and* have a **MIDI foot controller, synth pad, or keyboard** selected in `MIDI In`. You (or a friend in the room) can tap or hold a chord on the MIDI controller (e.g. `E5` power chord, `A minor`, `C major`) to give the Band Director and Magenta RT2 100% rock-solid chord/root conditioning while you play lead guitar over the top.

3. **Using a MIDI Guitar pickup or guitar-to-MIDI converter**
   - If you use hardware/software like a Boss GM-800, Roland GK pickup, Fishman TriplePlay, or JamOrigin MIDI Guitar 2 (which exposes a virtual MIDI port on macOS), selecting that port in `MIDI In` gives JamMate zero-latency polyphonic MIDI note & chord detection alongside your guitar audio.

4. **Routing a chord track or bass loop from a DAW (Logic Pro / Ableton) via macOS IAC Driver**
   - macOS has a built-in virtual MIDI cable called **IAC Driver** (enabled in *Audio MIDI Setup.app*). You can send MIDI notes from Logic Pro or Ableton into JamMate's `MIDI In` to test how the drummer and AI react to specific riffs or progressions.

> **Summary:** Leave **`MIDI In`** on **`None (Audio only)`** whenever you are just plugging in your electric guitar or bass and jamming. Pick a port from the dropdown only when you connect a USB MIDI keyboard, MIDI foot pedal, or virtual MIDI source.

---

## 2. The `Stems` Checkbox — What It Is & What It Does

Next to the **`Record jam to WAV`** checkbox in the **Record** panel, there is a **`Stems`** checkbox (`record_stems` in [`src/jammate/audio/recorder.py`](src/jammate/audio/recorder.py)).

### Mode 1: `Record jam to WAV` checked, `Stems` **UNCHECKED** (Default)
When `Stems` is unchecked, [`MasterRecorder`](src/jammate/audio/recorder.py) records a **single stereo WAV file** containing everything mixed together (your guitar + drums + AI accompaniment), exactly as you heard it in your headphones/speakers:
```text
recordings/
  jam_20261014_120000.wav      <-- 48 kHz, 24-bit stereo master mix
```
This is great for quickly listening back to an idea or sharing a voice-memo-style demo.

### Mode 2: `Record jam to WAV` checked, `Stems` **CHECKED**
In studio production, a **"stem"** is an isolated audio track for an individual instrument so you can remix, re-amp, or edit each part separately later.

When you check **`Stems`** before pressing **Start Jam**, JamMate switches from `MasterRecorder` to [`StemRecorder`](src/jammate/audio/recorder.py). Instead of one mixed file, it creates a **timestamped session folder** inside `recordings/` containing **4 separate, sample-accurate WAV files**:

```text
recordings/
  jam_20261014_120000/
    bass.wav        <-- 48 kHz, 24-bit MONO:   Your isolated instrument (guitar or bass) bus
    drums.wav       <-- 48 kHz, 24-bit STEREO: The isolated drum sampler bus
    ai_guitar.wav   <-- 48 kHz, 24-bit STEREO: The isolated Magenta RT2 AI accompaniment bus
    master.wav      <-- 48 kHz, 24-bit STEREO: The full stereo master mix (with soft-clipping)
```
*(Note: `bass.wav` is the filename used by convention from the spec for your live instrument input—whether you selected Guitar or Bass).*

### Why is `Stems` useful?
1. **Drag-and-drop into Logic Pro, GarageBand, Reaper, or Ableton:**
   All 4 WAV files start at the **exact same sample frame (`frame 0`)** and have the **exact same length down to the single sample**. If you drag `bass.wav`, `drums.wav`, and `ai_guitar.wav` onto 3 tracks in Logic Pro or GarageBand, they line up in perfect phase synchronization.
2. **Change volumes or mute the AI later:**
   Maybe you played an amazing guitar riff, the drums were great, but the AI band played something weird in bar 8. With stems, you don't lose the take—you can mute `ai_guitar.wav` in bar 8 or adjust the drum volume after the fact.
3. **Add amp plugins / effects to your guitar later:**
   Because `bass.wav` is your isolated guitar/bass signal (independent of the drums and AI), you can put a different amp simulator, reverb, or EQ on your guitar track in your DAW without affecting the drums.
4. **Zero impact on realtime audio latency:**
   Inside the audio callback ([`BlockProcessor.process`](src/jammate/audio/engine.py)), JamMate copies the 4 buses into a preallocated 7-channel lock-free ring buffer (`1 mono + 2 drum + 2 AI + 2 master = 7 floats per frame`) with zero memory allocations. A background writer thread (`jammate-stem-recorder`) drains the ring buffer and writes the 4 WAV files to disk safely off the audio thread.

---

## 3. How to Build the macOS `.app` Bundle (`JamMate.app`)

JamMate includes a packaging script ([`scripts/build_macos_app.py`](scripts/build_macos_app.py)) and `Makefile` targets that build a double-clickable macOS application bundle at **`dist/JamMate.app`**.

### Step-by-step on your Mac

#### Step 1: Verify the packaging configuration and assets (optional check)
From the project root (`/Users/mfreliszka/Dev/jammate`), run:
```bash
make check-app
# or directly:
uv run python scripts/build_macos_app.py --check
```
This verifies:
- [`packaging/Info.plist`](packaging/Info.plist) (`CFBundleIdentifier = com.jammate.app`, `arm64` architecture priority, Retina support, and `NSMicrophoneUsageDescription` so macOS prompts for microphone/USB audio interface permission);
- [`packaging/entitlements.plist`](packaging/entitlements.plist) (`com.apple.security.device.audio-input` and `com.apple.security.device.microphone`);
- All 6 drum kit WAV files in `assets/drums/` (`kick.wav`, `snare.wav`, `hihat_closed.wav`, `hihat_open.wav`, `crash.wav`, `tom.wav`);
- The C++20 native extension source [`src/jammate/native/_jammate_native.cpp`](src/jammate/native/_jammate_native.cpp).

#### Step 2: Build `dist/JamMate.app`
Run:
```bash
make app
# or directly:
uv run python scripts/build_macos_app.py --output-dir dist
```

### What does `make app` do under the hood?
1. **Compiles the C++20 native extension** (`_jammate_native.dylib`) using `clang++ -std=c++20 -O3 -ffast-math` if it isn't compiled yet.
2. **Creates the standard macOS bundle hierarchy** under `dist/JamMate.app`:
   ```text
   dist/JamMate.app/
     Contents/
       Info.plist                         <-- macOS bundle metadata + Mic permission prompt text
       PkgInfo                            <-- APPL????
       MacOS/
         JamMate                          <-- Executable launcher script
       Resources/
         entitlements.plist               <-- Audio/Mic entitlements
         assets/drums/*.wav               <-- Bundled drum kit samples
         jammate/                         <-- Full JamMate Python package + _jammate_native.dylib
   ```
3. **Signs the bundle with macOS `codesign` (ad-hoc signature)** attaching the audio/microphone entitlements (`com.apple.security.device.audio-input`) so macOS CoreAudio allows `JamMate.app` to read from your USB audio interface.
4. **Verifies the built bundle** (`verify_built_bundle`) to confirm every file and permission bit is valid.

#### Step 3: Launch `JamMate.app` on your Mac
Once `make app` finishes, you can launch it from Finder (double-click `dist/JamMate.app`), drag it to your Dock, or run from Terminal:
```bash
open dist/JamMate.app
```

---

## 4. Can I Send `JamMate.app` to a Friend for His Personal MacBook Pro?

### The Short Answer
**Yes, your friend can run JamMate on his MacBook Pro** (as long as it is an **Apple Silicon MacBook Pro — M1, M2, M3, or M4**), **BUT you cannot just send him `dist/JamMate.app` by itself over AirDrop/email right now.**

Here is why, and the **two ways** to give it to your friend.

---

### Why can't I just email/AirDrop `dist/JamMate.app` alone right now?

There are **3 things** `JamMate.app` depends on that live outside the `.app` folder:

1. **The Python Virtual Environment (`.venv`) path in the launcher:**
   Look at `dist/JamMate.app/Contents/MacOS/JamMate` created by `scripts/build_macos_app.py`:
   ```bash
   exec "/Users/mfreliszka/Dev/jammate/.venv/bin/python" -m jammate.main "$@"
   ```
   To keep local builds fast (~1 second instead of copying 2 GB of Python/PySide6/MLX/JAX libraries every time), `make app` creates a **local developer bundle** whose launcher points to your machine's `.venv/bin/python`. On your friend's MacBook Pro, `/Users/mfreliszka/...` does not exist!

2. **The Magenta RealTime 2 AI Model Weights (~3–5 GB):**
   If your friend checks the **`Enable AI Band`** box, JamMate looks for the downloaded Magenta RT2 checkpoints inside:
   ```text
   ~/Documents/Magenta/magenta-rt-v2/
   ```
   (`checkpoints/mrt2_small`, `checkpoints/musiccoca_base214_v2`, and `checkpoints/spectrostream_48khz`). Those multi-gigabyte AI weights are not stored inside the `.app` bundle. *(Note: Without the AI weights, JamMate still runs in **Drums + Monitoring** mode cleanly!)*

3. **macOS Gatekeeper Quarantine:**
   Because `JamMate.app` is signed with a free local **ad-hoc signature** (`codesign --sign -`) rather than a paid $99/year Apple Developer ID certificate notarized by Apple, when your friend downloads or AirDrops a `.app` from another computer, macOS Gatekeeper will pop up:
   > *"JamMate.app is damaged and can't be opened"* or *"Apple could not verify JamMate.app is free of malware."*

---

### How to Set Up Your Friend's MacBook Pro (2 Options)

#### Option A (Recommended & Most Reliable — Takes ~5 Minutes on His Mac)
Instead of sending only the `.app` file, send him the project folder (as a `.zip` or via Git) and have him run **4 commands** in his Terminal. This installs the exact Apple Silicon Python environment, downloads the AI weights, and builds `JamMate.app` natively on his MacBook Pro:

1. **Install `uv` (if he doesn't have it):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Unzip the `jammate` folder and install dependencies:**
   ```bash
   cd ~/Downloads/jammate
   uv sync
   ```
3. **Download the Magenta RT2 AI model weights (only needed once for AI accompaniment):**
   ```bash
   uv run mrt models init
   uv run mrt models download
   ```
   *(He can verify the models are ready with `uv run jammate --check-models`.)*
4. **Build his own native `JamMate.app` and move it to `/Applications`:**
   ```bash
   make app
   cp -R dist/JamMate.app /Applications/
   ```
   Now he can double-click **JamMate** from `/Applications` or Launchpad anytime!

---

#### Option B (If You Want to Send Him a Single Standalone `JamMate.app` File Without Him Running `uv sync`)
If your friend is not a developer and you want to hand him a self-contained `JamMate.app` that already has Python, PySide6, sounddevice, NumPy, SciPy, and MLX baked inside the `.app` bundle:

1. **Freeze the Python runtime into the `.app` using PyInstaller on your Mac:**
   ```bash
   uv add --dev pyinstaller
   uv run pyinstaller \
     --noconfirm \
     --windowed \
     --name JamMate \
     --osx-bundle-identifier com.jammate.app \
     --add-data "assets/drums:assets/drums" \
     --add-binary "src/jammate/native/_jammate_native.dylib:jammate/native" \
     src/jammate/main.py
   ```
   Then re-sign the resulting `dist/JamMate.app` with the microphone entitlements:
   ```bash
   codesign --force --deep --sign - --entitlements packaging/entitlements.plist dist/JamMate.app
   ```

2. **What your friend must do on his MacBook Pro after receiving `JamMate.app`:**
   - **Step 1 — Move `JamMate.app` to `/Applications`**
   - **Step 2 — Clear the macOS Gatekeeper quarantine flag** (required once because the app is ad-hoc signed rather than Apple-notarized):
     Open Terminal on his Mac and run:
     ```bash
     xattr -cr /Applications/JamMate.app
     ```
   - **Step 3 — Copy the AI Model Folder (if he wants AI Band enabled):**
     If he wants **Enable AI Band** to work (not just Drums + Monitor), copy your `~/Documents/Magenta/magenta-rt-v2` folder onto a USB drive (or zip it) and place it on his Mac at:
     ```text
     /Users/<his-username>/Documents/Magenta/magenta-rt-v2
     ```
   - **Step 4 — Open `JamMate.app` and click "Allow" when macOS asks for Microphone / Audio Interface access.**

---

### Quick Checklist for Your Friend's MacBook Pro

| Requirement | Needed? | Notes |
|---|---|---|
| **Apple Silicon (M1 / M2 / M3 / M4)** | **Yes** | Required for MLX GPU acceleration and the `arm64` native DSP library. |
| **Audio Interface or Built-in Mic/Headphones** | **Yes** | Select his audio interface in the **Input** and **Output** dropdowns at the top of the JamMate window. |
| **Microphone Permission in macOS** | **Yes** | *System Settings → Privacy & Security → Microphone → JamMate* (macOS prompts on first launch). |
| **`~/Documents/Magenta/magenta-rt-v2`** | **Only for AI Band** | Without this folder, **Monitor + Smart Drummer** still work 100%; adding this folder enables the **Magenta RT2 AI Band**. |

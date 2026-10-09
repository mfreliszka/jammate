# tests/data — real instrument recordings

Short recordings of the developer's electric guitar (D7). They are used to check analysis
against real playing (Steps 2.2–2.4) and to render jams offline (Step 3.5). Synthetic signals
live in `tests/signals.py`.

## Format

- Mono WAV, 48 kHz, PCM_24: the default output of `scripts/capture_input.py`.
- 10–20 s each.
- Clean DI signal only:
  - no distortion pedals or amp simulation;
  - no metronome or other bleed.
- Single notes, with the strings you aren't playing muted. The only exception is a file whose
  name says `chords`.
- Normal playing peaks at about -10 to -6 dBFS, with no `CLIP`.
- File name: `<instrument>_<content>[_<bpm>bpm].wav`. Leave out `_<bpm>bpm` when there is no
  tempo.

Record with, for example:

```bash
make capture FILE=tests/data/guitar_riff_e2_g2_a2_92bpm.wav DURATION=20
```

## Fixtures

Each row is the ground truth for one file: what was actually played. Notes are sounding pitch
(D6). A guitar's open strings are E2 A2 D3 G3 B3 E4; its 12th fret is E3 A3 D4 G4 B4 E5.

| File | Played | BPM | Used by |
|---|---|---|---|
| `guitar_open_strings.wav` | Open strings, **high to low**, ~2 s each with ~1 s gaps. Starts: E4 1.52 s, B3 4.75 s, G3 7.61 s, D3 10.76 s, A2 13.86 s, E2 16.70 s. | – | 2.2 |
| `guitar_12th_fret.wav` | 12th fret, **high to low**, ~2 s each. Starts: E5 0.60 s, B4 3.35 s, G4 6.70 s, D4 10.26 s, A3 13.54 s, E3 17.20 s. D4 is held about 30 cents sharp and starts more than 50 cents sharp. | – | 2.2 |
| `guitar_riff_e2_g2_a2_92bpm.wav` | Low E string, 8th notes: E2 ×8, G2 ×8, A2 ×8, played twice. That's 48 notes from 2.68 s to 18.05 s, then muted. Handling noise before 2.5 s. The first note splits into two onsets 70 ms apart. The last G of each cycle slides up to A2 about 225 ms after it is picked, which makes a ghost onset at 7.83 s and 15.68 s. | 92 (measured 91.9) | 2.2–2.4, 3.5 |
| `guitar_riff_e2_g2_a2_120bpm.wav` | Same riff: E2 ×8, G2 ×8, A2 ×8, played twice. That's 48 notes from 1.23 s to 12.98 s, then silence. Handling noise before 1.2 s. The last G of each cycle slides to A2, starting about 125 ms after the pick and arriving by about 225 ms, so it reads G#2 in passing. Several picks give two onsets 35 ms apart: the pick contact, then the note. | 120 (measured 119.2) | 2.2–2.4, 3.5 |
| `guitar_chords_e_g_a_100bpm.wav` | Strummed **open major chords** (not power chords), 8th notes: E ×8, G ×8, A ×8, played twice. Sections start at about E 1.2 s, G 3.6 s, A 5.9 s, E 8.3 s, G 10.8 s and A 13.1 s. The take stops mid-chord at 15.07 s, so the last A section has about 7 chords. Chroma confirms E = E G# B, G = G B D, A = A C# E. | 100 (measured 99.8) | 2.2 (no wrong notes), 2.3, 2.4 |

Start times come from a level threshold and are accurate to about ±20 ms.

What every file contains besides the playing:
- **Exact digital zeros at both ends**: 512 frames at the start and 3,328 at the end. These are
  stream start and close artifacts, harmless silence.
- **Unmuted open strings** ringing at about -49 to -55 dBFS between notes and before playing.
  They have real pitch content, unlike hum.
- **The electrical noise floor** at about -62 dBFS.

## To record (optional)

Step 2.2's required fixtures are all recorded. This one would also measure the power-chord
octave trap (D7):

```bash
make capture FILE=tests/data/guitar_chords_power_e5_g5_a5_100bpm.wav DURATION=17  # power chords E5 G5 A5, 8th notes
```

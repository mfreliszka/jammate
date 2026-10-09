"""Synthetic audio for tests: plucked bass or guitar notes, riffs with known timing, silence, noise.

All generators are deterministic and return float32 mono arrays (48 kHz by default). Riffs come
with exact ground truth (onset frames and MIDI notes), so analysis can be tested without a real
instrument (§65-§67).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jammate.analysis.notes import midi_to_hz, name_to_midi
from jammate.music.events import MidiNote, SampleFrame, seconds_to_frames

SAMPLE_RATE_HZ = 48_000
# Relative amplitudes of harmonics 1, 2, 3, … of a plucked string (fundamental first).
BASS_HARMONIC_AMPLITUDES = (1.0, 0.6, 0.35, 0.2, 0.12, 0.07)
# Electric guitar (D7): brighter, and at the attack the 2nd harmonic outweighs the fundamental, as
# is common on the low strings. Right after an onset the spectral peak is an octave above the
# note (an octave-error trap for pitch); the fundamental takes over as the harmonics decay.
GUITAR_HARMONIC_AMPLITUDES = (0.7, 1.0, 0.8, 0.55, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07)
DEFAULT_DECAY_SECONDS = 1.0
ATTACK_SECONDS = 0.002  # short enough for a sharp onset, long enough to avoid a click
RELEASE_SECONDS = 0.005  # fade at the end of every note, so cut-off notes don't click
# Idle noise floor measured on the developer's interface in Step 1.4.
INTERFACE_NOISE_FLOOR_DBFS = -65.0

# A riff step: a note (MIDI number or name), or None for a rest. On its own it lasts
# `step_beats`; as a `(note, beats)` tuple it lasts `beats`.
type RiffNote = MidiNote | str | None
type RiffStep = RiffNote | tuple[RiffNote, float]


@dataclass(frozen=True, eq=False)
class Riff:
    """A rendered riff plus its ground truth. `onset_frames`, `end_frames` and `midi_notes`
    have one entry per sounded note (rests excluded), in order."""

    samples: np.ndarray
    sample_rate_hz: int
    bpm: float
    onset_frames: tuple[SampleFrame, ...]
    end_frames: tuple[SampleFrame, ...]
    midi_notes: tuple[MidiNote, ...]


def plucked_note(
    note: MidiNote | str,
    duration_seconds: float,
    *,
    cents: float = 0.0,
    amplitude: float = 0.5,
    decay_seconds: float = DEFAULT_DECAY_SECONDS,
    harmonic_amplitudes: Sequence[float] = BASS_HARMONIC_AMPLITUDES,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """A plucked-string tone: fundamental + harmonics, each decaying exponentially.

    Harmonic `k` decays `k` times faster than the fundamental, like a real string. The default
    timbre is a bass; pass `harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES` for a guitar, or any
    weak fundamental such as `(0.3, 1.0, 0.5)` to provoke octave errors. `cents` detunes the note.
    The peak equals `amplitude`.
    """
    frame_count = seconds_to_frames(duration_seconds, sample_rate_hz)
    return _pluck(
        _midi_note(note),
        frame_count,
        cents=cents,
        amplitude=amplitude,
        decay_seconds=decay_seconds,
        harmonic_amplitudes=harmonic_amplitudes,
        sample_rate_hz=sample_rate_hz,
    )


def riff(
    steps: Sequence[RiffStep],
    bpm: float,
    *,
    step_beats: float = 1.0,
    repeats: int = 1,
    gate: float = 0.9,
    lead_in_seconds: float = 0.0,
    tail_seconds: float = 0.0,
    amplitude: float = 0.5,
    decay_seconds: float = DEFAULT_DECAY_SECONDS,
    harmonic_amplitudes: Sequence[float] = BASS_HARMONIC_AMPLITUDES,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> Riff:
    """Render `steps` (repeated `repeats` times) at `bpm` as plucked notes.

    Each note sounds for `gate` times its step length, then silence until the next step. Onsets
    sit at exact beat positions (rounded to the nearest frame, without accumulating rounding
    drift), after `lead_in_seconds` of silence; `tail_seconds` of silence follows the last step.

    Example: `riff(["E1", "E1", ("G1", 0.5), ("A1", 1.5), None], bpm=92, step_beats=1.0)`.
    Pass `harmonic_amplitudes=GUITAR_HARMONIC_AMPLITUDES` for a guitar riff.
    """
    if not (bpm > 0 and repeats >= 1 and 0 < gate <= 1):
        raise ValueError("bpm must be positive, repeats >= 1 and gate in (0, 1]")
    frames_per_beat = 60.0 * sample_rate_hz / bpm
    lead_in_frames = seconds_to_frames(lead_in_seconds, sample_rate_hz)

    def beat_to_frame(beat: float) -> SampleFrame:
        return lead_in_frames + math.floor(beat * frames_per_beat + 0.5)

    notes: list[tuple[MidiNote, SampleFrame, SampleFrame]] = []
    beat = 0.0
    for step in list(steps) * repeats:
        note, beats = step if isinstance(step, tuple) else (step, step_beats)
        if beats <= 0:
            raise ValueError(f"a step must last more than 0 beats, got {beats}")
        start_frame, step_end_frame = beat_to_frame(beat), beat_to_frame(beat + beats)
        if note is not None:
            sounding_frames = max(1, math.floor(gate * (step_end_frame - start_frame) + 0.5))
            notes.append((_midi_note(note), start_frame, start_frame + sounding_frames))
        beat += beats

    total_frames = beat_to_frame(beat) + seconds_to_frames(tail_seconds, sample_rate_hz)
    samples = np.zeros(total_frames, dtype=np.float32)
    for midi_note, start_frame, end_frame in notes:
        samples[start_frame:end_frame] = _pluck(
            midi_note,
            end_frame - start_frame,
            amplitude=amplitude,
            decay_seconds=decay_seconds,
            harmonic_amplitudes=harmonic_amplitudes,
            sample_rate_hz=sample_rate_hz,
        )
    return Riff(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        bpm=bpm,
        onset_frames=tuple(start for _, start, _ in notes),
        end_frames=tuple(end for _, _, end in notes),
        midi_notes=tuple(midi_note for midi_note, _, _ in notes),
    )


def silence(duration_seconds: float, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.zeros(seconds_to_frames(duration_seconds, sample_rate_hz), dtype=np.float32)


def noise(
    duration_seconds: float,
    *,
    rms_dbfs: float = INTERFACE_NOISE_FLOOR_DBFS,
    seed: int = 0,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """White Gaussian noise with exactly the given RMS level. The same `seed` gives the same
    samples."""
    frame_count = seconds_to_frames(duration_seconds, sample_rate_hz)
    samples = np.random.default_rng(seed).standard_normal(frame_count)
    if frame_count:
        samples *= 10.0 ** (rms_dbfs / 20.0) / np.sqrt(np.mean(samples**2))
    return samples.astype(np.float32)


def _midi_note(note: MidiNote | str) -> MidiNote:
    return name_to_midi(note) if isinstance(note, str) else note


def _pluck(
    midi_note: MidiNote,
    frame_count: int,
    *,
    amplitude: float,
    decay_seconds: float,
    sample_rate_hz: int,
    cents: float = 0.0,
    harmonic_amplitudes: Sequence[float] = BASS_HARMONIC_AMPLITUDES,
) -> np.ndarray:
    if not (0 < amplitude <= 1 and decay_seconds > 0 and frame_count >= 0):
        raise ValueError("amplitude must be in (0, 1], decay_seconds > 0 and duration >= 0")
    fundamental_hz = midi_to_hz(midi_note) * 2.0 ** (cents / 1200.0)
    seconds = np.arange(frame_count) / sample_rate_hz
    tone = np.zeros(frame_count)
    for harmonic, harmonic_amplitude in enumerate(harmonic_amplitudes, start=1):
        if harmonic * fundamental_hz >= sample_rate_hz / 2:
            break  # above Nyquist: would alias
        decay = np.exp(-seconds * harmonic / decay_seconds)
        tone += harmonic_amplitude * decay * np.sin(2 * np.pi * harmonic * fundamental_hz * seconds)

    envelope = np.ones(frame_count)
    attack_frames = min(seconds_to_frames(ATTACK_SECONDS, sample_rate_hz), frame_count // 2)
    release_frames = min(seconds_to_frames(RELEASE_SECONDS, sample_rate_hz), frame_count // 2)
    envelope[:attack_frames] = np.linspace(0.0, 1.0, attack_frames, endpoint=False)
    envelope[frame_count - release_frames :] = np.linspace(1.0, 0.0, release_frames)
    tone *= envelope

    peak = np.max(np.abs(tone)) if frame_count else 0.0
    if peak > 0:
        tone *= amplitude / peak
    return tone.astype(np.float32)

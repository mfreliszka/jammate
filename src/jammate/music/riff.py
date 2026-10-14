"""Bar-quantized riff fingerprinting, detection, and bounded session memory (Step 5.3).

At each bar boundary from `BeatClock`, `RiffDetector` quantizes the completed bar's onsets and
notes into a 16-step `BarFingerprint` (§41, §42, §98, §144, §146):
- `onset_mask`: 16-step normalized onset-strength mask (`0.0..1.0`);
- `pitch_classes`: 16-step relative pitch-class contour (`-1` for rest, `0..11` relative to
  `root_pc`);
- `root_pc`: dominant pitch class (`0..11`) of the bar;
- `mean_intensity`: average intensity across the bar (`0.0..1.0`).

It compares 1-bar and 2-bar sliding windows using a weighted similarity metric combining rhythmic
cosine similarity, pitch-contour agreement, and root pitch-class agreement. Consecutive matching
windows lock `riff_id` (`"riff_A"`, `"riff_B"`, ...) and increment `riff_iteration`. When a
playing bar diverges below `riff_change_threshold`, `RiffDetector` flags `riff_changed = True` and
either recalls a matching riff from `SessionMemory` (`max_riffs = 16`) or mints a new `riff_id`.

Realtime safety: runs on the analysis worker thread (never in the audio callback).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

from jammate.analysis.key import BassRegister
from jammate.config import RiffConfig
from jammate.music.beat_clock import BeatClockSnapshot
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    MusicalEvent,
    OnsetEvent,
    PitchEvent,
    PlayingStateUpdate,
    SampleFrame,
    SilenceEvent,
)


@dataclass(frozen=True, slots=True)
class BarFingerprint:
    """Quantized 1-bar musical fingerprint (§146)."""

    bar_index: int
    onset_mask: tuple[float, ...]
    pitch_classes: tuple[int, ...]
    root_pc: int | None
    mean_intensity: float = 0.0

    def __post_init__(self) -> None:
        if len(self.onset_mask) == 0 or len(self.onset_mask) != len(self.pitch_classes):
            raise ValueError("onset_mask and pitch_classes must be non-empty and equal length")
        if self.root_pc is not None and not 0 <= self.root_pc < 12:
            raise ValueError(f"root_pc must be in 0..11 or None, got {self.root_pc}")

    @property
    def steps_per_bar(self) -> int:
        return len(self.onset_mask)

    @property
    def active_steps(self) -> int:
        return sum(
            1
            for onset, pc in zip(self.onset_mask, self.pitch_classes, strict=True)
            if onset > 0.0 or pc >= 0
        )

    @property
    def is_silent(self) -> bool:
        return self.active_steps == 0

    @property
    def absolute_pitch_classes(self) -> tuple[int, ...]:
        if self.root_pc is None:
            return (-1,) * len(self.pitch_classes)
        return tuple(-1 if pc < 0 else (self.root_pc + pc) % 12 for pc in self.pitch_classes)


@dataclass(frozen=True, slots=True)
class RiffMemoryEntry:
    """Canonical fingerprint and musical preferences stored for a riff in `SessionMemory` (§146)."""

    riff_id: str
    fingerprint: BarFingerprint
    second_bar_fingerprint: BarFingerprint | None = None
    accent_mask: tuple[float, ...] = ()
    preferred_drum_pattern_id: str | None = None
    preferred_intensity_tier: str | None = None
    estimated_key: str | None = None
    bass_register: BassRegister = "low"
    times_played: int = 1
    last_seen_bar: int = 0

    @property
    def window_bars(self) -> int:
        return 2 if self.second_bar_fingerprint is not None else 1


@dataclass(frozen=True, slots=True)
class RiffDetectionResult:
    """Result of evaluating a completed bar in `RiffDetector` (Step 5.3, §146)."""

    bar_index: int
    fingerprint: BarFingerprint
    riff_id: str | None
    riff_iteration: int
    riff_changed: bool
    similarity: float
    recalled_entry: RiffMemoryEntry | None = None


def quantize_bar(
    bar_index: int,
    bar_start_frame: SampleFrame,
    bar_end_frame: SampleFrame,
    *,
    onsets: Sequence[OnsetEvent] = (),
    notes: Sequence[PitchEvent] = (),
    silences: Sequence[SilenceEvent] = (),
    mean_intensity: float = 0.0,
    steps_per_bar: int = 16,
    initial_note: MidiNote | None = None,
) -> BarFingerprint:
    """Quantize a bar's events in `[bar_start_frame, bar_end_frame)` onto `steps_per_bar` steps.

    Uses nearest-step rounding (`[-0.5, steps_per_bar - 0.5)`) so a beat-1 attack played slightly
    early (e.g. `-15 ms` timing jitter) lands on step 0 of this bar rather than step 15 of the
    previous bar, and snaps stabilized `PitchEvent`s back to their preceding attack onset.
    """
    if steps_per_bar <= 0:
        raise ValueError(f"steps_per_bar must be > 0, got {steps_per_bar}")
    bar_frames = max(1, bar_end_frame - bar_start_frame)
    step_frames = bar_frames / float(steps_per_bar)

    raw_onset = [0.0] * steps_per_bar
    bar_onsets: list[OnsetEvent] = []
    for onset in onsets:
        rel_steps = (onset.frame_index - bar_start_frame) / step_frames
        if -0.5 <= rel_steps < steps_per_bar - 0.5:
            step_idx = max(0, min(steps_per_bar - 1, round(rel_steps)))
            raw_onset[step_idx] = max(raw_onset[step_idx], float(onset.strength))
            bar_onsets.append(onset)

    attack_midi: list[int | None] = [None] * steps_per_bar
    lowest_midi_by_pc: dict[int, int] = {}

    sorted_notes = sorted(notes, key=lambda n: n.frame_index)
    for idx, note in enumerate(sorted_notes):
        prev_note_frame = sorted_notes[idx - 1].frame_index if idx > 0 else -1
        effective_frame = note.frame_index
        # Snap stabilized note start back to its attack onset if one preceded it within ~1.25 steps.
        snap_window = 1.25 * step_frames
        candidate_onsets = [
            o
            for o in onsets
            if o.frame_index > prev_note_frame
            and 0 <= note.frame_index - o.frame_index <= snap_window
        ]
        if candidate_onsets:
            effective_frame = candidate_onsets[-1].frame_index

        rel_steps = (effective_frame - bar_start_frame) / step_frames
        if -0.5 <= rel_steps < steps_per_bar - 0.5:
            step_idx = max(0, min(steps_per_bar - 1, round(rel_steps)))
            attack_midi[step_idx] = note.midi_note
            pc = note.midi_note % 12
            if pc not in lowest_midi_by_pc or note.midi_note < lowest_midi_by_pc[pc]:
                lowest_midi_by_pc[pc] = note.midi_note
            if raw_onset[step_idx] <= 0.0:
                raw_onset[step_idx] = max(0.1, 0.6 * float(note.confidence))

    peak_onset = max(raw_onset, default=0.0)
    if peak_onset > 0.0:
        onset_mask = tuple(val / peak_onset for val in raw_onset)
    else:
        onset_mask = (0.0,) * steps_per_bar

    silence_steps: set[int] = set()
    for silence in silences:
        rel_steps = (silence.frame_index - bar_start_frame) / step_frames
        if 0.0 <= rel_steps < steps_per_bar:
            silence_steps.add(max(0, min(steps_per_bar - 1, math.floor(rel_steps))))

    # Hold each attacked note for up to 1 beat (steps_per_bar // 4 steps) until the next attack so
    # staccato vs legato articulation of the same riff produces a consistent contour.
    max_hold_steps = max(1, steps_per_bar // 4)
    step_midi: list[int] = [-1] * steps_per_bar
    current_midi: int = -1
    hold_remaining = 0

    if initial_note is not None and attack_midi[0] is None and 0 not in silence_steps:
        current_midi = initial_note
        hold_remaining = max_hold_steps
        pc = initial_note % 12
        if pc not in lowest_midi_by_pc or initial_note < lowest_midi_by_pc[pc]:
            lowest_midi_by_pc[pc] = initial_note

    for s in range(steps_per_bar):
        if attack_midi[s] is not None:
            current_midi = attack_midi[s]  # type: ignore[assignment]
            hold_remaining = max_hold_steps
        elif hold_remaining <= 0:
            current_midi = -1
        if current_midi >= 0 and hold_remaining > 0:
            step_midi[s] = current_midi
            hold_remaining -= 1

    if all(m < 0 for m in step_midi):
        return BarFingerprint(
            bar_index=bar_index,
            onset_mask=onset_mask,
            pitch_classes=(-1,) * steps_per_bar,
            root_pc=None,
            mean_intensity=max(0.0, min(1.0, float(mean_intensity))),
        )

    # Determine dominant root_pc with metric downbeat prior and lowest-pitch tie-breaker.
    pc_scores = [0.0] * 12
    beat_stride = max(1, steps_per_bar // 4)
    for s, midi_val in enumerate(step_midi):
        if midi_val < 0:
            continue
        pc = midi_val % 12
        if s == 0:
            metric_w = 2.5
        elif s % beat_stride == 0:
            metric_w = 1.4
        else:
            metric_w = 1.0
        attack_bonus = 1.5 if attack_midi[s] is not None else 0.5
        pc_scores[pc] += metric_w * attack_bonus * (1.0 + 0.5 * onset_mask[s])

    if lowest_midi_by_pc:
        min_midi = min(lowest_midi_by_pc.values())
        for pc, midi_val in lowest_midi_by_pc.items():
            if midi_val == min_midi:
                pc_scores[pc] += 0.35

    root_pc = max(range(12), key=lambda pc: pc_scores[pc])
    pitch_classes = tuple(-1 if m < 0 else (m % 12 - root_pc) % 12 for m in step_midi)
    return BarFingerprint(
        bar_index=bar_index,
        onset_mask=onset_mask,
        pitch_classes=pitch_classes,
        root_pc=root_pc,
        mean_intensity=max(0.0, min(1.0, float(mean_intensity))),
    )


def _smoothed_onset_vector(mask: tuple[float, ...]) -> list[float]:
    n = len(mask)
    return [mask[i] + 0.22 * (mask[(i - 1) % n] + mask[(i + 1) % n]) for i in range(n)]


def rhythmic_cosine_similarity(a: BarFingerprint, b: BarFingerprint) -> float:
    """Cosine similarity (`0.0..1.0`) between two 16-step onset masks with neighbor tolerance."""
    if len(a.onset_mask) != len(b.onset_mask):
        raise ValueError("BarFingerprints must have the same steps_per_bar")
    va = _smoothed_onset_vector(a.onset_mask)
    vb = _smoothed_onset_vector(b.onset_mask)
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def pitch_contour_similarity(a: BarFingerprint, b: BarFingerprint) -> float:
    """Pitch-class contour agreement (`0.0..1.0`) across active steps."""
    if len(a.pitch_classes) != len(b.pitch_classes):
        raise ValueError("BarFingerprints must have the same steps_per_bar")
    if a.root_pc is None or b.root_pc is None:
        return 0.0
    abs_a = a.absolute_pitch_classes
    abs_b = b.absolute_pitch_classes
    matches = 0.0
    total = 0.0
    for s, (pa, pb) in enumerate(zip(abs_a, abs_b, strict=True)):
        if pa < 0 and pb < 0:
            continue
        step_w = 1.25 if (a.onset_mask[s] > 0.0 or b.onset_mask[s] > 0.0) else 0.75
        if pa >= 0 and pb >= 0:
            total += step_w
            if pa == pb:
                matches += step_w
        else:
            # One bar has a held/attacked note and the other is a rest (e.g. a missed note).
            total += 0.45 * step_w
    if total <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, matches / total))


def fingerprint_similarity(
    a: BarFingerprint,
    b: BarFingerprint,
    config: RiffConfig | None = None,
) -> float:
    """Weighted similarity (`0.0..1.0`) between two `BarFingerprint`s (§146)."""
    if a.is_silent or b.is_silent:
        return 0.0
    cfg = config or RiffConfig()
    rhythm_sim = rhythmic_cosine_similarity(a, b)
    pitch_sim = pitch_contour_similarity(a, b)
    root_sim = 1.0 if (a.root_pc is not None and a.root_pc == b.root_pc) else 0.0
    weight_sum = cfg.rhythm_weight + cfg.pitch_weight + cfg.root_weight
    score = (
        cfg.rhythm_weight * rhythm_sim + cfg.pitch_weight * pitch_sim + cfg.root_weight * root_sim
    ) / weight_sum
    return max(0.0, min(1.0, score))


def two_bar_similarity(
    window_a: tuple[BarFingerprint, BarFingerprint],
    window_b: tuple[BarFingerprint, BarFingerprint],
    config: RiffConfig | None = None,
) -> float:
    """Similarity (`0.0..1.0`) between two 2-bar windows."""
    sim_0 = fingerprint_similarity(window_a[0], window_b[0], config)
    sim_1 = fingerprint_similarity(window_a[1], window_b[1], config)
    return 0.5 * (sim_0 + sim_1)


def _blend_accent_mask(
    old_mask: tuple[float, ...],
    new_mask: tuple[float, ...],
    alpha: float = 0.35,
) -> tuple[float, ...]:
    if not old_mask or len(old_mask) != len(new_mask):
        return new_mask
    blended = [(1.0 - alpha) * o + alpha * n for o, n in zip(old_mask, new_mask, strict=True)]
    peak = max(blended, default=0.0)
    if peak > 0.0:
        return tuple(v / peak for v in blended)
    return tuple(blended)


def _riff_label_for_index(index: int) -> str:
    """Convert 0-based riff index into `'riff_A'`, `'riff_B'`, ..., `'riff_Z'`, `'riff_AA'`."""
    if index < 26:
        return f"riff_{chr(ord('A') + index)}"
    q, r = divmod(index, 26)
    return f"riff_{chr(ord('A') + q - 1)}{chr(ord('A') + r)}"


def _normalize_groove_pattern_id(pattern_id: str | None) -> str | None:
    """Strip temporary `_fill` suffix so `SessionMemory` stores groove pattern IDs."""
    if pattern_id is None:
        return None
    if pattern_id.endswith("_fill"):
        return pattern_id[:-5]
    return pattern_id


class SessionMemory:
    """Bounded table (`max_riffs = 16`) mapping `riff_id -> RiffMemoryEntry` (§98, §146)."""

    def __init__(self, max_riffs: int = 16) -> None:
        if max_riffs <= 0:
            raise ValueError(f"max_riffs must be > 0, got {max_riffs}")
        self._max_riffs = max_riffs
        self._entries: dict[str, RiffMemoryEntry] = {}
        self._minted_count = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def max_riffs(self) -> int:
        return self._max_riffs

    @property
    def entries(self) -> dict[str, RiffMemoryEntry]:
        return dict(self._entries)

    def get(self, riff_id: str) -> RiffMemoryEntry | None:
        return self._entries.get(riff_id)

    def next_riff_id(self) -> str:
        riff_id = _riff_label_for_index(self._minted_count)
        self._minted_count += 1
        return riff_id

    def find_match(
        self,
        bar_fp: BarFingerprint,
        prev_bar_fp: BarFingerprint | None = None,
        *,
        config: RiffConfig | None = None,
        threshold: float | None = None,
        exclude_riff_id: str | None = None,
    ) -> tuple[RiffMemoryEntry, float] | None:
        """Find the best matching `RiffMemoryEntry` at or above `threshold`."""
        cfg = config or RiffConfig()
        min_score = cfg.riff_lock_similarity if threshold is None else threshold
        best_entry: RiffMemoryEntry | None = None
        best_score = -1.0

        for riff_id, entry in self._entries.items():
            if riff_id == exclude_riff_id:
                continue
            score = fingerprint_similarity(bar_fp, entry.fingerprint, cfg)
            if entry.second_bar_fingerprint is not None:
                score_second = fingerprint_similarity(bar_fp, entry.second_bar_fingerprint, cfg)
                score = max(score, score_second)
                if prev_bar_fp is not None:
                    score_2bar = two_bar_similarity(
                        (prev_bar_fp, bar_fp),
                        (entry.fingerprint, entry.second_bar_fingerprint),
                        cfg,
                    )
                    score = max(score, score_2bar)
            if score >= min_score and score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None:
            return None
        return best_entry, best_score

    def store(self, entry: RiffMemoryEntry) -> RiffMemoryEntry:
        """Insert or replace `entry`, evicting the oldest-seen riff if at capacity."""
        norm_pat = _normalize_groove_pattern_id(entry.preferred_drum_pattern_id)
        if norm_pat != entry.preferred_drum_pattern_id:
            entry = replace(entry, preferred_drum_pattern_id=norm_pat)
        if entry.riff_id not in self._entries and len(self._entries) >= self._max_riffs:
            oldest_id = min(
                self._entries.values(),
                key=lambda e: (e.last_seen_bar, e.riff_id),
            ).riff_id
            del self._entries[oldest_id]
        self._entries[entry.riff_id] = entry
        return entry

    def record_play(
        self,
        riff_id: str,
        bar_fp: BarFingerprint,
        *,
        preferred_drum_pattern_id: str | None = None,
        preferred_intensity_tier: str | None = None,
        estimated_key: str | None = None,
        bass_register: BassRegister | None = None,
        increment_times_played: bool = True,
    ) -> RiffMemoryEntry:
        """Update an existing `RiffMemoryEntry` when `riff_id` is played again."""
        existing = self._entries[riff_id]
        norm_pat = _normalize_groove_pattern_id(preferred_drum_pattern_id)
        updated = replace(
            existing,
            accent_mask=_blend_accent_mask(existing.accent_mask, bar_fp.onset_mask),
            preferred_drum_pattern_id=(
                norm_pat if norm_pat is not None else existing.preferred_drum_pattern_id
            ),
            preferred_intensity_tier=(
                preferred_intensity_tier
                if preferred_intensity_tier is not None
                else existing.preferred_intensity_tier
            ),
            estimated_key=(estimated_key if estimated_key is not None else existing.estimated_key),
            bass_register=(bass_register if bass_register is not None else existing.bass_register),
            times_played=existing.times_played + (1 if increment_times_played else 0),
            last_seen_bar=bar_fp.bar_index,
        )
        self._entries[riff_id] = updated
        return updated

    def update_preferences(
        self,
        riff_id: str,
        *,
        preferred_drum_pattern_id: str | None = None,
        preferred_intensity_tier: str | None = None,
        estimated_key: str | None = None,
        bass_register: BassRegister | None = None,
    ) -> RiffMemoryEntry | None:
        """Update stored arrangement preferences for `riff_id`."""
        existing = self._entries.get(riff_id)
        if existing is None:
            return None
        updated = replace(
            existing,
            preferred_drum_pattern_id=(
                preferred_drum_pattern_id
                if preferred_drum_pattern_id is not None
                else existing.preferred_drum_pattern_id
            ),
            preferred_intensity_tier=(
                preferred_intensity_tier
                if preferred_intensity_tier is not None
                else existing.preferred_intensity_tier
            ),
            estimated_key=(estimated_key if estimated_key is not None else existing.estimated_key),
            bass_register=(bass_register if bass_register is not None else existing.bass_register),
        )
        self._entries[riff_id] = updated
        return updated

    def clear(self) -> None:
        self._entries.clear()
        self._minted_count = 0


class RiffDetector:
    """Sliding 1-bar and 2-bar riff detector with `SessionMemory` recall (Step 5.3, §41, §146)."""

    def __init__(self, config: RiffConfig | None = None) -> None:
        self._config = config or RiffConfig()
        self._memory = SessionMemory(max_riffs=self._config.max_riffs)
        self._recent_playing_bars: deque[BarFingerprint] = deque(maxlen=8)
        self._active_riff_id: str | None = None
        self._riff_iteration: int = 0
        self._riff_changed: bool = False
        self._two_bar_phase: int = 0
        self._silent_bars: int = 0
        self._last_recalled_entry: RiffMemoryEntry | None = None

        # Event-stream state for live / offline BeatClock-driven bar evaluation:
        self._onsets: deque[OnsetEvent] = deque(maxlen=512)
        self._notes: deque[PitchEvent] = deque(maxlen=512)
        self._silences: deque[SilenceEvent] = deque(maxlen=256)
        self._latest_intensity: float = 0.0
        self._intensity_samples: list[tuple[SampleFrame, float]] = []
        self._current_note: MidiNote | None = None
        self._note_at_frame: list[tuple[SampleFrame, MidiNote | None]] = []
        self._last_evaluated_bar: int = 0

    @property
    def config(self) -> RiffConfig:
        return self._config

    @property
    def memory(self) -> SessionMemory:
        return self._memory

    @property
    def riff_id(self) -> str | None:
        return self._active_riff_id

    @property
    def riff_iteration(self) -> int:
        return self._riff_iteration

    @property
    def riff_changed(self) -> bool:
        return self._riff_changed

    @property
    def last_recalled_entry(self) -> RiffMemoryEntry | None:
        return self._last_recalled_entry

    def reset(self) -> None:
        """Reset all riff-tracking state and clear `SessionMemory`."""
        self._memory.clear()
        self._recent_playing_bars.clear()
        self._active_riff_id = None
        self._riff_iteration = 0
        self._riff_changed = False
        self._two_bar_phase = 0
        self._silent_bars = 0
        self._last_recalled_entry = None
        self._onsets.clear()
        self._notes.clear()
        self._silences.clear()
        self._latest_intensity = 0.0
        self._intensity_samples.clear()
        self._current_note = None
        self._note_at_frame.clear()
        self._last_evaluated_bar = 0

    def handle_event(self, event: MusicalEvent) -> None:
        """Record an analysis event for subsequent bar quantization."""
        match event:
            case OnsetEvent():
                self._onsets.append(event)
            case PitchEvent():
                self._notes.append(event)
                self._current_note = event.midi_note
                self._note_at_frame.append((event.frame_index, event.midi_note))
            case SilenceEvent():
                self._silences.append(event)
                self._current_note = None
                self._note_at_frame.append((event.frame_index, None))
            case IntensityUpdate(intensity=intensity):
                self._latest_intensity = intensity
                self._intensity_samples.append((event.frame_index, intensity))
            case PlayingStateUpdate():
                pass

    def set_steps_per_bar(self, steps_per_bar: int) -> None:
        """Update `steps_per_bar` (`16` for 4/4, `12` for 3/4 and 6/8) and reset bars."""
        if steps_per_bar < 4:
            raise ValueError(f"steps_per_bar must be >= 4, got {steps_per_bar}")
        if steps_per_bar == self._config.steps_per_bar:
            return
        self._config = self._config.model_copy(update={"steps_per_bar": steps_per_bar})
        self.reset()

    def advance_to(
        self,
        frame_index: SampleFrame,
        beat_clock: BeatClockSnapshot | None,
        *,
        preferred_drum_pattern_id: str | None = None,
        preferred_intensity_tier: str | None = None,
        estimated_key: str | None = None,
        bass_register: BassRegister = "low",
    ) -> RiffDetectionResult | None:
        """Evaluate any newly completed bars up to `frame_index` according to `beat_clock`."""
        if beat_clock is None:
            self._last_evaluated_bar = 0
            return None

        target_steps = 12 if beat_clock.beats_per_bar in (3, 6) else self._config.steps_per_bar
        if (
            self._recent_playing_bars
            and self._recent_playing_bars[-1].steps_per_bar != target_steps
        ):
            self.reset()

        pos = beat_clock.position_at(frame_index)
        target_completed_bar = pos.bar - 1
        if target_completed_bar <= self._last_evaluated_bar:
            return None

        latest_result: RiffDetectionResult | None = None
        for bar_idx in range(self._last_evaluated_bar + 1, target_completed_bar + 1):
            bar_start_frame = beat_clock.beat_to_frame(bar_idx, 1)
            bar_end_frame = beat_clock.beat_to_frame(bar_idx + 1, 1)

            initial_note = self._note_active_at(bar_start_frame)
            mean_int = self._mean_intensity_in(bar_start_frame, bar_end_frame)
            fp = quantize_bar(
                bar_idx,
                bar_start_frame,
                bar_end_frame,
                onsets=tuple(self._onsets),
                notes=tuple(self._notes),
                silences=tuple(self._silences),
                mean_intensity=mean_int,
                steps_per_bar=target_steps,
                initial_note=initial_note,
            )
            latest_result = self.observe_bar(
                fp,
                preferred_drum_pattern_id=preferred_drum_pattern_id,
                preferred_intensity_tier=preferred_intensity_tier,
                estimated_key=estimated_key,
                bass_register=bass_register,
            )
            self._last_evaluated_bar = bar_idx
            self._prune_events_before(bar_start_frame - (bar_end_frame - bar_start_frame))

        return latest_result

    def observe_bar(
        self,
        bar_fp: BarFingerprint,
        *,
        preferred_drum_pattern_id: str | None = None,
        preferred_intensity_tier: str | None = None,
        estimated_key: str | None = None,
        bass_register: BassRegister = "low",
    ) -> RiffDetectionResult:
        """Process one completed `BarFingerprint` and update active riff state and memory."""
        if bar_fp.is_silent:
            self._silent_bars += 1
            self._riff_changed = False
            self._last_recalled_entry = None
            if self._silent_bars >= 2:
                self._active_riff_id = None
                self._riff_iteration = 0
                self._two_bar_phase = 0
            return RiffDetectionResult(
                bar_index=bar_fp.bar_index,
                fingerprint=bar_fp,
                riff_id=self._active_riff_id,
                riff_iteration=self._riff_iteration,
                riff_changed=False,
                similarity=0.0,
                recalled_entry=None,
            )

        self._silent_bars = 0
        prev_bar = self._recent_playing_bars[-1] if self._recent_playing_bars else None
        self._recent_playing_bars.append(bar_fp)

        # Case 1: Currently locked on an active riff.
        if self._active_riff_id is not None:
            current_entry = self._memory.get(self._active_riff_id)
            if current_entry is not None:
                sim = self._similarity_to_entry(bar_fp, prev_bar, current_entry)
                if sim >= self._config.riff_change_threshold:
                    if current_entry.window_bars == 2:
                        self._two_bar_phase = (self._two_bar_phase + 1) % 2
                        inc = self._two_bar_phase == 0
                        if inc:
                            self._riff_iteration += 1
                    else:
                        inc = True
                        self._riff_iteration += 1
                    updated = self._memory.record_play(
                        current_entry.riff_id,
                        bar_fp,
                        preferred_drum_pattern_id=preferred_drum_pattern_id,
                        preferred_intensity_tier=preferred_intensity_tier,
                        estimated_key=estimated_key,
                        bass_register=bass_register,
                        increment_times_played=inc,
                    )
                    self._riff_changed = False
                    self._last_recalled_entry = updated
                    return RiffDetectionResult(
                        bar_index=bar_fp.bar_index,
                        fingerprint=bar_fp,
                        riff_id=self._active_riff_id,
                        riff_iteration=self._riff_iteration,
                        riff_changed=False,
                        similarity=sim,
                        recalled_entry=updated,
                    )

                # Bar diverged below riff_change_threshold!
                # Check if it matches an earlier riff in SessionMemory.
                match = self._memory.find_match(
                    bar_fp,
                    prev_bar,
                    config=self._config,
                    threshold=self._config.riff_lock_similarity,
                    exclude_riff_id=current_entry.riff_id,
                )
                if match is not None:
                    matched_entry, match_sim = match
                    self._active_riff_id = matched_entry.riff_id
                    self._riff_iteration = 1
                    self._two_bar_phase = 1 if matched_entry.window_bars == 2 else 0
                    self._riff_changed = True
                    updated = self._memory.record_play(
                        matched_entry.riff_id,
                        bar_fp,
                        preferred_drum_pattern_id=(
                            matched_entry.preferred_drum_pattern_id or preferred_drum_pattern_id
                        ),
                        preferred_intensity_tier=(
                            matched_entry.preferred_intensity_tier or preferred_intensity_tier
                        ),
                        estimated_key=estimated_key,
                        bass_register=bass_register,
                        increment_times_played=True,
                    )
                    self._last_recalled_entry = updated
                    return RiffDetectionResult(
                        bar_index=bar_fp.bar_index,
                        fingerprint=bar_fp,
                        riff_id=self._active_riff_id,
                        riff_iteration=self._riff_iteration,
                        riff_changed=True,
                        similarity=match_sim,
                        recalled_entry=updated,
                    )

                # Otherwise mint a new riff_id on this bar boundary (§146).
                new_id = self._memory.next_riff_id()
                new_entry = self._memory.store(
                    RiffMemoryEntry(
                        riff_id=new_id,
                        fingerprint=bar_fp,
                        accent_mask=bar_fp.onset_mask,
                        preferred_drum_pattern_id=preferred_drum_pattern_id,
                        preferred_intensity_tier=preferred_intensity_tier,
                        estimated_key=estimated_key,
                        bass_register=bass_register,
                        times_played=1,
                        last_seen_bar=bar_fp.bar_index,
                    )
                )
                self._active_riff_id = new_id
                self._riff_iteration = 1
                self._two_bar_phase = 0
                self._riff_changed = True
                self._last_recalled_entry = new_entry
                return RiffDetectionResult(
                    bar_index=bar_fp.bar_index,
                    fingerprint=bar_fp,
                    riff_id=new_id,
                    riff_iteration=1,
                    riff_changed=True,
                    similarity=sim,
                    recalled_entry=new_entry,
                )

        # Case 2: No active riff locked yet (or after a multi-bar silence).
        match = self._memory.find_match(
            bar_fp,
            prev_bar,
            config=self._config,
            threshold=self._config.riff_lock_similarity,
        )
        if match is not None:
            matched_entry, match_sim = match
            self._active_riff_id = matched_entry.riff_id
            self._riff_iteration = 1
            self._two_bar_phase = 1 if matched_entry.window_bars == 2 else 0
            self._riff_changed = True
            updated = self._memory.record_play(
                matched_entry.riff_id,
                bar_fp,
                preferred_drum_pattern_id=(
                    matched_entry.preferred_drum_pattern_id or preferred_drum_pattern_id
                ),
                preferred_intensity_tier=(
                    matched_entry.preferred_intensity_tier or preferred_intensity_tier
                ),
                estimated_key=estimated_key,
                bass_register=bass_register,
                increment_times_played=True,
            )
            self._last_recalled_entry = updated
            return RiffDetectionResult(
                bar_index=bar_fp.bar_index,
                fingerprint=bar_fp,
                riff_id=self._active_riff_id,
                riff_iteration=self._riff_iteration,
                riff_changed=True,
                similarity=match_sim,
                recalled_entry=updated,
            )

        # Check 1-bar sliding window lock:
        if prev_bar is not None:
            sim_1 = fingerprint_similarity(bar_fp, prev_bar, self._config)
            if sim_1 >= self._config.riff_lock_similarity:
                new_id = self._memory.next_riff_id()
                entry = self._memory.store(
                    RiffMemoryEntry(
                        riff_id=new_id,
                        fingerprint=prev_bar,
                        accent_mask=_blend_accent_mask(prev_bar.onset_mask, bar_fp.onset_mask),
                        preferred_drum_pattern_id=preferred_drum_pattern_id,
                        preferred_intensity_tier=preferred_intensity_tier,
                        estimated_key=estimated_key,
                        bass_register=bass_register,
                        times_played=2,
                        last_seen_bar=bar_fp.bar_index,
                    )
                )
                self._active_riff_id = new_id
                self._riff_iteration = 2
                self._two_bar_phase = 0
                self._riff_changed = False
                self._last_recalled_entry = entry
                return RiffDetectionResult(
                    bar_index=bar_fp.bar_index,
                    fingerprint=bar_fp,
                    riff_id=new_id,
                    riff_iteration=2,
                    riff_changed=False,
                    similarity=sim_1,
                    recalled_entry=entry,
                )

        # Check 2-bar sliding window lock:
        if len(self._recent_playing_bars) >= 4:
            b0, b1, b2, b3 = list(self._recent_playing_bars)[-4:]
            sim_2 = two_bar_similarity((b0, b1), (b2, b3), self._config)
            if sim_2 >= self._config.riff_lock_similarity:
                new_id = self._memory.next_riff_id()
                entry = self._memory.store(
                    RiffMemoryEntry(
                        riff_id=new_id,
                        fingerprint=b0,
                        second_bar_fingerprint=b1,
                        accent_mask=_blend_accent_mask(b0.onset_mask, b2.onset_mask),
                        preferred_drum_pattern_id=preferred_drum_pattern_id,
                        preferred_intensity_tier=preferred_intensity_tier,
                        estimated_key=estimated_key,
                        bass_register=bass_register,
                        times_played=2,
                        last_seen_bar=bar_fp.bar_index,
                    )
                )
                self._active_riff_id = new_id
                self._riff_iteration = 2
                self._two_bar_phase = 0
                self._riff_changed = False
                self._last_recalled_entry = entry
                return RiffDetectionResult(
                    bar_index=bar_fp.bar_index,
                    fingerprint=bar_fp,
                    riff_id=new_id,
                    riff_iteration=2,
                    riff_changed=False,
                    similarity=sim_2,
                    recalled_entry=entry,
                )

        sim_any = (
            fingerprint_similarity(bar_fp, prev_bar, self._config) if prev_bar is not None else 0.0
        )
        self._riff_changed = False
        self._last_recalled_entry = None
        return RiffDetectionResult(
            bar_index=bar_fp.bar_index,
            fingerprint=bar_fp,
            riff_id=None,
            riff_iteration=0,
            riff_changed=False,
            similarity=sim_any,
            recalled_entry=None,
        )

    def _similarity_to_entry(
        self,
        bar_fp: BarFingerprint,
        prev_bar: BarFingerprint | None,
        entry: RiffMemoryEntry,
    ) -> float:
        if entry.second_bar_fingerprint is None:
            return fingerprint_similarity(bar_fp, entry.fingerprint, self._config)
        expected_fp = (
            entry.fingerprint if self._two_bar_phase == 0 else entry.second_bar_fingerprint
        )
        sim_expected = fingerprint_similarity(bar_fp, expected_fp, self._config)
        sim_other = fingerprint_similarity(
            bar_fp,
            entry.second_bar_fingerprint if self._two_bar_phase == 0 else entry.fingerprint,
            self._config,
        )
        best = max(sim_expected, sim_other)
        if prev_bar is not None and self._two_bar_phase == 1:
            sim_2bar = two_bar_similarity(
                (prev_bar, bar_fp),
                (entry.fingerprint, entry.second_bar_fingerprint),
                self._config,
            )
            best = max(best, sim_2bar)
        return best

    def _note_active_at(self, frame_index: SampleFrame) -> MidiNote | None:
        active: MidiNote | None = None
        for ev_frame, note in self._note_at_frame:
            if ev_frame <= frame_index:
                active = note
            else:
                break
        return active

    def _mean_intensity_in(
        self,
        start_frame: SampleFrame,
        end_frame: SampleFrame,
    ) -> float:
        in_window = [
            val for frame, val in self._intensity_samples if start_frame <= frame < end_frame
        ]
        if in_window:
            return sum(in_window) / len(in_window)
        return self._latest_intensity

    def _prune_events_before(self, min_frame: SampleFrame) -> None:
        while self._onsets and self._onsets[0].frame_index < min_frame:
            self._onsets.popleft()
        while self._notes and self._notes[0].frame_index < min_frame:
            self._notes.popleft()
        while self._silences and self._silences[0].frame_index < min_frame:
            self._silences.popleft()
        if len(self._intensity_samples) > 128:
            self._intensity_samples = [(f, v) for f, v in self._intensity_samples if f >= min_frame]
        if len(self._note_at_frame) > 128:
            last_before: tuple[SampleFrame, MidiNote | None] | None = None
            kept: list[tuple[SampleFrame, MidiNote | None]] = []
            for f, n in self._note_at_frame:
                if f < min_frame:
                    last_before = (f, n)
                else:
                    kept.append((f, n))
            self._note_at_frame = ([last_before] if last_before is not None else []) + kept

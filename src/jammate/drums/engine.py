"""Look-ahead drum scheduler, decision state machine and worker thread (Steps 3.3-3.4, §43-§49).

Separates musical decisions (`DrummerController`) from sample-frame scheduling
(`DrumStepScheduler`) and audio rendering (`DrumSampler`) (§125, §126):

- `DrummerController` is a pure state machine over `MusicalState` snapshots:
  `LISTENING -> WAITING_FOR_BAR -> PLAYING -> HOLDING -> STOPPED`.
- `DrumStepScheduler` converts upcoming 16-step bar positions into exact stream sample frames via
  `BeatClockSnapshot` and pushes each hit into `DrumEventRing` at most once (§78).
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from jammate.config import DrumConfig
from jammate.drums.adaptive import AccentTracker, adapt_pattern_to_accents
from jammate.drums.humanize import Humanizer
from jammate.drums.patterns import (
    INTENSITY_TIERS,
    STYLE_FAMILIES,
    DrumPattern,
    IntensityTier,
    StepHit,
    StyleFamily,
    get_fill_pattern,
    get_pattern,
    step_beat_offset,
    steps_per_bar_for_beats,
)
from jammate.drums.sampler import DrumEvent, DrumEventRing, DrumPiece
from jammate.music.beat_clock import BeatClock, BeatClockSnapshot, beat_to_frame
from jammate.music.director import BandDirective
from jammate.music.events import MusicalEvent, OnsetEvent, SampleFrame, seconds_to_frames
from jammate.music.state import MusicalState

logger = logging.getLogger(__name__)


class DrummerState(StrEnum):
    """Lifecycle states of the virtual drummer (Step 3.4, §30, §71)."""

    LISTENING = "listening"
    WAITING_FOR_BAR = "waiting_for_bar"
    PLAYING = "playing"
    HOLDING = "holding"
    STOPPED = "stopped"


@dataclass(frozen=True)
class DrummerDecision:
    """Decision output of `DrummerController.update` for the current `MusicalState`."""

    state: DrummerState
    style: StyleFamily
    tier: IntensityTier
    entry_bar: int | None
    pattern: DrumPattern | None
    is_fill: bool
    crash_on_downbeat: bool


def select_intensity_tier(
    intensity: float,
    current_tier: IntensityTier,
    config: DrumConfig,
) -> IntensityTier:
    """Map `intensity` (`0.0..1.0`) to a tier with hysteresis around `current_tier` (§48)."""
    low_mid = config.low_mid_intensity_threshold
    mid_high = config.mid_high_intensity_threshold
    hyst = config.intensity_hysteresis

    if current_tier == "low":
        if intensity >= mid_high + hyst:
            return "high"
        if intensity >= low_mid + hyst:
            return "mid"
        return "low"
    if current_tier == "mid":
        if intensity < low_mid - hyst:
            return "low"
        if intensity >= mid_high + hyst:
            return "high"
        return "mid"
    # current_tier == "high"
    if intensity < low_mid - hyst:
        return "low"
    if intensity < mid_high - hyst:
        return "mid"
    return "high"


class DrummerController:
    """Musical decision state machine for the virtual drummer (Step 3.4, §30, §48, §49, §71).

    Transitions:
    - `LISTENING -> WAITING_FOR_BAR`: once `tempo_locked` holds, `tempo_confidence >= threshold`,
      and `user_is_playing` is True, schedules entry for `bar + 1`.
    - `WAITING_FOR_BAR -> PLAYING`: on reaching `entry_bar`, enters with a crash on beat 1.
    - `PLAYING`: updates intensity tier with hysteresis at bar boundaries only, plays a fill every
      `fill_every_bars` bars, and monitors silence:
      - `< silence_hold_bars` (e.g. 1 beat): stays in `PLAYING`;
      - `>= silence_hold_bars` (2 bars): transitions to `HOLDING` (plays lighter `low` pattern);
      - `>= silence_stop_bars` (8 bars): transitions to `STOPPED`.
    - `HOLDING -> PLAYING`: immediately when the player resumes (`user_is_playing` becomes True).
    - `HOLDING -> STOPPED`: when silence reaches `silence_stop_bars`.
    - `STOPPED -> WAITING_FOR_BAR`: when the player resumes with locked tempo, re-enters on the
      next bar boundary with a crash.
    """

    def __init__(
        self,
        config: DrumConfig,
        *,
        beats_per_bar: int = 4,
        manual_bpm_active: bool = False,
    ) -> None:
        self._config = config
        self._beats_per_bar = beats_per_bar
        self._manual_bpm_active = manual_bpm_active
        self._style: StyleFamily = config.style
        self._fill_every_bars: int = config.fill_every_bars
        self._state: DrummerState = (
            DrummerState.PLAYING if manual_bpm_active else DrummerState.LISTENING
        )
        self._active_tier: IntensityTier = "mid"
        self._pending_tier: IntensityTier = "mid"
        self._entry_bar: int | None = 1 if manual_bpm_active else None
        self._last_bar: int | None = 1 if manual_bpm_active else None
        self._silent_since_bar_pos: float | None = None
        self._crash_bars: set[int] = {1} if manual_bpm_active else set()
        self._forced_fill_bars: set[int] = set()
        self.accent_tracker = AccentTracker(
            steps_per_bar=steps_per_bar_for_beats(beats_per_bar),
            window_bars=config.accent_window_bars,
            decay=config.accent_decay,
        )

    @property
    def beats_per_bar(self) -> int:
        return self._beats_per_bar

    def set_beats_per_bar(self, beats_per_bar: int) -> None:
        """Update `beats_per_bar` (`4`, `3`, or `6`) and sync the accent grid resolution."""
        if beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be >= 1, got {beats_per_bar}")
        self._beats_per_bar = int(beats_per_bar)
        self.accent_tracker.set_steps_per_bar(steps_per_bar_for_beats(self._beats_per_bar))

    @property
    def manual_bpm_active(self) -> bool:
        return self._manual_bpm_active

    def set_manual_bpm_active(self, active: bool) -> None:
        """Enable or disable manual BPM mode."""
        self._manual_bpm_active = bool(active)
        if self._manual_bpm_active and self._state in (
            DrummerState.LISTENING,
            DrummerState.STOPPED,
        ):
            self._state = DrummerState.PLAYING
            if self._entry_bar is None:
                self._entry_bar = 1
            if self._last_bar is None:
                self._last_bar = 1
            self._crash_bars.add(self._entry_bar)

    @property
    def state(self) -> DrummerState:
        return self._state

    @property
    def style(self) -> StyleFamily:
        return self._style

    def set_style(self, style: StyleFamily) -> None:
        self._style = style

    @property
    def fill_every_bars(self) -> int:
        return self._fill_every_bars

    def set_fill_every_bars(self, bars: int) -> None:
        if bars < 1:
            raise ValueError("fill_every_bars must be >= 1")
        self._fill_every_bars = int(bars)

    @property
    def active_tier(self) -> IntensityTier:
        return self._active_tier

    def set_tier(self, tier: IntensityTier) -> None:
        self._active_tier = tier
        self._pending_tier = tier

    @property
    def entry_bar(self) -> int | None:
        return self._entry_bar

    @property
    def crash_bars(self) -> frozenset[int]:
        return frozenset(self._crash_bars)

    @property
    def accent_profile(self) -> tuple[float, ...]:
        return self.accent_tracker.accent_profile

    def record_onset(self, onset: OnsetEvent, clock: BeatClockSnapshot | None) -> None:
        self.accent_tracker.record_onset(onset, clock)

    def handle_event(self, event: MusicalEvent, clock: BeatClockSnapshot | None) -> None:
        self.accent_tracker.handle_event(event, clock)

    def apply_directive(
        self,
        directive: BandDirective,
        current_bar: int | None = None,
    ) -> None:
        """Apply a `BandDirective` from `BandDirector` (Step 5.5, §148)."""
        if (
            directive.recalled_accent_mask is not None
            and len(directive.recalled_accent_mask) == self.accent_tracker.steps_per_bar
        ):
            self.accent_tracker.set_accent_profile(directive.recalled_accent_mask)
        if directive.recalled_pattern_id is not None:
            pat_id = directive.recalled_pattern_id
            for meter_suffix in ("_3_4", "_6_8"):
                if pat_id.endswith(meter_suffix):
                    pat_id = pat_id[: -len(meter_suffix)]
                    break
            for candidate_tier in INTENSITY_TIERS:
                suffix = f"_{candidate_tier}"
                if pat_id.endswith(suffix):
                    style_prefix = pat_id[: -len(suffix)]
                    if style_prefix in STYLE_FAMILIES:
                        self._style = style_prefix  # type: ignore[assignment]
                    self._active_tier = candidate_tier
                    self._pending_tier = candidate_tier
                    break
        if current_bar is not None and current_bar >= 1:
            if directive.trigger_crash_now:
                self._crash_bars.add(current_bar)
            if directive.trigger_fill_now and directive.drum_role == "fill":
                self._forced_fill_bars.add(current_bar)

    def reset(self) -> None:
        self._state = DrummerState.PLAYING if self._manual_bpm_active else DrummerState.LISTENING
        self._active_tier = "mid"
        self._pending_tier = "mid"
        self._entry_bar = 1 if self._manual_bpm_active else None
        self._last_bar = 1 if self._manual_bpm_active else None
        self._silent_since_bar_pos = None
        self._crash_bars = {1} if self._manual_bpm_active else set()
        self._forced_fill_bars.clear()
        self.accent_tracker.reset()

    def update(self, musical_state: MusicalState) -> DrummerDecision:
        """Advance the decision state machine from `musical_state` and return a decision."""
        if self._manual_bpm_active:
            bar = musical_state.bar or 1
            self._update_tier(musical_state.intensity, bar)
            return self._decision_for_bar(bar)

        if (
            not musical_state.tempo_locked
            or musical_state.bar is None
            or musical_state.beat is None
        ):
            self._state = DrummerState.LISTENING
            self._entry_bar = None
            self._last_bar = None
            self._silent_since_bar_pos = None
            return self._decision_for_bar(musical_state.bar)

        bar = musical_state.bar
        beat = musical_state.beat
        phase = musical_state.beat_phase or 0.0
        bar_pos = (bar - 1) + ((beat - 1) + phase) / self._beats_per_bar
        ready_to_enter = (
            musical_state.tempo_locked
            and musical_state.tempo_confidence >= self._config.entry_tempo_confidence
            and musical_state.user_is_playing
        )

        if self._state == DrummerState.LISTENING:
            if ready_to_enter:
                self._arm_entry(bar, musical_state.intensity)

        elif self._state == DrummerState.WAITING_FOR_BAR:
            if musical_state.tempo_confidence < self._config.entry_tempo_confidence:
                self._state = DrummerState.LISTENING
                self._entry_bar = None
            elif self._entry_bar is not None and bar >= self._entry_bar:
                self._state = DrummerState.PLAYING
                self._last_bar = bar
                self._silent_since_bar_pos = None if musical_state.user_is_playing else bar_pos

        elif self._state == DrummerState.PLAYING:
            self._update_tier(musical_state.intensity, bar)
            if musical_state.user_is_playing:
                self._silent_since_bar_pos = None
            else:
                if self._silent_since_bar_pos is None:
                    self._silent_since_bar_pos = bar_pos
                silent_bars = bar_pos - self._silent_since_bar_pos
                if silent_bars >= self._config.silence_stop_bars:
                    self._state = DrummerState.STOPPED
                elif silent_bars >= self._config.silence_hold_bars:
                    self._state = DrummerState.HOLDING

        elif self._state == DrummerState.HOLDING:
            self._update_tier(musical_state.intensity, bar)
            if musical_state.user_is_playing:
                self._silent_since_bar_pos = None
                self._state = DrummerState.PLAYING
            else:
                if self._silent_since_bar_pos is None:
                    self._silent_since_bar_pos = bar_pos
                silent_bars = bar_pos - self._silent_since_bar_pos
                if silent_bars >= self._config.silence_stop_bars:
                    self._state = DrummerState.STOPPED

        elif self._state == DrummerState.STOPPED and ready_to_enter:
            self._arm_entry(bar, musical_state.intensity)

        return self._decision_for_bar(bar)

    def pattern_for_bar(self, bar: int) -> DrumPattern | None:
        """Return the pattern to schedule for `bar` (1-based), or None when silent."""
        if bar < 1:
            return None
        if self._state in (DrummerState.LISTENING, DrummerState.STOPPED):
            return None
        if self._state == DrummerState.WAITING_FOR_BAR:
            if self._entry_bar is None or bar < self._entry_bar:
                return None
            return self._build_playing_pattern(bar, self._active_tier)
        if self._state == DrummerState.HOLDING:
            swing = self._config.blues_swing if self._style == "blues" else None
            return get_pattern(
                self._style,
                "low",
                swing=swing,
                beats_per_bar=self._beats_per_bar,
            )
        # PLAYING
        if self._entry_bar is not None and bar < self._entry_bar:
            return None
        return self._build_playing_pattern(bar, self._active_tier)

    def is_fill_bar(self, bar: int) -> bool:
        if self._state != DrummerState.PLAYING or self._entry_bar is None or bar < self._entry_bar:
            return False
        if bar in self._forced_fill_bars:
            return True
        bars_since_entry = bar - self._entry_bar + 1
        return bars_since_entry > 0 and bars_since_entry % self._fill_every_bars == 0

    def _arm_entry(self, current_bar: int, intensity: float) -> None:
        entry_bar = current_bar + 1
        initial_tier = select_intensity_tier(intensity, "mid", self._config)
        self._active_tier = initial_tier
        self._pending_tier = initial_tier
        self._entry_bar = entry_bar
        self._last_bar = current_bar
        self._silent_since_bar_pos = None
        self._crash_bars.add(entry_bar)
        self._state = DrummerState.WAITING_FOR_BAR
        logger.info(
            "Drummer armed to enter on bar %d (tier=%s, style=%s).",
            entry_bar,
            initial_tier,
            self._style,
        )

    def _update_tier(self, intensity: float, bar: int) -> None:
        self._pending_tier = select_intensity_tier(intensity, self._pending_tier, self._config)
        if self._last_bar is None or bar > self._last_bar:
            self._active_tier = self._pending_tier
            self._last_bar = bar

    def _build_playing_pattern(self, bar: int, tier: IntensityTier) -> DrumPattern:
        swing = self._config.blues_swing if self._style == "blues" else None
        base = get_pattern(
            self._style,
            tier,
            swing=swing,
            beats_per_bar=self._beats_per_bar,
        )
        if self._config.adaptive_accents_enabled:
            base = adapt_pattern_to_accents(base, self.accent_tracker.accent_profile, self._config)
        if self._entry_bar is not None and bar >= self._entry_bar:
            bars_since_entry = bar - self._entry_bar + 1
            if bar in self._forced_fill_bars or (
                bars_since_entry > 0 and bars_since_entry % self._fill_every_bars == 0
            ):
                return get_fill_pattern(
                    base,
                    swing=swing,
                    beats_per_bar=self._beats_per_bar,
                )
        return base

    def _decision_for_bar(self, bar: int | None) -> DrummerDecision:
        effective_tier: IntensityTier = (
            "low" if self._state == DrummerState.HOLDING else self._active_tier
        )
        pattern = (
            self.pattern_for_bar(bar)
            if bar is not None and self._state in (DrummerState.PLAYING, DrummerState.HOLDING)
            else None
        )
        is_fill = bar is not None and self.is_fill_bar(bar)
        crash = bar is not None and bar in self._crash_bars and pattern is not None
        return DrummerDecision(
            state=self._state,
            style=self._style,
            tier=effective_tier,
            entry_bar=self._entry_bar,
            pattern=pattern,
            is_fill=is_fill,
            crash_on_downbeat=crash,
        )


def step_to_frame(
    global_step: int,
    clock: BeatClockSnapshot,
    swing: float = 0.0,
) -> SampleFrame:
    """Convert global step (`bar_index * steps_per_bar + step_index`) to a stream frame."""
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")
    steps_per_bar = steps_per_bar_for_beats(clock.beats_per_bar)
    bar_index = global_step // steps_per_bar
    step_index = global_step % steps_per_bar
    beat_offset = step_beat_offset(
        step_index,
        swing=swing,
        beats_per_bar=clock.beats_per_bar,
        steps_per_bar=steps_per_bar,
    )
    beat_zero_based = int(beat_offset)
    beat_phase = beat_offset - beat_zero_based
    return beat_to_frame(
        bar=bar_index + 1,
        beat=beat_zero_based + 1,
        beat_phase=beat_phase,
        anchor_frame=clock.anchor_frame,
        bpm=clock.bpm,
        sample_rate_hz=clock.sample_rate_hz,
        beats_per_bar=clock.beats_per_bar,
    )


class DrumStepScheduler:
    """Deterministic look-ahead step scheduler (§46, §78).

    Tracks `_last_scheduled_step` so every grid step is evaluated and pushed into `event_ring` at
    most once, even when BeatClock shifts phase or adapts BPM mid-bar.
    """

    def __init__(
        self,
        event_ring: DrumEventRing,
        *,
        humanizer: Humanizer | None = None,
    ) -> None:
        self._ring = event_ring
        self._humanizer = humanizer
        self._last_scheduled_step: int | None = None
        self._last_pushed_frame: int | None = None
        self._last_steps_per_bar: int | None = None
        self.events_scheduled = 0

    @property
    def last_scheduled_step(self) -> int | None:
        return self._last_scheduled_step

    def reset(self) -> None:
        """Clear step cursor when the clock stops or the drummer stops."""
        self._last_scheduled_step = None
        self._last_pushed_frame = None
        self._last_steps_per_bar = None

    def schedule_window(
        self,
        current_frame: SampleFrame,
        lookahead_frames: int,
        clock: BeatClockSnapshot,
        pattern_for_bar: Callable[[int], DrumPattern | None],
        *,
        crash_on_bars: frozenset[int] = frozenset(),
        timing_offset_frames: int = 0,
        start_step: int | None = None,
        humanizer: Humanizer | None = None,
    ) -> list[DrumEvent]:
        """Schedule all unscheduled steps in `[current_frame, current_frame + lookahead_frames)`."""
        if lookahead_frames <= 0:
            return []
        window_end = current_frame + lookahead_frames
        active_humanizer = self._humanizer if humanizer is None else humanizer
        steps_per_bar = steps_per_bar_for_beats(clock.beats_per_bar)
        if self._last_steps_per_bar is not None and self._last_steps_per_bar != steps_per_bar:
            self._last_scheduled_step = None
        self._last_steps_per_bar = steps_per_bar

        if self._last_scheduled_step is None:
            if start_step is not None:
                next_step = max(0, start_step)
            else:
                next_step = self._first_step_at_or_after(
                    current_frame, clock, pattern_for_bar, timing_offset_frames
                )
        else:
            next_step = self._last_scheduled_step + 1
            if start_step is not None and start_step > next_step:
                next_step = start_step
            # If the clock re-anchored far ahead, don't burst-schedule bars from the distant past.
            earliest_valid = current_frame - max(lookahead_frames * 2, round(clock.frames_per_beat))
            probe_frame = step_to_frame(next_step, clock) + timing_offset_frames
            if probe_frame < earliest_valid:
                next_step = self._first_step_at_or_after(
                    current_frame, clock, pattern_for_bar, timing_offset_frames
                )

        scheduled: list[DrumEvent] = []
        k = next_step
        while True:
            bar = k // steps_per_bar + 1
            step_in_bar = k % steps_per_bar
            pattern = pattern_for_bar(bar)
            swing = pattern.swing if pattern is not None else 0.0
            raw_frame = step_to_frame(k, clock, swing=swing) + timing_offset_frames
            if raw_frame >= window_end:
                break

            self._last_scheduled_step = k
            if pattern is not None:
                step_hits: list[StepHit] = []
                hits = pattern.steps[step_in_bar % len(pattern.steps)]
                if step_in_bar == 0 and bar in crash_on_bars:
                    has_crash = any(hit.piece == DrumPiece.CRASH for hit in hits)
                    if not has_crash:
                        step_hits.append(StepHit(DrumPiece.CRASH, 0.90))
                step_hits.extend(hits)

                if active_humanizer is not None and active_humanizer.enabled:
                    is_fill = pattern.name in (
                        "fill_simple",
                        "fill_simple_12",
                    ) or pattern.name.endswith("_fill")
                    humanized = active_humanizer.humanize_step(
                        tuple(step_hits),
                        bar_index=bar - 1,
                        step_index=step_in_bar,
                        tier=pattern.tier,
                        is_fill=is_fill,
                    )
                    candidate_events = sorted(
                        (
                            max(current_frame, raw_frame + h.timing_offset_frames),
                            int(h.piece),
                            h.piece,
                            h.velocity,
                        )
                        for h in humanized
                    )
                    for hit_frame, _order, piece, velocity in candidate_events:
                        if (
                            self._last_pushed_frame is not None
                            and hit_frame < self._last_pushed_frame
                        ):
                            hit_frame = self._last_pushed_frame
                        event = DrumEvent(hit_frame, piece, velocity)
                        if self._ring.push(event.frame_index, event.piece, event.gain):
                            self._last_pushed_frame = event.frame_index
                            scheduled.append(event)
                            self.events_scheduled += 1
                else:
                    target_frame = max(current_frame, raw_frame)
                    if (
                        self._last_pushed_frame is not None
                        and target_frame < self._last_pushed_frame
                    ):
                        target_frame = self._last_pushed_frame
                    for hit in step_hits:
                        event = DrumEvent(target_frame, hit.piece, hit.velocity)
                        if self._ring.push(event.frame_index, event.piece, event.gain):
                            self._last_pushed_frame = event.frame_index
                            scheduled.append(event)
                            self.events_scheduled += 1
            k += 1

        return scheduled

    def _first_step_at_or_after(
        self,
        current_frame: SampleFrame,
        clock: BeatClockSnapshot,
        pattern_for_bar: Callable[[int], DrumPattern | None],
        timing_offset_frames: int,
    ) -> int:
        steps_per_bar = steps_per_bar_for_beats(clock.beats_per_bar)
        pos = clock.position_at(max(0, current_frame - timing_offset_frames))
        k = max(0, (pos.bar - 1) * steps_per_bar)
        while True:
            bar = k // steps_per_bar + 1
            pattern = pattern_for_bar(bar)
            swing = pattern.swing if pattern is not None else 0.0
            if step_to_frame(k, clock, swing=swing) + timing_offset_frames >= current_frame:
                return k
            k += 1


class DrumEngine:
    """Look-ahead drum scheduler worker thread and decision owner (Steps 3.3-3.4, §43, §78).

    Wakes every `scheduler_interval_seconds`, updates `DrummerController` from `MusicalState`,
    and schedules upcoming drum hits into `event_ring` up to `lookahead_seconds` ahead of the
    current stream frame. Can also be driven deterministically without a thread via
    `schedule_available(current_frame)`.
    """

    def __init__(
        self,
        config: DrumConfig,
        sample_rate_hz: int,
        event_ring: DrumEventRing,
        beat_clock: BeatClock,
        state_provider: Callable[[], MusicalState],
        frame_provider: Callable[[], SampleFrame],
        *,
        beats_per_bar: int = 4,
        manual_bpm_active: bool = False,
    ) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._beat_clock = beat_clock
        self._state_provider = state_provider
        self._frame_provider = frame_provider
        self.controller = DrummerController(
            config,
            beats_per_bar=beats_per_bar,
            manual_bpm_active=manual_bpm_active,
        )
        self.humanizer = Humanizer(config, sample_rate_hz=sample_rate_hz)
        self.scheduler = DrumStepScheduler(event_ring, humanizer=self.humanizer)
        self._lookahead_frames = max(1, seconds_to_frames(config.lookahead_seconds, sample_rate_hz))
        self._timing_offset_seconds = float(config.timing_offset_seconds)
        self._timing_offset_frames = seconds_to_frames(self._timing_offset_seconds, sample_rate_hz)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def accent_tracker(self) -> AccentTracker:
        return self.controller.accent_tracker

    def handle_event(self, event: MusicalEvent) -> None:
        """Forward onset events to `AccentTracker` using the current `BeatClock` snapshot."""
        self.controller.handle_event(event, self._beat_clock.snapshot)

    def apply_directive(
        self,
        directive: BandDirective,
        current_bar: int | None = None,
    ) -> None:
        """Forward a `BandDirective` from `BandDirector` to `DrummerController`."""
        self.controller.apply_directive(directive, current_bar)

    def set_beats_per_bar(self, beats_per_bar: int) -> None:
        """Update meter beats per bar (`4`, `3`, or `6`)."""
        self.controller.set_beats_per_bar(beats_per_bar)
        self.scheduler.reset()

    def set_manual_bpm_active(self, active: bool) -> None:
        """Enable or disable manual BPM mode on `DrummerController`."""
        self.controller.set_manual_bpm_active(active)

    def reset_context(self) -> None:
        """Reset drummer controller and step scheduler state (Step 6.1)."""
        self.controller.reset()
        self.scheduler.reset()

    @property
    def drummer_state(self) -> DrummerState:
        return self.controller.state

    @property
    def style(self) -> StyleFamily:
        return self.controller.style

    def set_style(self, style: StyleFamily) -> None:
        self.controller.set_style(style)

    @property
    def timing_offset_seconds(self) -> float:
        return self._timing_offset_seconds

    def set_timing_offset_seconds(self, offset_seconds: float) -> None:
        if not -0.1 <= offset_seconds <= 0.1:
            raise ValueError("timing_offset_seconds must be between -0.1 and 0.1")
        self._timing_offset_seconds = float(offset_seconds)
        self._timing_offset_frames = seconds_to_frames(
            self._timing_offset_seconds, self._sample_rate_hz
        )

    @property
    def tier(self) -> IntensityTier:
        return self.controller.active_tier

    def set_tier(self, tier: IntensityTier) -> None:
        self.controller.set_tier(tier)

    def current_pattern(self, bar: int = 1) -> DrumPattern:
        swing = self._config.blues_swing if self.style == "blues" else None
        base = get_pattern(
            self.style,
            self.tier,
            swing=swing,
            beats_per_bar=self.controller.beats_per_bar,
        )
        if self._config.adaptive_accents_enabled:
            base = adapt_pattern_to_accents(base, self.controller.accent_profile, self._config)
        if bar > 0 and bar % self.controller.fill_every_bars == 0:
            return get_fill_pattern(
                base,
                swing=swing,
                beats_per_bar=self.controller.beats_per_bar,
            )
        return base

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="jammate-drums", daemon=False)
        self._thread.start()
        logger.info(
            "Drum scheduler started: style=%s, lookahead=%.0f ms.",
            self.style,
            self._config.lookahead_seconds * 1000,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            logger.info(
                "Drum scheduler stopped (%d events scheduled).",
                self.scheduler.events_scheduled,
            )

    def schedule_available(self, current_frame: SampleFrame | None = None) -> list[DrumEvent]:
        """Run one decision + scheduling pass at `current_frame`."""
        clock = self._beat_clock.snapshot
        musical_state = self._state_provider()
        self.controller.update(musical_state)
        if clock is None or self.controller.state in (DrummerState.LISTENING, DrummerState.STOPPED):
            self.scheduler.reset()
            return []
        frame = self._frame_provider() if current_frame is None else current_frame
        return self.scheduler.schedule_window(
            current_frame=frame,
            lookahead_frames=self._lookahead_frames,
            clock=clock,
            pattern_for_bar=self.controller.pattern_for_bar,
            crash_on_bars=self.controller.crash_bars,
            timing_offset_frames=self._timing_offset_frames,
        )

    def _run(self) -> None:
        interval = self._config.scheduler_interval_seconds
        while not self._stop_event.is_set():
            self.schedule_available()
            self._stop_event.wait(interval)

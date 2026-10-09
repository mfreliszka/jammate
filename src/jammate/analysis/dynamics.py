"""Intensity and playing state: how hard, and whether, the player is playing (§47, §70).

Both trackers run on the analysis worker thread, once per hop, after pitch and onset detection,
and emit an event only when their value changes. Pure logic: test them with hand-made hops.
"""

import math
from collections import deque

from jammate.analysis.pitch import PitchFrame
from jammate.audio.levels import amplitude_to_dbfs
from jammate.config import AnalysisConfig
from jammate.music.events import (
    IntensityUpdate,
    MidiNote,
    OnsetEvent,
    PlayingState,
    PlayingStateUpdate,
    SampleFrame,
    frames_to_seconds,
    seconds_to_frames,
)


def _unit(value: float, low: float, high: float) -> float:
    return min(1.0, max(0.0, (value - low) / (high - low)))


class IntensityTracker:
    """Intensity 0.0-1.0 from three parts, each mapped to 0-1 (§47):
    - level: the window RMS between `quiet_dbfs` and `loud_dbfs`;
    - attack: the mean peak of the attacks in the last `window_seconds`;
    - density: attacks per second in that window, against `full_density_onsets_per_second`.

    Their weighted mean is smoothed (fast rise, slow fall). While the player isn't playing (see
    `PlayingStateTracker`), it falls towards 0: noise or a hum is not intensity. An
    `IntensityUpdate` is emitted when the smoothed value has moved `report_step` since the last
    one.
    """

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._config = config.intensity
        self._sample_rate_hz = sample_rate_hz
        self._window_frames = seconds_to_frames(self._config.window_seconds, sample_rate_hz)
        hop_seconds = config.hop_frames / sample_rate_hz
        self._rise = 1.0 - math.exp(-hop_seconds / self._config.rise_seconds)
        self._fall = 1.0 - math.exp(-hop_seconds / self._config.fall_seconds)
        self._attacks: deque[tuple[SampleFrame, float]] = deque()  # (frame, attack 0-1)
        self.value = 0.0
        self._reported = 0.0

    def update(
        self, frame: PitchFrame, onset: OnsetEvent | None, user_is_playing: bool
    ) -> IntensityUpdate | None:
        config = self._config
        if onset is not None:
            attack_dbfs = amplitude_to_dbfs(onset.strength)
            attack = _unit(attack_dbfs, config.attack_quiet_dbfs, config.attack_loud_dbfs)
            self._attacks.append((onset.frame_index, attack))
        while self._attacks and self._attacks[0][0] <= frame.frame_index - self._window_frames:
            self._attacks.popleft()

        level = _unit(frame.rms_dbfs, config.quiet_dbfs, config.loud_dbfs)
        attack = sum(a for _, a in self._attacks) / len(self._attacks) if self._attacks else 0.0
        window_seconds = config.window_seconds
        density = min(
            1.0, len(self._attacks) / window_seconds / config.full_density_onsets_per_second
        )
        weights = config.level_weight + config.attack_weight + config.density_weight
        target = (
            config.level_weight * level
            + config.attack_weight * attack
            + config.density_weight * density
        ) / weights
        if not user_is_playing:
            target = 0.0
        smoothing = self._rise if target > self.value else self._fall
        self.value += smoothing * (target - self.value)

        if abs(self.value - self._reported) < config.report_step:
            return None
        self._reported = self.value
        return IntensityUpdate(
            frame_index=frame.frame_index,
            timestamp_seconds=frames_to_seconds(frame.frame_index, self._sample_rate_hz),
            intensity=min(1.0, max(0.0, self.value)),
        )


class PlayingStateTracker:
    """The playing state (§70), decided in this order each hop:
    1. an attack within `hold_seconds`: PLAYING, or TOO_QUIET if that attack stayed below the
       level gate;
    2. a current note: SUSTAINING;
    3. above the level gate with an attack within `noise_seconds`: SUSTAINING (an unpitched
       chord ringing);
    4. above the level gate, no attack for `noise_seconds`, and the level steady within
       `noise_level_range_db` over that time: NOISE;
    5. otherwise STOPPED (including unidentified sound that is neither attacked nor steady).

    "Above the gate" has `gate_hysteresis_db` of hysteresis, so a level hovering at the gate
    doesn't flicker between SUSTAINING and STOPPED.
    """

    def __init__(self, config: AnalysisConfig, sample_rate_hz: int) -> None:
        self._config = config.playing
        self._gate_dbfs = config.level_gate_dbfs
        self._sample_rate_hz = sample_rate_hz
        self._hold_frames = seconds_to_frames(self._config.hold_seconds, sample_rate_hz)
        self._noise_frames = seconds_to_frames(self._config.noise_seconds, sample_rate_hz)
        self._noise_hops = max(
            1, round(self._config.noise_seconds * sample_rate_hz / config.hop_frames)
        )
        self._levels_dbfs: deque[float] = deque(maxlen=self._noise_hops)
        self._last_onset_frame: SampleFrame | None = None
        self._last_attack_dbfs = -math.inf
        self._above_gate = False
        self.state = PlayingState.STOPPED

    def update(
        self, frame: PitchFrame, onset: OnsetEvent | None, current_note: MidiNote | None
    ) -> PlayingStateUpdate | None:
        if onset is not None:
            self._last_onset_frame = onset.frame_index
            # The onset is reported one hop after its attack, so this frame's level is the
            # level just after the attack.
            self._last_attack_dbfs = frame.rms_dbfs
        threshold = self._gate_dbfs - (self._config.gate_hysteresis_db if self._above_gate else 0.0)
        self._above_gate = frame.rms_dbfs >= threshold
        self._levels_dbfs.append(frame.rms_dbfs)

        state = self._decide(frame.frame_index, current_note)
        if state == self.state:
            return None
        self.state = state
        return PlayingStateUpdate(
            frame_index=frame.frame_index,
            timestamp_seconds=frames_to_seconds(frame.frame_index, self._sample_rate_hz),
            state=state,
        )

    def _decide(self, frame_index: SampleFrame, current_note: MidiNote | None) -> PlayingState:
        since_onset = (
            math.inf if self._last_onset_frame is None else frame_index - self._last_onset_frame
        )
        if since_onset <= self._hold_frames:
            if self._last_attack_dbfs >= self._gate_dbfs:
                return PlayingState.PLAYING
            return PlayingState.TOO_QUIET
        if current_note is not None:
            return PlayingState.SUSTAINING
        if not self._above_gate:
            return PlayingState.STOPPED
        if since_onset <= self._noise_frames:
            return PlayingState.SUSTAINING
        steady = (
            len(self._levels_dbfs) == self._noise_hops
            and max(self._levels_dbfs) - min(self._levels_dbfs) <= self._config.noise_level_range_db
        )
        return PlayingState.NOISE if steady else PlayingState.STOPPED

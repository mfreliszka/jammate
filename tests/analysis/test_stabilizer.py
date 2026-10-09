import pytest

from jammate.analysis.notes import midi_to_hz
from jammate.analysis.pitch import PitchFrame
from jammate.analysis.stabilizer import NoteStabilizer
from jammate.config import AnalysisConfig
from jammate.music.events import PitchEvent, SilenceEvent

SR = 48_000
CONFIG = AnalysisConfig(instrument="guitar")  # 10 ms hops; 7 hops make a note
HOP = CONFIG.hop_frames
PLAYING_DBFS = -12.0


class Feed:
    """Feeds hand-made hops to a stabilizer and collects what it emits."""

    def __init__(self, config: AnalysisConfig = CONFIG) -> None:
        self.stabilizer = NoteStabilizer(config, SR)
        self.frame_index = 0
        self.events: list[PitchEvent | SilenceEvent] = []

    def add(
        self, midi: float | None, hops: int, rms_dbfs: float = PLAYING_DBFS
    ) -> list[PitchEvent | SilenceEvent]:
        """`hops` hops at fractional MIDI pitch `midi` (None = unvoiced). Returns new events."""
        new = []
        for _ in range(hops):
            self.frame_index += HOP
            frequency_hz = None if midi is None else 440.0 * 2 ** ((midi - 69) / 12)
            frame = PitchFrame(self.frame_index, frequency_hz, 0.95, rms_dbfs, rms_dbfs + 3)
            event = self.stabilizer.update(frame)
            if event is not None:
                new.append(event)
        self.events.extend(new)
        return new

    def glide(self, from_midi: float, to_midi: float, hops: int) -> list:
        new = []
        for step in range(1, hops + 1):
            new += self.add(from_midi + (to_midi - from_midi) * step / hops, 1)
        return new

    def notes(self) -> list[int | None]:
        return [e.midi_note if isinstance(e, PitchEvent) else None for e in self.events]


def test_a_steady_note_becomes_one_event_after_the_minimum_duration() -> None:
    feed = Feed()

    assert feed.add(40.0, 6) == []
    (event,) = feed.add(40.0, 1)
    assert feed.add(40.0, 50) == []

    assert isinstance(event, PitchEvent)
    assert event.midi_note == 40
    assert event.frame_index == HOP  # the first hop of the evidence
    assert event.timestamp_seconds == HOP / SR
    assert event.frequency_hz == pytest.approx(midi_to_hz(40))
    assert event.confidence == 0.95
    assert feed.stabilizer.current_note == 40


def test_brief_flicker_is_ignored() -> None:
    feed = Feed()

    feed.add(40.0, 10)
    feed.add(41.0, 3)  # E2 E2 F2 E2 (§26)
    feed.add(40.0, 10)

    assert feed.notes() == [40]


def test_hysteresis_keeps_a_sharp_note() -> None:
    feed = Feed()

    feed.add(40.0, 10)
    feed.add(40.62, 30)  # 62 cents sharp: nearer F2, but within the 70-cent hysteresis
    assert feed.notes() == [40]

    feed.add(41.0, 10)
    assert feed.notes() == [40, 41]


def test_no_note_is_committed_between_two_notes() -> None:
    feed = Feed()

    feed.add(62.56, 10)  # an attack 56 cents sharp of D4 (44 cents flat of D#4)
    assert feed.notes() == []

    feed.add(62.3, 10)  # settles 30 cents sharp
    assert feed.notes() == [62]


def test_a_slide_reports_only_its_start_and_end() -> None:
    feed = Feed()

    feed.add(43.0, 20)
    feed.glide(43.0, 45.0, 10)  # G2 to A2 in 100 ms, through G#2
    feed.add(45.0, 20)

    assert feed.notes() == [43, 45]


def test_a_short_stop_inside_a_slide_is_not_a_note() -> None:
    feed = Feed()

    feed.add(43.0, 20)
    feed.add(44.0, 6)  # 60 ms on G#2, as in the 92 BPM recording
    feed.add(45.0, 20)

    assert feed.notes() == [43, 45]


def test_an_octave_jump_needs_longer_confirmation() -> None:
    feed = Feed()

    feed.add(40.0, 20)
    feed.add(52.0, 10)  # 100 ms an octave up: an octave error
    feed.add(40.0, 10)
    assert feed.notes() == [40]

    feed.add(52.0, 14)
    assert feed.notes() == [40]
    (event,) = feed.add(52.0, 1)  # 150 ms: a real octave jump
    assert event.midi_note == 52
    assert event.frame_index == feed.frame_index - 14 * HOP


def test_silence_ends_the_note_only_after_the_hold() -> None:
    feed = Feed()

    feed.add(40.0, 20)
    feed.add(None, 5)  # 50 ms dropout between two picks of the same note
    feed.add(40.0, 20)
    assert feed.notes() == [40]

    silence_start = feed.frame_index + HOP
    assert feed.add(None, 6) == []
    (event,) = feed.add(None, 1)
    assert isinstance(event, SilenceEvent)
    assert event.frame_index == silence_start
    assert feed.stabilizer.current_note is None
    assert feed.add(None, 20) == []


def test_the_same_note_after_silence_is_a_new_event() -> None:
    feed = Feed()

    feed.add(40.0, 20)
    feed.add(None, 10)
    feed.add(40.0, 20)

    assert feed.notes() == [40, None, 40]


def test_repeated_picks_of_the_current_note_emit_nothing() -> None:
    feed = Feed()

    for _ in range(8):
        feed.add(40.0, 20)
        feed.add(None, 2)

    assert feed.notes() == [40]


def test_a_quiet_sound_never_ends_or_replaces_the_current_note() -> None:
    feed = Feed()

    feed.add(50.0, 20, rms_dbfs=-17.0)
    feed.add(74.0, 30, rms_dbfs=-34.0)  # 4th harmonic ringing 17 dB down after a mute

    assert feed.notes() == [50]
    assert feed.stabilizer.current_note == 50


def test_a_new_note_far_below_the_playing_level_waits_for_the_level_to_fall() -> None:
    feed = Feed()

    feed.add(40.0, 20, rms_dbfs=-10.0)
    feed.add(None, 10)
    feed.add(52.0, 50, rms_dbfs=-32.0)  # an open string ringing after release: 22 dB down
    assert feed.notes() == [40, None]

    # The playing level falls 3 dB/s; after about 2.4 s a steady quiet note is accepted.
    feed.add(52.0, 250, rms_dbfs=-32.0)
    assert feed.notes() == [40, None, 52]


def test_reset_drops_the_evidence_collected_so_far() -> None:
    feed = Feed()

    feed.add(40.0, 5)
    feed.stabilizer.reset()
    assert feed.add(40.0, 6) == []
    assert len(feed.add(40.0, 1)) == 1

from collections.abc import Callable

import pytest

from tests.fakes import FakeClock


def test_fake_clock_advances_only_when_told() -> None:
    clock = FakeClock()

    assert clock() == 0.0
    assert clock.frame_index == 0
    clock.advance(0.5)
    assert clock() == 0.5
    assert clock.seconds == 0.5
    assert clock.frame_index == 24_000
    clock.advance_frames(128)
    assert clock.frame_index == 24_128


def test_fake_clock_resolution_is_one_frame() -> None:
    clock = FakeClock(sample_rate_hz=48_000)

    clock.advance(1e-6)  # less than half a frame
    assert clock.frame_index == 0
    clock.advance(1 / 48_000)
    assert clock.frame_index == 1


def test_fake_clock_start_frame_and_sample_rate() -> None:
    clock = FakeClock(sample_rate_hz=44_100, start_frame=44_100)

    assert clock() == 1.0
    clock.advance(1.0)
    assert clock.frame_index == 88_200


def test_fake_clock_never_goes_backwards() -> None:
    clock = FakeClock()

    with pytest.raises(ValueError):
        clock.advance(-0.5)
    with pytest.raises(ValueError):
        clock.advance_frames(-1)
    with pytest.raises(ValueError):
        FakeClock(start_frame=-1)


def test_fake_clock_stands_in_for_monotonic_and_sleep() -> None:
    clock = FakeClock()

    def wait_until(
        deadline_seconds: float, sleep: Callable[[float], None], monotonic: Callable[[], float]
    ) -> int:
        polls = 0
        while monotonic() < deadline_seconds:
            sleep(0.1)
            polls += 1
        return polls

    assert wait_until(1.0, clock.advance, clock) == 10
    assert clock() == pytest.approx(1.0)

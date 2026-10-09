import math

import numpy as np
import pytest

from jammate.audio.levels import SILENCE_FLOOR_DBFS, InputLevel, amplitude_to_dbfs, measure_level


def test_amplitude_to_dbfs() -> None:
    assert amplitude_to_dbfs(1.0) == 0.0
    assert amplitude_to_dbfs(0.5) == pytest.approx(-6.02, abs=0.01)
    assert amplitude_to_dbfs(0.0) == SILENCE_FLOOR_DBFS
    assert amplitude_to_dbfs(1e-9) == SILENCE_FLOOR_DBFS


def test_measure_level_of_sine() -> None:
    samples = 0.5 * np.sin(2 * math.pi * 82.41 * np.arange(48_000) / 48_000).astype(np.float32)

    level = measure_level(samples)

    assert level.peak_dbfs == pytest.approx(-6.02, abs=0.05)
    assert level.rms_dbfs == pytest.approx(-9.03, abs=0.05)


def test_measure_level_of_nothing_is_silence() -> None:
    level = measure_level(np.zeros(0, dtype=np.float32))

    assert level == InputLevel(SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS)

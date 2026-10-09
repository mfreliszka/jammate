"""Signal level math shared by the level meter and the analysis level gate. Pure functions."""

import math
from dataclasses import dataclass

import numpy as np

SILENCE_FLOOR_DBFS = -120.0
CLIP_THRESHOLD_DBFS = -0.1


@dataclass(frozen=True)
class InputLevel:
    rms_dbfs: float
    peak_dbfs: float


def amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return SILENCE_FLOOR_DBFS
    return max(SILENCE_FLOOR_DBFS, 20.0 * math.log10(amplitude))


def measure_level(samples: np.ndarray) -> InputLevel:
    if samples.size == 0:
        return InputLevel(SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    return InputLevel(amplitude_to_dbfs(rms), amplitude_to_dbfs(peak))

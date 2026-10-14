"""Native C++20 realtime audio/DSP kernels with NumPy fallback (Step 7.1, §19, §87, §89, §151).

Loads `_jammate_native.dylib` via `ctypes.CDLL`, which automatically releases the Python GIL
around every C++ call. When the shared library is not present (or `use_native=False` is passed),
falls back transparently to the equivalent vectorized NumPy implementations.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jammate.analysis.pitch import YinResult, yin_lag_range

logger = logging.getLogger(__name__)

NATIVE_DIR = Path(__file__).resolve().parent
CPP_SOURCE_PATH = NATIVE_DIR / "_jammate_native.cpp"
DYLIB_PATH = NATIVE_DIR / "_jammate_native.dylib"

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)

_lib: ctypes.CDLL | None = None
_load_attempted: bool = False


def compile_native_extension(
    source_path: Path = CPP_SOURCE_PATH,
    output_path: Path = DYLIB_PATH,
) -> Path:
    """Compile `_jammate_native.cpp` into `_jammate_native.dylib` using `clang++ -std=c++20 -O3`."""
    cmd = [
        "clang++",
        "-std=c++20",
        "-O3",
        "-ffast-math",
        "-fPIC",
        "-dynamiclib",
        str(source_path),
        "-o",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.jammate_yin_cmndf_f32.argtypes = [
        _FLOAT_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        _FLOAT_PTR,
    ]
    lib.jammate_yin_cmndf_f32.restype = None

    lib.jammate_yin_cmndf_f64.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        _DOUBLE_PTR,
    ]
    lib.jammate_yin_cmndf_f64.restype = None

    lib.jammate_yin_search_dip_f64.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.jammate_yin_search_dip_f64.restype = None

    lib.jammate_mix_voice_f32.argtypes = [
        _FLOAT_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        _FLOAT_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.jammate_mix_voice_f32.restype = ctypes.c_int32

    lib.jammate_ring_write_f32.argtypes = [
        _FLOAT_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        _FLOAT_PTR,
        ctypes.c_int32,
    ]
    lib.jammate_ring_write_f32.restype = None

    lib.jammate_ring_copy_range_f32.argtypes = [
        _FLOAT_PTR,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        _FLOAT_PTR,
    ]
    lib.jammate_ring_copy_range_f32.restype = None


def get_native_lib(*, auto_compile: bool = True) -> ctypes.CDLL | None:
    """Return the loaded `_jammate_native` `ctypes.CDLL`, or `None` if unavailable."""
    global _lib, _load_attempted
    if _lib is not None:
        return _lib
    if _load_attempted and not auto_compile:
        return None
    _load_attempted = True

    if not DYLIB_PATH.is_file() and auto_compile and CPP_SOURCE_PATH.is_file():
        try:
            compile_native_extension(CPP_SOURCE_PATH, DYLIB_PATH)
        except Exception as exc:
            logger.debug("Could not auto-compile _jammate_native.dylib (%s); using NumPy.", exc)

    if not DYLIB_PATH.is_file():
        return None

    try:
        loaded = ctypes.CDLL(str(DYLIB_PATH))
        _bind_signatures(loaded)
        _lib = loaded
        return _lib
    except Exception as exc:
        logger.warning("Failed to load %s (%s); falling back to NumPy.", DYLIB_PATH, exc)
        return None


def native_available() -> bool:
    """Return True if the compiled C++20 extension `_jammate_native` is loaded and active."""
    return get_native_lib(auto_compile=True) is not None


def yin_cmndf_numpy(window: np.ndarray, tau_max: int) -> np.ndarray:
    """Pure-NumPy cumulative mean normalized difference function `d'(0..tau_max+1)`."""
    x = np.asarray(window, dtype=np.float64)
    integration_frames = len(x) - tau_max - 1
    if integration_frames < tau_max:
        raise ValueError(f"window of {len(x)} frames is too short for tau_max={tau_max}")

    fft_frames = 1 << (len(x) + integration_frames).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(x, fft_frames) * np.conj(np.fft.rfft(x[:integration_frames], fft_frames)),
        fft_frames,
    )[: tau_max + 2]
    cumulative_energy = np.concatenate(([0.0], np.cumsum(x * x)))
    lags = np.arange(tau_max + 2)
    shifted_energy = cumulative_energy[lags + integration_frames] - cumulative_energy[lags]
    difference = np.maximum(shifted_energy[0] + shifted_energy - 2.0 * correlation, 0.0)

    running_sum = np.cumsum(difference[1:])
    normalized = np.ones(tau_max + 2, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized[1:] = np.where(
            running_sum > 0, (difference[1:] * lags[1:]) / running_sum, 1.0
        ).astype(np.float32)
    return normalized


def yin_cmndf_f32(
    window: np.ndarray,
    tau_max: int,
    *,
    use_native: bool = True,
) -> np.ndarray:
    """Compute the YIN CMNDF array `d'(0..tau_max+1)` as `float32` (C++20 or NumPy fallback)."""
    integration_frames = len(window) - tau_max - 1
    if integration_frames < tau_max:
        raise ValueError(f"window of {len(window)} frames is too short for tau_max={tau_max}")

    lib = get_native_lib() if use_native else None
    if lib is None:
        return yin_cmndf_numpy(window, tau_max)

    x_f32 = np.ascontiguousarray(window, dtype=np.float32)
    out = np.empty(tau_max + 2, dtype=np.float32)
    lib.jammate_yin_cmndf_f32(
        x_f32.ctypes.data_as(_FLOAT_PTR),
        len(x_f32),
        integration_frames,
        tau_max,
        out.ctypes.data_as(_FLOAT_PTR),
    )
    return out


def yin_estimate(
    window: np.ndarray,
    sample_rate_hz: int,
    min_hz: float,
    max_hz: float,
    threshold: float,
    *,
    use_native: bool = True,
) -> YinResult:
    """Estimate fundamental frequency and confidence using C++20 YIN or the NumPy fallback."""
    tau_min, tau_max = yin_lag_range(sample_rate_hz, min_hz, max_hz)
    integration_frames = len(window) - tau_max - 1
    if integration_frames < tau_max:
        raise ValueError(f"window of {len(window)} frames is too short for {min_hz} Hz")

    lib = get_native_lib() if use_native else None
    if lib is None:
        from jammate.analysis.pitch import yin as yin_numpy

        return yin_numpy(window, sample_rate_hz, min_hz, max_hz, threshold)

    x_f64 = np.ascontiguousarray(window, dtype=np.float64)
    cmndf = np.empty(tau_max + 2, dtype=np.float64)
    lib.jammate_yin_cmndf_f64(
        x_f64.ctypes.data_as(_DOUBLE_PTR),
        len(x_f64),
        integration_frames,
        tau_max,
        cmndf.ctypes.data_as(_DOUBLE_PTR),
    )
    period = ctypes.c_double(0.0)
    confidence = ctypes.c_double(0.0)
    lib.jammate_yin_search_dip_f64(
        cmndf.ctypes.data_as(_DOUBLE_PTR),
        tau_min,
        tau_max,
        float(threshold),
        ctypes.byref(period),
        ctypes.byref(confidence),
    )
    if period.value <= 0.0:
        return YinResult(None, float(confidence.value))
    return YinResult(sample_rate_hz / float(period.value), float(confidence.value))


def mix_drum_voices_f32(
    samples: Sequence[np.ndarray],
    active: list[bool],
    piece_indices: Sequence[int],
    gains: Sequence[float],
    sample_positions: list[int],
    out: np.ndarray,
    start_frame: int,
    end_frame: int,
    *,
    use_native: bool = True,
    mono_scratch: np.ndarray | None = None,
) -> None:
    """Mix active drum voices into `out[start_frame:end_frame]` in place with zero allocations."""
    slice_frames = end_frame - start_frame
    if slice_frames <= 0:
        return

    lib = (_lib or get_native_lib(auto_compile=False)) if use_native else None
    channels = 1 if out.ndim == 1 else int(out.shape[1])

    if lib is not None and out.flags.c_contiguous and out.dtype == np.float32:
        out_ptr = out.ctypes.data_as(_FLOAT_PTR)
        for i, is_active in enumerate(active):
            if not is_active:
                continue
            sample = samples[piece_indices[i]]
            sample_len = int(sample.shape[0])
            new_pos = int(
                lib.jammate_mix_voice_f32(
                    sample.ctypes.data_as(_FLOAT_PTR),
                    sample_len,
                    int(sample_positions[i]),
                    float(gains[i]),
                    out_ptr,
                    int(start_frame),
                    int(end_frame),
                    channels,
                )
            )
            sample_positions[i] = new_pos
            if new_pos >= sample_len:
                active[i] = False
        return

    # Pure-NumPy fallback
    is_2d = out.ndim == 2
    for i, is_active in enumerate(active):
        if not is_active:
            continue
        pos = sample_positions[i]
        sample = samples[piece_indices[i]]
        sample_len = sample.shape[0]
        n = min(slice_frames, sample_len - pos)
        if n > 0:
            if mono_scratch is not None and len(mono_scratch) >= n:
                scratch = mono_scratch[:n]
                np.multiply(sample[pos : pos + n], gains[i], out=scratch)
            else:
                scratch = sample[pos : pos + n] * gains[i]
            target = out[start_frame : start_frame + n]
            if is_2d:
                np.add(target, scratch[:, np.newaxis], out=target)
            else:
                np.add(target, scratch, out=target)
        new_pos = pos + n
        sample_positions[i] = new_pos
        if new_pos >= sample_len:
            active[i] = False


def ring_write_f32(
    ring_buf: np.ndarray,
    write_pos: int,
    frames: np.ndarray,
    *,
    use_native: bool = True,
) -> None:
    """Write `frames` `(frame_count, channels)` into `ring_buf` at `write_pos`."""
    frame_count = int(frames.shape[0])
    if frame_count <= 0:
        return
    capacity = int(ring_buf.shape[0])
    channels = 1 if ring_buf.ndim == 1 else int(ring_buf.shape[1])
    lib = (_lib or get_native_lib(auto_compile=False)) if use_native else None
    if (
        lib is not None
        and ring_buf.flags.c_contiguous
        and frames.flags.c_contiguous
        and ring_buf.dtype == np.float32
        and frames.dtype == np.float32
    ):
        lib.jammate_ring_write_f32(
            ring_buf.ctypes.data_as(_FLOAT_PTR),
            capacity,
            channels,
            int(write_pos),
            frames.ctypes.data_as(_FLOAT_PTR),
            frame_count,
        )
        return

    start = write_pos % capacity
    first = min(frame_count, capacity - start)
    ring_buf[start : start + first] = frames[:first]
    if first < frame_count:
        ring_buf[: frame_count - first] = frames[first:]


def ring_copy_range_f32(
    ring_buf: np.ndarray,
    start_pos: int,
    frame_count: int,
    out: np.ndarray,
    *,
    use_native: bool = True,
) -> None:
    """Copy `frame_count` frames starting at `start_pos` from `ring_buf` into `out`."""
    if frame_count <= 0:
        return
    capacity = int(ring_buf.shape[0])
    channels = 1 if ring_buf.ndim == 1 else int(ring_buf.shape[1])
    lib = (_lib or get_native_lib(auto_compile=False)) if use_native else None
    if (
        lib is not None
        and ring_buf.flags.c_contiguous
        and out.flags.c_contiguous
        and ring_buf.dtype == np.float32
        and out.dtype == np.float32
    ):
        lib.jammate_ring_copy_range_f32(
            ring_buf.ctypes.data_as(_FLOAT_PTR),
            capacity,
            channels,
            int(start_pos),
            int(frame_count),
            out.ctypes.data_as(_FLOAT_PTR),
        )
        return

    offset = start_pos % capacity
    first = min(frame_count, capacity - offset)
    out[:first] = ring_buf[offset : offset + first]
    if first < frame_count:
        out[first:frame_count] = ring_buf[: frame_count - first]

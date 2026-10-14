"""Native C++20 realtime audio/DSP acceleration with pure-Python/NumPy fallback (Step 7.1, §151)."""

from jammate.native.kernels import (
    CPP_SOURCE_PATH,
    DYLIB_PATH,
    compile_native_extension,
    get_native_lib,
    mix_drum_voices_f32,
    native_available,
    ring_copy_range_f32,
    ring_write_f32,
    yin_cmndf_f32,
    yin_cmndf_numpy,
    yin_estimate,
)

__all__ = [
    "CPP_SOURCE_PATH",
    "DYLIB_PATH",
    "compile_native_extension",
    "get_native_lib",
    "mix_drum_voices_f32",
    "native_available",
    "ring_copy_range_f32",
    "ring_write_f32",
    "yin_cmndf_f32",
    "yin_cmndf_numpy",
    "yin_estimate",
]

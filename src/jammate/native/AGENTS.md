# native/ — C++20 DSP & realtime callback acceleration

Isolates hot inner loops behind a compiled C++20 library (`_jammate_native.cpp` →
`_jammate_native.dylib`) loaded via `ctypes.CDLL`, with automatic pure-NumPy fallback when not
compiled (Step 7.1, §19, §87, §89, §151).

| Module | Status | Thread |
|---|---|---|
| `_jammate_native.cpp` | done (7.1) | C++20 kernels (`yin_cmndf`, `mix_voice_f32`, `ring_write_f32`, `ring_copy_range_f32`) |
| `kernels.py` | done (7.1) | `yin_cmndf_f32` / `yin_estimate` on analysis worker; `mix_drum_voices_f32` / `ring_*` on callback |

## Invariants

- Every `ctypes.CDLL` call automatically releases the Python GIL around C++ execution.
- All C++ kernels operate in-place on caller-owned contiguous float32/float64 buffers with zero
  heap allocations and zero locks.
- Every kernel has an exact NumPy fallback (`use_native=False`) tested for `< 1e-6` numerical
  parity in `tests/native/test_native.py`.

// C++20 realtime audio/DSP kernels for JamMate (Step 7.1, §19, §87, §89, §151).
// Compiled into `_jammate_native.dylib` on macOS Apple Silicon (`clang++ -std=c++20 -O3 -dynamiclib`).
// Loaded via `ctypes.CDLL`, which automatically releases the Python GIL during every call.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {

// Compute the YIN cumulative mean normalized difference function d'(tau) for lags 0..tau_max+1
// over `integration_frames` samples of `x` (length >= integration_frames + tau_max + 2).
void jammate_yin_cmndf_f32(
    const float* __restrict x,
    int32_t window_len,
    int32_t integration_frames,
    int32_t tau_max,
    float* __restrict out_cmndf
) {
    if (window_len <= 0 || integration_frames <= 0 || tau_max < 1) {
        return;
    }
    const int32_t max_lag = tau_max + 1;
    out_cmndf[0] = 1.0f;
    double running_sum = 0.0;

    for (int32_t tau = 1; tau <= max_lag; ++tau) {
        const float* __restrict a = x;
        const float* __restrict b = x + tau;
        double diff = 0.0;
        for (int32_t j = 0; j < integration_frames; ++j) {
            const double d = static_cast<double>(a[j]) - static_cast<double>(b[j]);
            diff += d * d;
        }
        running_sum += diff;
        if (running_sum > 0.0) {
            out_cmndf[tau] = static_cast<float>((diff * static_cast<double>(tau)) / running_sum);
        } else {
            out_cmndf[tau] = 1.0f;
        }
    }
}

void jammate_yin_cmndf_f64(
    const double* __restrict x,
    int32_t window_len,
    int32_t integration_frames,
    int32_t tau_max,
    double* __restrict out_cmndf
) {
    if (window_len <= 0 || integration_frames <= 0 || tau_max < 1) {
        return;
    }
    const int32_t max_lag = tau_max + 1;
    out_cmndf[0] = 1.0;
    double running_sum = 0.0;

    for (int32_t tau = 1; tau <= max_lag; ++tau) {
        const double* __restrict a = x;
        const double* __restrict b = x + tau;
        double diff = 0.0;
        for (int32_t j = 0; j < integration_frames; ++j) {
            const double d = a[j] - b[j];
            diff += d * d;
        }
        running_sum += diff;
        if (running_sum > 0.0) {
            out_cmndf[tau] = (diff * static_cast<double>(tau)) / running_sum;
        } else {
            out_cmndf[tau] = 1.0;
        }
    }
}

// Search `cmndf[tau_min..tau_max]` for the first dip below `threshold` and refine with parabolic
// interpolation. Writes `*out_period_frames` (<= 0 if unvoiced) and `*out_confidence` (0..1).
void jammate_yin_search_dip_f64(
    const double* __restrict cmndf,
    int32_t tau_min,
    int32_t tau_max,
    double threshold,
    double* __restrict out_period_frames,
    double* __restrict out_confidence
) {
    int32_t first_below = -1;
    int32_t best_tau = tau_min;
    double best_val = cmndf[tau_min];

    for (int32_t tau = tau_min; tau <= tau_max; ++tau) {
        const double val = cmndf[tau];
        if (val < best_val) {
            best_val = val;
            best_tau = tau;
        }
        if (first_below < 0 && val < threshold) {
            first_below = tau;
        }
    }

    if (first_below < 0) {
        *out_period_frames = -1.0;
        *out_confidence = std::clamp(1.0 - best_val, 0.0, 1.0);
        return;
    }

    int32_t tau = first_below;
    while (tau < tau_max && cmndf[tau + 1] < cmndf[tau]) {
        ++tau;
    }

    const double before = cmndf[tau - 1];
    const double at = cmndf[tau];
    const double after = cmndf[tau + 1];
    const double curvature = before - 2.0 * at + after;
    double offset = 0.0;
    if (curvature > 0.0) {
        offset = std::clamp(0.5 * (before - after) / curvature, -0.5, 0.5);
    }
    *out_period_frames = static_cast<double>(tau) + offset;
    *out_confidence = std::clamp(1.0 - at, 0.0, 1.0);
}

// Mix one active drum voice into `out` (1D mono or 2D interleaved `channels` float32) over
// `[start_frame, end_frame)` and return the new sample position.
int32_t jammate_mix_voice_f32(
    const float* __restrict sample,
    int32_t sample_len,
    int32_t sample_pos,
    float gain,
    float* __restrict out,
    int32_t start_frame,
    int32_t end_frame,
    int32_t channels
) {
    const int32_t slice_frames = end_frame - start_frame;
    if (slice_frames <= 0 || sample_pos >= sample_len) {
        return sample_pos;
    }
    const int32_t remaining = sample_len - sample_pos;
    const int32_t n = std::min(slice_frames, remaining);
    const float* __restrict src = sample + sample_pos;

    if (channels == 1) {
        float* __restrict dst = out + start_frame;
        for (int32_t i = 0; i < n; ++i) {
            dst[i] += src[i] * gain;
        }
    } else if (channels == 2) {
        float* __restrict dst = out + (start_frame * 2);
        for (int32_t i = 0; i < n; ++i) {
            const float v = src[i] * gain;
            dst[2 * i] += v;
            dst[2 * i + 1] += v;
        }
    } else {
        float* __restrict dst = out + (start_frame * channels);
        for (int32_t i = 0; i < n; ++i) {
            const float v = src[i] * gain;
            for (int32_t c = 0; c < channels; ++c) {
                dst[i * channels + c] += v;
            }
        }
    }
    return sample_pos + n;
}

// Lock-free SPSC ring buffer slice write with wrap-around.
void jammate_ring_write_f32(
    float* __restrict ring_buf,
    int32_t capacity_frames,
    int32_t channels,
    int64_t write_pos,
    const float* __restrict src_frames,
    int32_t frame_count
) {
    if (frame_count <= 0 || capacity_frames <= 0 || channels <= 0) {
        return;
    }
    const int32_t start = static_cast<int32_t>(write_pos % capacity_frames);
    const int32_t first = std::min(frame_count, capacity_frames - start);
    const size_t bytes_per_frame = static_cast<size_t>(channels) * sizeof(float);

    std::memcpy(
        ring_buf + static_cast<size_t>(start) * channels,
        src_frames,
        static_cast<size_t>(first) * bytes_per_frame
    );
    if (first < frame_count) {
        std::memcpy(
            ring_buf,
            src_frames + static_cast<size_t>(first) * channels,
            static_cast<size_t>(frame_count - first) * bytes_per_frame
        );
    }
}

// Lock-free SPSC ring buffer slice read with wrap-around.
void jammate_ring_copy_range_f32(
    const float* __restrict ring_buf,
    int32_t capacity_frames,
    int32_t channels,
    int64_t start_pos,
    int32_t frame_count,
    float* __restrict dst_frames
) {
    if (frame_count <= 0 || capacity_frames <= 0 || channels <= 0) {
        return;
    }
    const int32_t offset = static_cast<int32_t>(start_pos % capacity_frames);
    const int32_t first = std::min(frame_count, capacity_frames - offset);
    const size_t bytes_per_frame = static_cast<size_t>(channels) * sizeof(float);

    std::memcpy(
        dst_frames,
        ring_buf + static_cast<size_t>(offset) * channels,
        static_cast<size_t>(first) * bytes_per_frame
    );
    if (first < frame_count) {
        std::memcpy(
            dst_frames + static_cast<size_t>(first) * channels,
            ring_buf,
            static_cast<size_t>(frame_count - first) * bytes_per_frame
        );
    }
}

}  // extern "C"

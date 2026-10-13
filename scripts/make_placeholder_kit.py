"""Generate license-safe synthetic drum kit WAVs into `assets/drums/` (§45, §135).

Synthesizes five mono 48 kHz WAVs (`kick.wav`, `snare.wav`, `closed_hat.wav`, `crash.wav`,
`tom.wav`) deterministically from pitched sine sweeps, inharmonic partials and filtered noise
with smooth attack/decay envelopes so every sample starts and ends at 0.0 (click-free).

Example:
    uv run python scripts/make_placeholder_kit.py
    uv run python scripts/make_placeholder_kit.py --output-dir assets/drums --overwrite
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "drums"
DEFAULT_SAMPLE_RATE_HZ = 48_000
KIT_PIECES = ("kick", "snare", "closed_hat", "crash", "tom")


def _envelope(
    sample_count: int,
    sample_rate_hz: int,
    attack_seconds: float,
    decay_tau_seconds: float,
) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    attack_samples = max(1, round(attack_seconds * sample_rate_hz))
    attack = np.ones(sample_count, dtype=np.float64)
    attack[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
    decay = np.exp(-t / decay_tau_seconds)
    # Cosine tail over the last 5 ms guarantees an exact zero ending.
    tail_samples = min(sample_count // 4, max(1, round(0.005 * sample_rate_hz)))
    tail = np.ones(sample_count, dtype=np.float64)
    tail[-tail_samples:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tail_samples)))
    return attack * decay * tail


def _normalize(wave: np.ndarray, peak: float) -> np.ndarray:
    max_abs = float(np.max(np.abs(wave)))
    if max_abs > 0.0:
        wave = wave * (peak / max_abs)
    wave[0] = 0.0
    wave[-1] = 0.0
    return wave.astype(np.float32)


def synthesize_kick(sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> np.ndarray:
    duration_seconds = 0.35
    n = round(duration_seconds * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    # Exponential frequency sweep from 155 Hz down to 48 Hz.
    f_start, f_end, sweep_tau = 155.0, 48.0, 0.035
    freq = f_end + (f_start - f_end) * np.exp(-t / sweep_tau)
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate_hz
    body = np.sin(phase) * _envelope(n, sample_rate_hz, 0.0015, 0.095)
    click = np.sin(2.0 * np.pi * 950.0 * t) * _envelope(n, sample_rate_hz, 0.0005, 0.004)
    return _normalize(body + 0.25 * click, peak=0.85)


def synthesize_snare(sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ, seed: int = 1) -> np.ndarray:
    duration_seconds = 0.25
    n = round(duration_seconds * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)

    body = (
        0.65 * np.sin(2.0 * np.pi * 185.0 * t) + 0.35 * np.sin(2.0 * np.pi * 330.0 * t)
    ) * _envelope(n, sample_rate_hz, 0.001, 0.035)
    sos = signal.butter(4, [900.0, 9_000.0], btype="bandpass", fs=sample_rate_hz, output="sos")
    wires = signal.sosfilt(sos, rng.standard_normal(n)) * _envelope(n, sample_rate_hz, 0.001, 0.065)
    return _normalize(0.55 * body + 0.85 * wires, peak=0.80)


def synthesize_closed_hat(
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ, seed: int = 2
) -> np.ndarray:
    duration_seconds = 0.08
    n = round(duration_seconds * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)

    ratios = (1.0, 1.34, 1.67, 1.92, 2.51, 3.17)
    base_hz = 3_400.0
    metal = sum(np.sign(np.sin(2.0 * np.pi * base_hz * r * t)) for r in ratios) / len(ratios)
    sos = signal.butter(4, 6_500.0, btype="highpass", fs=sample_rate_hz, output="sos")
    hiss = signal.sosfilt(sos, rng.standard_normal(n) + 0.5 * metal)
    env = _envelope(n, sample_rate_hz, 0.0008, 0.016)
    return _normalize(hiss * env, peak=0.60)


def synthesize_crash(sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ, seed: int = 3) -> np.ndarray:
    duration_seconds = 1.10
    n = round(duration_seconds * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)

    partials_hz = (512.0, 788.0, 1_195.0, 1_710.0, 2_480.0, 3_620.0, 5_190.0, 7_430.0)
    shimmer = sum(
        np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi)) for f in partials_hz
    ) / len(partials_hz)
    sos = signal.butter(3, [1_800.0, 12_000.0], btype="bandpass", fs=sample_rate_hz, output="sos")
    wash = signal.sosfilt(sos, rng.standard_normal(n) + 0.45 * shimmer)
    env = _envelope(n, sample_rate_hz, 0.002, 0.28)
    return _normalize(wash * env, peak=0.70)


def synthesize_tom(sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> np.ndarray:
    duration_seconds = 0.30
    n = round(duration_seconds * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    f_start, f_end, sweep_tau = 190.0, 110.0, 0.045
    freq = f_end + (f_start - f_end) * np.exp(-t / sweep_tau)
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate_hz
    body = np.sin(phase) * _envelope(n, sample_rate_hz, 0.001, 0.085)
    overtone = 0.2 * np.sin(1.58 * phase) * _envelope(n, sample_rate_hz, 0.001, 0.03)
    return _normalize(body + overtone, peak=0.80)


def generate_kit(sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> dict[str, np.ndarray]:
    """Return all five placeholder kit pieces as 1D float32 arrays at `sample_rate_hz`."""
    return {
        "kick": synthesize_kick(sample_rate_hz),
        "snare": synthesize_snare(sample_rate_hz),
        "closed_hat": synthesize_closed_hat(sample_rate_hz),
        "crash": synthesize_crash(sample_rate_hz),
        "tom": synthesize_tom(sample_rate_hz),
    }


def write_kit(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    overwrite: bool = False,
) -> list[Path]:
    """Write the placeholder drum kit WAVs into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kit = generate_kit(sample_rate_hz)
    written: list[Path] = []
    for name, samples in kit.items():
        path = output_dir / f"{name}.wav"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists (pass --overwrite to replace it)")
        sf.write(path, samples, sample_rate_hz, subtype="FLOAT")
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate license-safe placeholder drum kit WAVs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for WAV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"sample rate in Hz (default: {DEFAULT_SAMPLE_RATE_HZ})",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing WAV files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = write_kit(args.output_dir, args.sample_rate, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        info = sf.info(path)
        print(f"wrote {path} ({info.frames} frames, {info.samplerate} Hz, {info.duration:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

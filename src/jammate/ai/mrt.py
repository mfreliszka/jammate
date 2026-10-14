"""Magenta RealTime 2 (mrt2_small, MLX) in-process accompaniment adapter (Step 4.2, D1, §§50-52)."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np
from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

from jammate.ai.base import (
    AccompanimentEngine,
    AccompanimentStats,
    AccompanimentStatus,
    NoteTracker,
)
from jammate.ai.null import NullAccompaniment
from jammate.audio.ring_buffer import RingBuffer
from jammate.config import AccompanimentConfig
from jammate.music.director import BandDirective
from jammate.music.events import MidiNote, seconds_to_frames
from jammate.music.styles import STYLE_PRESETS, StylePreset, get_style_preset, map_follow_improvise

logger = logging.getLogger(__name__)

MRT2_SAMPLE_RATE_HZ = 48_000
MRT2_SAMPLES_PER_FRAME = 1_920  # 40 ms at 48 kHz
MRT2_CHANNELS = 2
WORKER_POLL_SECONDS = 0.008


class MrtWaveformProtocol(Protocol):
    """Minimal structural interface of `magenta_rt.audio.Waveform`."""

    @property
    def samples(self) -> np.ndarray: ...

    @property
    def sample_rate(self) -> int: ...


class MrtSystemProtocol(Protocol):
    """Minimal structural interface of `magenta_rt.mlx.system.MagentaRT2System`."""

    def embed_style(self, text_or_audio: str) -> Any: ...

    def generate(
        self,
        conditioning: dict[str, list[int] | np.ndarray] | None = None,
        cfg_scales: dict[str, float] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        frames: int = 25,
        state: Any = None,
    ) -> tuple[MrtWaveformProtocol, Any]: ...


_mlx_lock = threading.Lock()
_mlx_executor: ThreadPoolExecutor | None = None
_mlx_system_cache: dict[str, MrtSystemProtocol] = {}


def _get_mlx_executor() -> ThreadPoolExecutor:
    global _mlx_executor
    with _mlx_lock:
        if _mlx_executor is None:
            _mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jammate-mlx")
        return _mlx_executor


def shutdown_mlx_runtime() -> None:
    """Shut down and join the dedicated MLX runtime thread on application exit (§84)."""
    global _mlx_executor
    with _mlx_lock:
        executor = _mlx_executor
        _mlx_executor = None
        _mlx_system_cache.clear()
    if executor is not None:
        executor.shutdown(wait=True)


atexit.register(shutdown_mlx_runtime)


class _ThreadBoundMrtSystem:
    """Dispatches all MLX calls onto the single `jammate-mlx` thread owning `Stream(gpu, 1)`."""

    def __init__(self, executor: ThreadPoolExecutor, inner: Any) -> None:
        self._executor = executor
        self._inner = inner

    def embed_style(self, text_or_audio: str) -> Any:
        return self._executor.submit(self._inner.embed_style, text_or_audio).result()

    def tokenize_style(self, embedding: Any) -> Any:
        return self._executor.submit(self._inner.tokenize_style, embedding).result()

    def generate(
        self,
        conditioning: dict[str, list[int] | np.ndarray] | None = None,
        cfg_scales: dict[str, float] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        frames: int = 25,
        state: Any = None,
    ) -> tuple[MrtWaveformProtocol, Any]:
        return self._executor.submit(
            self._inner.generate,
            conditioning=conditioning,
            cfg_scales=cfg_scales,
            temperature=temperature,
            top_k=top_k,
            frames=frames,
            state=state,
        ).result()


def _load_mrt2_system_on_mlx_thread(model_size: str) -> Any:
    from magenta_rt import paths
    from magenta_rt.mlx.system import MagentaRT2System, MagentaRT2SystemStdMlxfn

    mlxfn_path = paths.models_dir() / model_size / f"{model_size}.mlxfn"
    if mlxfn_path.is_file():
        return MagentaRT2SystemStdMlxfn(size=model_size)
    return MagentaRT2System(size=model_size)


def default_mrt2_system_factory(model_size: str = "mrt2_small") -> MrtSystemProtocol:
    """Instantiate and warm up `mrt2_small` on the dedicated MLX thread (§83)."""
    with _mlx_lock:
        cached = _mlx_system_cache.get(model_size)
        if cached is not None:
            return cached
    executor = _get_mlx_executor()
    inner = executor.submit(_load_mrt2_system_on_mlx_thread, model_size).result()
    bound = _ThreadBoundMrtSystem(executor, inner)
    with _mlx_lock:
        _mlx_system_cache[model_size] = bound
    return bound


class Mrt2Accompaniment:
    """In-process `mrt2_small` accompaniment engine with background chunk generation (Step 4.2)."""

    def __init__(
        self,
        config: AccompanimentConfig,
        sample_rate_hz: int = MRT2_SAMPLE_RATE_HZ,
        *,
        system_factory: Callable[[str], MrtSystemProtocol] = default_mrt2_system_factory,
        preload_presets: bool = True,
    ) -> None:
        if sample_rate_hz != MRT2_SAMPLE_RATE_HZ:
            raise ValueError(
                f"Mrt2Accompaniment requires {MRT2_SAMPLE_RATE_HZ} Hz, got {sample_rate_hz} Hz"
            )
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        capacity_frames = max(
            config.chunk_frames * MRT2_SAMPLES_PER_FRAME * 2,
            seconds_to_frames(config.ring_capacity_seconds, sample_rate_hz),
        )
        self._lead_frames = max(
            MRT2_SAMPLES_PER_FRAME,
            seconds_to_frames(config.target_lead_seconds, sample_rate_hz),
        )
        self._output_ring = RingBuffer(capacity_frames=capacity_frames, channels=MRT2_CHANNELS)
        self._note_tracker = NoteTracker()
        self._style: str = config.style
        self._intensity: float = 0.5
        self._follow_improvise: float = config.follow_improvise
        self._effective_follow_improvise: float = config.follow_improvise
        self._gate_gain: float = 1.0
        self._ai_gain_multiplier: float = 1.0
        self._status = AccompanimentStatus.LOADING
        self._last_error: str | None = None
        self._ms_per_frame: float = 0.0
        self._frames_generated: int = 0
        self._stream_state: Any = None
        self._style_cache: dict[str, list[int] | np.ndarray] = {}
        self._gen_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        t0 = time.perf_counter()
        self._system = system_factory(config.model_size)
        if preload_presets:
            for preset in STYLE_PRESETS.values():
                self._get_or_embed_style(preset.name)
        else:
            self._get_or_embed_style(self._style)
        self._status = AccompanimentStatus.READY
        logger.info(
            "MRT2 (%s) ready in %.2f s (%d style embeddings cached).",
            config.model_size,
            time.perf_counter() - t0,
            len(self._style_cache),
        )

    @property
    def output_ring(self) -> RingBuffer | None:
        if self._status == AccompanimentStatus.FAILED:
            return None
        return self._output_ring

    @property
    def status(self) -> AccompanimentStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def gate_gain(self) -> float:
        return self._gate_gain

    @property
    def ai_gain_multiplier(self) -> float:
        return self._ai_gain_multiplier

    @property
    def effective_follow_improvise(self) -> float:
        return self._effective_follow_improvise

    def set_gate_gain(self, gain: float) -> None:
        """Set the musical entry/silence gate gain (`0.0..1.0`, Step 4.3)."""
        self._gate_gain = max(0.0, min(1.0, float(gain)))

    def apply_directive(self, directive: BandDirective) -> None:
        """Apply a `BandDirective` from `BandDirector` (Step 5.5, §148)."""
        self._effective_follow_improvise = max(
            0.0, min(1.0, float(directive.effective_follow_improvise))
        )
        self._ai_gain_multiplier = max(0.0, min(1.0, float(directive.ai_gain_multiplier)))
        self._note_tracker.set_harmonic_anchor(
            directive.target_key_pc,
            user_is_playing=directive.ai_role != "silent",
        )

    def start(self) -> None:
        if self._status == AccompanimentStatus.FAILED or self.is_running:
            return
        self._stop_event.clear()
        self._status = AccompanimentStatus.RUNNING
        self._thread = threading.Thread(target=self._run, name="jammate-ai", daemon=False)
        self._thread.start()
        logger.info(
            "MRT2 worker started: style=%s, chunk=%d frames (%.0f ms), target lead=%.0f ms.",
            self._style,
            self._config.chunk_frames,
            self._config.chunk_frames * 40.0,
            self._config.target_lead_seconds * 1000.0,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._note_tracker.all_notes_off()
        if self._status != AccompanimentStatus.FAILED:
            self._status = AccompanimentStatus.STOPPED
        logger.info(
            "MRT2 worker stopped (%d frames generated, %.1f ms/frame).",
            self._frames_generated,
            self._ms_per_frame,
        )

    def set_style(self, style: str) -> None:
        self._get_or_embed_style(style)
        self._style = style

    def note_on(self, midi_note: MidiNote) -> None:
        self._note_tracker.on_pitch(midi_note)

    def note_off(self, midi_note: MidiNote) -> None:
        if self._note_tracker.active_note == midi_note:
            self._note_tracker.on_silence()

    def set_intensity(self, intensity: float) -> None:
        self._intensity = max(0.0, min(1.0, float(intensity)))

    def set_follow_improvise(self, follow_improvise: float) -> None:
        if not 0.0 <= follow_improvise <= 1.0:
            raise ValueError("follow_improvise must be between 0.0 and 1.0")
        self._follow_improvise = float(follow_improvise)
        self._effective_follow_improvise = float(follow_improvise)

    def stats(self) -> AccompanimentStats:
        return AccompanimentStats(
            status=self._status,
            style=self._style,
            active_note=self._note_tracker.active_note,
            intensity=self._intensity,
            follow_improvise=self._effective_follow_improvise,
            buffer_seconds=self._output_ring.available_frames() / self._sample_rate_hz,
            ms_per_frame=self._ms_per_frame,
            frames_generated=self._frames_generated,
            last_error=self._last_error,
        )

    def generate_chunk(self, frames: int | None = None) -> np.ndarray:
        """Generate one chunk of `frames` (defaults to `config.chunk_frames`) and push to ring."""
        chunk_frames = self._config.chunk_frames if frames is None else frames
        style_key = self._style
        style_cond = self._get_or_embed_style(style_key)
        pianoroll = self._note_tracker.consume_pianoroll()
        drum_cond = [self._config.drum_token]

        preset = self._resolve_preset(style_key)
        mapping = map_follow_improvise(preset, self._effective_follow_improvise, self._intensity)
        conditioning: dict[str, list[int] | np.ndarray] = {
            MUSICCOCA.key: style_cond,
            PIANOROLL_WITH_ONSETS.key: pianoroll,
            DRUM_PIANOROLL.key: drum_cond,
        }

        t0 = time.perf_counter()
        with self._gen_lock:
            waveform, self._stream_state = self._system.generate(
                conditioning=conditioning,
                cfg_scales=mapping.cfg_scales(),
                temperature=mapping.temperature,
                top_k=mapping.top_k,
                frames=chunk_frames,
                state=self._stream_state,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        step_ms = elapsed_ms / max(1, chunk_frames)
        if self._frames_generated == 0:
            self._ms_per_frame = step_ms
        else:
            self._ms_per_frame = 0.8 * self._ms_per_frame + 0.2 * step_ms
        self._frames_generated += chunk_frames

        samples = np.ascontiguousarray(waveform.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = np.column_stack((samples, samples))
        elif samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)

        effective_gain = self._gate_gain * self._ai_gain_multiplier
        if effective_gain < 1.0:
            samples = (
                np.zeros_like(samples)
                if effective_gain <= 0.0
                else samples * np.float32(effective_gain)
            )

        self._output_ring.write(samples)
        return samples

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._output_ring.available_frames() < self._lead_frames:
                    self.generate_chunk()
                else:
                    self._stop_event.wait(WORKER_POLL_SECONDS)
        except Exception as exc:
            self._status = AccompanimentStatus.FAILED
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.error("MRT2 generation worker failed: %s", self._last_error)

    def _get_or_embed_style(self, style: str) -> list[int] | np.ndarray:
        cached = self._style_cache.get(style)
        if cached is not None:
            return cached
        prompt = STYLE_PRESETS[style].prompt if style in STYLE_PRESETS else style  # type: ignore[index]
        embedding = self._system.embed_style(prompt)
        tokenize_fn = getattr(self._system, "tokenize_style", None)
        if callable(tokenize_fn):
            tokens = tokenize_fn(embedding)
            value: list[int] | np.ndarray = (
                tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            )
        elif isinstance(embedding, list):
            value = embedding
        else:
            value = np.asarray(embedding)
        self._style_cache[style] = value
        return value

    @staticmethod
    def _resolve_preset(style: str) -> StylePreset:
        if style in STYLE_PRESETS:
            return get_style_preset(style)
        return STYLE_PRESETS["rock"]


def create_accompaniment_engine(
    config: AccompanimentConfig,
    sample_rate_hz: int = MRT2_SAMPLE_RATE_HZ,
    *,
    system_factory: Callable[[str], MrtSystemProtocol] = default_mrt2_system_factory,
    preload_presets: bool = True,
) -> AccompanimentEngine:
    """Build `Mrt2Accompaniment` when enabled, falling back to `NullAccompaniment` (§82)."""
    if not config.enabled:
        return NullAccompaniment(
            style=config.style,
            follow_improvise=config.follow_improvise,
        )
    try:
        return Mrt2Accompaniment(
            config,
            sample_rate_hz=sample_rate_hz,
            system_factory=system_factory,
            preload_presets=preload_presets,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("MRT2 unavailable (%s); falling back to NullAccompaniment.", reason)
        return NullAccompaniment(
            style=config.style,
            follow_improvise=config.follow_improvise,
            unavailable_reason=reason,
        )

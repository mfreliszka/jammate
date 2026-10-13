"""Tests for Mrt2Accompaniment and create_accompaniment_engine (Step 4.2)."""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

from jammate.ai.base import TOKEN_HELD, TOKEN_OFF, TOKEN_ONSET, AccompanimentStatus
from jammate.ai.mrt import (
    MRT2_SAMPLES_PER_FRAME,
    Mrt2Accompaniment,
    create_accompaniment_engine,
)
from jammate.ai.null import NullAccompaniment
from jammate.config import AccompanimentConfig, JamMateConfig
from jammate.session import JamSession


@dataclass(frozen=True)
class FakeWaveform:
    samples: np.ndarray
    sample_rate: int = 48_000


class FakeMrtSystem:
    """Fast deterministic double for `MagentaRT2System`."""

    def __init__(self, *, fail_after_calls: int | None = None, target_calls: int = 2) -> None:
        import threading

        self.embedded_prompts: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.fail_after_calls = fail_after_calls
        self._target_calls = target_calls
        self.reached_target = threading.Event()

    def embed_style(self, text_or_audio: str) -> np.ndarray:
        self.embedded_prompts.append(text_or_audio)
        return np.array([len(text_or_audio), 1, 2, 3], dtype=np.int32)

    def tokenize_style(self, embedding: np.ndarray) -> np.ndarray:
        return embedding

    def generate(
        self,
        conditioning: dict[str, list[int] | np.ndarray] | None = None,
        cfg_scales: dict[str, float] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        frames: int = 25,
        state: Any = None,
    ) -> tuple[FakeWaveform, int]:
        call_index = len(self.generate_calls)
        if self.fail_after_calls is not None and call_index >= self.fail_after_calls:
            raise RuntimeError("simulated MLX inference failure")
        cond_copy: dict[str, Any] = {}
        if conditioning is not None:
            for k, v in conditioning.items():
                cond_copy[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else list(v)
        self.generate_calls.append(
            {
                "conditioning": cond_copy,
                "cfg_scales": dict(cfg_scales or {}),
                "temperature": temperature,
                "top_k": top_k,
                "frames": frames,
                "state": state,
            }
        )
        if len(self.generate_calls) >= self._target_calls:
            self.reached_target.set()
        next_state = 1 if state is None else int(state) + 1
        samples = np.full((frames * MRT2_SAMPLES_PER_FRAME, 2), 0.25, dtype=np.float32)
        return FakeWaveform(samples=samples), next_state


def test_only_ai_mrt_imports_magenta_rt_in_src() -> None:
    """Architectural rule (§51, §52): no module outside `ai/mrt.py` imports `magenta_rt`."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "jammate"
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        if path.name == "mrt.py" and path.parent.name == "ai":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "magenta_rt" or alias.name.startswith("magenta_rt."):
                        offenders.append(str(path.relative_to(src_root)))
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "magenta_rt" or node.module.startswith("magenta_rt."))
            ):
                offenders.append(str(path.relative_to(src_root)))
    assert offenders == []


def test_mrt2_accompaniment_caches_styles_and_conditions_notes_and_drums() -> None:
    fake_sys = FakeMrtSystem()
    cfg = AccompanimentConfig(
        enabled=True,
        style="stoner_rock",
        chunk_frames=2,
        drum_token=0,
        follow_improvise=0.25,
    )
    engine = Mrt2Accompaniment(
        cfg,
        system_factory=lambda _size: fake_sys,
        preload_presets=True,
    )
    assert engine.status == AccompanimentStatus.READY
    # All 5 style presets were pre-embedded once during initialization.
    assert len(fake_sys.embedded_prompts) == 5

    # Switching between presets uses the cache without calling embed_style again.
    engine.set_style("funk")
    engine.set_style("stoner_rock")
    assert len(fake_sys.embedded_prompts) == 5

    # Trigger note E2 (MIDI 40): first chunk has TOKEN_ONSET (2), second chunk has TOKEN_HELD (1).
    engine.note_on(40)
    chunk1 = engine.generate_chunk()
    assert chunk1.shape == (2 * MRT2_SAMPLES_PER_FRAME, 2)
    assert engine.output_ring is not None
    assert engine.output_ring.available_frames() == 2 * MRT2_SAMPLES_PER_FRAME

    chunk2 = engine.generate_chunk()
    assert chunk2.shape == (2 * MRT2_SAMPLES_PER_FRAME, 2)

    # Release E2: third chunk has all TOKEN_OFF (0).
    engine.note_off(40)
    engine.generate_chunk()

    assert len(fake_sys.generate_calls) == 3
    call0, call1, call2 = fake_sys.generate_calls

    # Streaming state is threaded continuously across calls.
    assert call0["state"] is None
    assert call1["state"] == 1
    assert call2["state"] == 2

    # Pianoroll conditioning reflects onset -> held -> off.
    roll0 = call0["conditioning"][PIANOROLL_WITH_ONSETS.key]
    roll1 = call1["conditioning"][PIANOROLL_WITH_ONSETS.key]
    roll2 = call2["conditioning"][PIANOROLL_WITH_ONSETS.key]
    assert roll0[40] == TOKEN_ONSET
    assert roll1[40] == TOKEN_HELD
    assert np.all(roll2 == TOKEN_OFF)

    # Drum token and style token are passed on every chunk.
    assert call0["conditioning"][DRUM_PIANOROLL.key] == [0]
    assert MUSICCOCA.key in call0["conditioning"]
    assert set(call0["cfg_scales"].keys()) == {"musiccoca", "notes", "drums"}

    stats = engine.stats()
    assert stats.frames_generated == 6
    assert stats.ms_per_frame >= 0.0
    assert stats.buffer_seconds == pytest.approx(6 * MRT2_SAMPLES_PER_FRAME / 48_000.0)


def test_mrt2_worker_thread_maintains_lead_buffer_and_stops_cleanly() -> None:
    fake_sys = FakeMrtSystem(target_calls=2)
    cfg = AccompanimentConfig(
        enabled=True,
        chunk_frames=2,
        target_lead_seconds=0.16,  # 4 frames = 7,680 samples
        ring_capacity_seconds=1.0,
    )
    engine = Mrt2Accompaniment(
        cfg,
        system_factory=lambda _size: fake_sys,
        preload_presets=False,
    )
    engine.start()
    try:
        assert fake_sys.reached_target.wait(timeout=2.0)
        assert engine.output_ring is not None
        assert engine.output_ring.available_frames() >= 3_840
    finally:
        engine.stop()
    assert not engine.is_running
    assert engine.status == AccompanimentStatus.STOPPED


def test_missing_or_failing_model_falls_back_to_null_and_leaves_jam_running(fake_stream) -> None:
    del fake_stream

    def broken_factory(_size: str) -> FakeMrtSystem:
        raise FileNotFoundError("mrt2_small.safetensors not found")

    cfg = JamMateConfig(ai=AccompanimentConfig(enabled=True))
    session = JamSession(
        cfg,
        accompaniment_factory=lambda ai_cfg, sr: create_accompaniment_engine(
            ai_cfg, sr, system_factory=broken_factory
        ),
    )
    assert isinstance(session.ai.inner, NullAccompaniment)
    assert session.ai.status == AccompanimentStatus.UNAVAILABLE
    assert "mrt2_small.safetensors" in (session.ai.stats().last_error or "")

    session.start()
    assert session.is_running
    assert session.drums.is_running
    session.stop()


def test_mrt2_gate_gain_scales_output_when_gated() -> None:
    fake_sys = FakeMrtSystem()
    cfg = AccompanimentConfig(enabled=True, chunk_frames=1)
    engine = Mrt2Accompaniment(
        cfg,
        system_factory=lambda _size: fake_sys,
        preload_presets=False,
    )
    engine.set_gate_gain(0.0)
    gated_samples = engine.generate_chunk()
    assert np.all(gated_samples == 0.0)

    engine.set_gate_gain(1.0)
    active_samples = engine.generate_chunk()
    assert np.allclose(active_samples, 0.25)


def test_thread_bound_mrt_system_dispatches_on_single_thread_and_shuts_down() -> None:
    from jammate.ai.mrt import _get_mlx_executor, _ThreadBoundMrtSystem, shutdown_mlx_runtime

    seen_threads: set[int] = set()

    class ThreadRecordingSystem(FakeMrtSystem):
        def embed_style(self, text_or_audio: str) -> np.ndarray:
            seen_threads.add(threading.get_ident())
            return super().embed_style(text_or_audio)

        def tokenize_style(self, embedding: np.ndarray) -> np.ndarray:
            seen_threads.add(threading.get_ident())
            return super().tokenize_style(embedding)

        def generate(self, **kwargs: Any) -> tuple[FakeWaveform, dict[str, int]]:
            seen_threads.add(threading.get_ident())
            return super().generate(**kwargs)

    try:
        executor = _get_mlx_executor()
        bound = _ThreadBoundMrtSystem(executor, ThreadRecordingSystem())
        emb = bound.embed_style("stoner rock")
        _ = bound.tokenize_style(emb)
        wav, _ = bound.generate(frames=1)
        assert wav.samples.shape == (MRT2_SAMPLES_PER_FRAME, 2)
        assert len(seen_threads) == 1
        assert next(iter(seen_threads)) != threading.get_ident()
    finally:
        shutdown_mlx_runtime()
        assert not any(t.name.startswith("jammate-mlx") for t in threading.enumerate())

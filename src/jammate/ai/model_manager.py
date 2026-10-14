"""Magenta RealTime 2 model asset manager and startup verifier (Step 7.2, §86, §151).

Inspects `MAGENTA_HOME` (`~/Documents/Magenta/magenta-rt-v2` by default) to verify required
checkpoint and resource files (`mrt2_small` / `mrt2_base`, `musiccoca`, and `spectrostream`),
produces human-readable status reports for the CLI (`--check-models`) and GUI, and provides a
non-blocking background setup/download trigger when model weights are missing.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jammate.config import ModelSize

logger = logging.getLogger(__name__)

DEFAULT_MAGENTA_SUBDIR = "magenta-rt-v2"

MUSICCOCA_REQUIRED_FILES: tuple[str, ...] = (
    "resources/musiccoca/spm.model",
    "resources/musiccoca/text_encoder.tflite",
    "resources/musiccoca/pretrained_vector_quantizer.tflite",
)

SPECTROSTREAM_REQUIRED_FILES: tuple[str, ...] = (
    "resources/spectrostream/encoder.safetensors",
    "resources/spectrostream/decoder.safetensors",
)

MODEL_CHECKPOINT_FILES: dict[ModelSize, str] = {
    "mrt2_small": "checkpoints/mrt2_small.safetensors",
    "mrt2_base": "checkpoints/mrt2_base.safetensors",
}


class ModelAssetStatus(StrEnum):
    """Health/availability state of local Magenta RT2 model weights (§86, §151)."""

    READY = "ready"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ModelAssetReport:
    """Result of inspecting the local `MAGENTA_HOME` directory for required MRT2 files."""

    status: ModelAssetStatus
    root_dir: Path
    model_size: ModelSize
    present_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    total_bytes: int

    @property
    def is_ready(self) -> bool:
        return self.status == ModelAssetStatus.READY

    @property
    def required_count(self) -> int:
        return len(self.present_files) + len(self.missing_files)

    def summary_line(self) -> str:
        """Concise one-line status suitable for GUI labels and status bars."""
        present = len(self.present_files)
        total = self.required_count
        if self.status == ModelAssetStatus.READY:
            size_mb = self.total_bytes / (1024.0 * 1024.0)
            return f"Ready ({self.model_size}, {present}/{total} files, {size_mb:.1f} MB)"
        if self.status == ModelAssetStatus.PARTIAL:
            return (
                f"Partial ({self.model_size}: {present}/{total} files present, "
                f"{len(self.missing_files)} missing)"
            )
        return f"Missing ({self.model_size}: run `uv run mrt models download`)"


def resolve_magenta_root(magenta_home: Path | str | None = None) -> Path:
    """Resolve the `magenta-rt-v2` asset directory from `magenta_home` or `MAGENTA_HOME` (§86).

    Accepts either:
    - A direct path to the `magenta-rt-v2` folder (containing `checkpoints/` and `resources/`), or
    - The parent `Magenta` base directory (in which case `magenta-rt-v2` is appended).
    """
    raw = magenta_home if magenta_home is not None else os.environ.get("MAGENTA_HOME")
    if raw is None:
        return Path.home() / "Documents" / "Magenta" / DEFAULT_MAGENTA_SUBDIR

    candidate = Path(raw).expanduser()
    if candidate.name == DEFAULT_MAGENTA_SUBDIR:
        return candidate
    if (candidate / "checkpoints").exists() or (candidate / "resources").exists():
        return candidate
    subdir = candidate / DEFAULT_MAGENTA_SUBDIR
    if subdir.exists():
        return subdir
    return candidate


def required_model_files(model_size: ModelSize = "mrt2_small") -> tuple[str, ...]:
    """Return the relative file paths required under `magenta-rt-v2` for `model_size`."""
    if model_size not in MODEL_CHECKPOINT_FILES:
        raise ValueError(f"Unsupported model_size: {model_size!r}")
    return (
        MODEL_CHECKPOINT_FILES[model_size],
        *MUSICCOCA_REQUIRED_FILES,
        *SPECTROSTREAM_REQUIRED_FILES,
    )


def inspect_model_assets(
    magenta_home: Path | str | None = None,
    *,
    model_size: ModelSize = "mrt2_small",
) -> ModelAssetReport:
    """Verify required MRT2 checkpoint, MusicCoCa, and SpectroStream files in `magenta_home`."""
    root = resolve_magenta_root(magenta_home)
    req_files = required_model_files(model_size)

    if not root.is_dir():
        return ModelAssetReport(
            status=ModelAssetStatus.MISSING,
            root_dir=root,
            model_size=model_size,
            present_files=(),
            missing_files=req_files,
            total_bytes=0,
        )

    present: list[str] = []
    missing: list[str] = []
    total_bytes = 0

    for rel_path in req_files:
        full_path = root / rel_path
        try:
            if full_path.is_file():
                size = full_path.stat().st_size
                if size > 0:
                    present.append(rel_path)
                    total_bytes += size
                else:
                    missing.append(rel_path)
            else:
                missing.append(rel_path)
        except OSError:
            missing.append(rel_path)

    if not missing:
        status = ModelAssetStatus.READY
    elif present:
        status = ModelAssetStatus.PARTIAL
    else:
        status = ModelAssetStatus.MISSING

    return ModelAssetReport(
        status=status,
        root_dir=root,
        model_size=model_size,
        present_files=tuple(present),
        missing_files=tuple(missing),
        total_bytes=total_bytes,
    )


def format_model_status(report: ModelAssetReport) -> str:
    """Format a human-readable report for `--check-models` and diagnostics (§86, §151)."""
    lines = [
        f"Magenta RT2 Model Asset Check ({report.model_size})",
        f"  Root directory: {report.root_dir}",
        f"  Status:         {report.status.value.upper()} — {report.summary_line()}",
        "  Files:",
    ]
    present_set = set(report.present_files)
    for rel_path in required_model_files(report.model_size):
        marker = "OK" if rel_path in present_set else "MISSING"
        lines.append(f"    [{marker:<7}] {rel_path}")

    if not report.is_ready:
        lines.append("")
        lines.append("  To initialize and download Magenta RT2 weights, run:")
        lines.append("    uv run mrt models init && uv run mrt models download")
    return "\n".join(lines)


def populate_placeholder_model_assets(
    target_dir: Path,
    *,
    model_size: ModelSize = "mrt2_small",
) -> ModelAssetReport:
    """Create non-empty placeholder files for all required `model_size` assets in `target_dir`."""
    for rel_path in required_model_files(model_size):
        file_path = target_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"JAMMATE_MRT2_PLACEHOLDER")
    return inspect_model_assets(target_dir, model_size=model_size)


def default_download_runner(magenta_home: Path) -> bool:
    """Default command runner invoking `mrt models init` and `mrt models download`."""
    env = dict(os.environ)
    base_dir = magenta_home.parent if magenta_home.name == DEFAULT_MAGENTA_SUBDIR else magenta_home
    env["MAGENTA_HOME"] = str(base_dir)
    try:
        subprocess.run(
            ["mrt", "models", "init"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        subprocess.run(
            ["mrt", "models", "download"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return True
    except Exception as exc:
        logger.warning("Background MRT2 model setup failed: %s", exc)
        return False


class ModelSetupTask:
    """Non-blocking background setup/download trigger for missing MRT2 model assets (§86, §151)."""

    def __init__(
        self,
        magenta_home: Path | str | None = None,
        *,
        model_size: ModelSize = "mrt2_small",
        runner: Callable[[Path], bool] = default_download_runner,
        on_complete: Callable[[ModelAssetReport], None] | None = None,
    ) -> None:
        self._root_dir = resolve_magenta_root(magenta_home)
        self._model_size: ModelSize = model_size
        self._runner = runner
        self._on_complete = on_complete
        self._thread: threading.Thread | None = None
        self._last_report: ModelAssetReport | None = None
        self._succeeded: bool | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def succeeded(self) -> bool | None:
        return self._succeeded

    @property
    def last_report(self) -> ModelAssetReport | None:
        return self._last_report

    def start(self) -> None:
        """Start the background setup/download thread if not already running."""
        if self.is_running:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="jammate-model-setup",
            daemon=False,
        )
        self._thread.start()

    def join(self, timeout: float = 5.0) -> ModelAssetReport:
        """Wait for the background task to complete and return the resulting `ModelAssetReport`."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None
        if self._last_report is None:
            self._last_report = inspect_model_assets(self._root_dir, model_size=self._model_size)
        return self._last_report

    def _run(self) -> None:
        ok = False
        try:
            ok = bool(self._runner(self._root_dir))
        except Exception as exc:
            logger.warning("ModelSetupTask runner raised %s: %s", type(exc).__name__, exc)
            ok = False
        report = inspect_model_assets(self._root_dir, model_size=self._model_size)
        self._last_report = report
        self._succeeded = ok and report.is_ready
        if self._on_complete is not None:
            self._on_complete(report)

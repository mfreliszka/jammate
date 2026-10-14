"""Unit tests for the Magenta RT2 model asset manager (`ai/model_manager.py`, Step 7.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jammate.ai.model_manager import (
    ModelAssetReport,
    ModelAssetStatus,
    ModelSetupTask,
    format_model_status,
    inspect_model_assets,
    populate_placeholder_model_assets,
    required_model_files,
    resolve_magenta_root,
)
from jammate.main import main as cli_main


def test_inspect_model_assets_detects_missing_directory(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist" / "magenta-rt-v2"
    report = inspect_model_assets(nonexistent, model_size="mrt2_small")

    assert report.status == ModelAssetStatus.MISSING
    assert not report.is_ready
    assert report.present_files == ()
    assert report.missing_files == required_model_files("mrt2_small")
    assert report.total_bytes == 0
    assert "Missing" in report.summary_line()

    formatted = format_model_status(report)
    assert "MISSING" in formatted
    assert "uv run mrt models init && uv run mrt models download" in formatted


def test_inspect_model_assets_detects_partial_and_zero_byte_files(tmp_path: Path) -> None:
    root = tmp_path / "magenta-rt-v2"
    req = required_model_files("mrt2_small")

    # Create first two files with content, third file as 0-byte empty file.
    for rel in req[:2]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"valid_weights")
    zero_file = root / req[2]
    zero_file.parent.mkdir(parents=True, exist_ok=True)
    zero_file.write_bytes(b"")

    report = inspect_model_assets(root, model_size="mrt2_small")
    assert report.status == ModelAssetStatus.PARTIAL
    assert not report.is_ready
    assert report.present_files == req[:2]
    assert req[2] in report.missing_files
    assert len(report.missing_files) == len(req) - 2
    assert "Partial" in report.summary_line()


@pytest.mark.parametrize("model_size", ["mrt2_small", "mrt2_base"])
def test_inspect_model_assets_detects_valid_ready_directory_and_parent_resolution(
    tmp_path: Path,
    model_size: str,
) -> None:
    magenta_parent = tmp_path / "Magenta"
    v2_dir = magenta_parent / "magenta-rt-v2"
    report = populate_placeholder_model_assets(v2_dir, model_size=model_size)  # type: ignore[arg-type]

    assert report.status == ModelAssetStatus.READY
    assert report.is_ready
    assert report.missing_files == ()
    assert len(report.present_files) == len(required_model_files(model_size))  # type: ignore[arg-type]
    assert report.total_bytes > 0
    assert "Ready" in report.summary_line()

    # Passing the parent `Magenta` directory resolves automatically to `Magenta/magenta-rt-v2`.
    assert resolve_magenta_root(magenta_parent) == v2_dir
    via_parent = inspect_model_assets(magenta_parent, model_size=model_size)  # type: ignore[arg-type]
    assert via_parent.is_ready


def test_model_setup_task_populates_assets_on_background_thread(tmp_path: Path) -> None:
    root = tmp_path / "magenta-rt-v2"
    completed_reports: list[ModelAssetReport] = []

    def fake_runner(target_root: Path) -> bool:
        populate_placeholder_model_assets(target_root, model_size="mrt2_small")
        return True

    task = ModelSetupTask(
        root,
        model_size="mrt2_small",
        runner=fake_runner,
        on_complete=completed_reports.append,
    )
    task.start()
    final_report = task.join(timeout=2.0)

    assert not task.is_running
    assert task.succeeded is True
    assert final_report.is_ready
    assert len(completed_reports) == 1
    assert completed_reports[0].is_ready


def test_cli_check_models_flag_returns_exit_code_for_missing_and_valid_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    v2_dir = tmp_path / "magenta-rt-v2"
    monkeypatch.setenv("MAGENTA_HOME", str(v2_dir))

    # Missing -> exit code 1
    assert cli_main(["--check-models"]) == 1
    out_missing = capsys.readouterr().out
    assert "MISSING" in out_missing

    # Valid -> exit code 0
    populate_placeholder_model_assets(v2_dir, model_size="mrt2_small")
    assert cli_main(["--check-models"]) == 0
    out_ready = capsys.readouterr().out
    assert "READY" in out_ready

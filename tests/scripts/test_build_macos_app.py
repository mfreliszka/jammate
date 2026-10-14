"""Tests for standalone macOS `.app` packaging and `scripts/build_macos_app.py` (Step 7.2)."""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_macos_app.py"
    spec = importlib.util.spec_from_file_location("build_macos_app", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_packaging_config_passes_on_repository() -> None:
    mod = _load_build_script()
    report = mod.validate_packaging_config()

    assert report.ok is True
    assert report.errors == ()
    assert report.bundle_identifier == "com.jammate.app"
    assert report.executable_name == "JamMate"
    assert "arm64" in report.architectures
    assert "USB audio interface" in report.microphone_description
    assert "com.apple.security.device.audio-input" in report.entitlements
    assert len(report.drum_assets) == 5
    assert report.native_source_present is True


def test_validate_packaging_config_detects_missing_metadata_and_assets(tmp_path: Path) -> None:
    mod = _load_build_script()
    bad_repo = tmp_path / "bad_repo"
    (bad_repo / "packaging").mkdir(parents=True)
    (bad_repo / "assets" / "drums").mkdir(parents=True)

    # Write an incomplete Info.plist (wrong bundle ID, no mic usage description, no arm64)
    bad_info = {
        "CFBundleIdentifier": "com.wrong.id",
        "CFBundleExecutable": "JamMate",
        "LSArchitecturePriority": ["x86_64"],
        "NSHighResolutionCapable": False,
    }
    (bad_repo / "packaging" / "Info.plist").write_bytes(plistlib.dumps(bad_info))
    (bad_repo / "packaging" / "entitlements.plist").write_bytes(plistlib.dumps({}))

    report = mod.validate_packaging_config(bad_repo)
    assert report.ok is False
    joined_errors = "\n".join(report.errors)
    assert "CFBundleIdentifier" in joined_errors
    assert "arm64" in joined_errors
    assert "NSMicrophoneUsageDescription" in joined_errors
    assert "com.apple.security.device.audio-input" in joined_errors
    assert "Missing or empty drum sample" in joined_errors


def test_build_macos_app_check_and_bundle_assembly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_build_script()

    assert mod.main(["--check"]) == 0
    check_output = capsys.readouterr().out
    assert "JamMate macOS Packaging Check: OK" in check_output
    assert "com.jammate.app" in check_output

    dist_dir = tmp_path / "dist"
    assert mod.main(["--output-dir", str(dist_dir), "--no-sign"]) == 0
    app_path = dist_dir / "JamMate.app"
    assert app_path.is_dir()
    assert mod.verify_built_bundle(app_path) == ()

    info = plistlib.loads((app_path / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "com.jammate.app"
    assert (app_path / "Contents" / "PkgInfo").read_text(encoding="utf-8") == "APPL????"
    assert (app_path / "Contents" / "MacOS" / "JamMate").is_file()
    assert (app_path / "Contents" / "Resources" / "assets" / "drums" / "kick.wav").is_file()
    assert (app_path / "Contents" / "Resources" / "jammate" / "main.py").is_file()

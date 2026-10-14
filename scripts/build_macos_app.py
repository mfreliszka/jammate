"""Build and validate the standalone Apple Silicon `JamMate.app` macOS bundle (Step 7.2, §86, §151).

Usage examples:
    uv run python scripts/build_macos_app.py --check
    uv run python scripts/build_macos_app.py --output-dir dist
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jammate.drums.sampler import KIT_FILENAMES
from jammate.native.kernels import CPP_SOURCE_PATH, DYLIB_PATH, compile_native_extension

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_BUNDLE_NAME = "JamMate.app"
EXPECTED_BUNDLE_ID = "com.jammate.app"
EXPECTED_EXECUTABLE = "JamMate"


@dataclass(frozen=True, slots=True)
class PackagingCheckReport:
    """Validation report for `packaging/Info.plist`, `entitlements.plist`, and bundled assets."""

    ok: bool
    bundle_identifier: str
    executable_name: str
    architectures: tuple[str, ...]
    microphone_description: str
    entitlements: tuple[str, ...]
    drum_assets: tuple[str, ...]
    native_source_present: bool
    errors: tuple[str, ...]


def validate_packaging_config(repo_root: Path = REPO_ROOT) -> PackagingCheckReport:
    """Verify `packaging/Info.plist`, `packaging/entitlements.plist`, drum WAVs, and C++ source."""
    errors: list[str] = []
    info_plist_path = repo_root / "packaging" / "Info.plist"
    entitlements_path = repo_root / "packaging" / "entitlements.plist"
    drums_dir = repo_root / "assets" / "drums"
    cpp_path = repo_root / "src" / "jammate" / "native" / "_jammate_native.cpp"

    bundle_id = ""
    executable = ""
    archs: tuple[str, ...] = ()
    mic_desc = ""
    entitlement_keys: tuple[str, ...] = ()
    drum_files: list[str] = []

    if not info_plist_path.is_file():
        errors.append(f"Missing Info.plist: {info_plist_path}")
    else:
        try:
            info = plistlib.loads(info_plist_path.read_bytes())
        except Exception as exc:
            errors.append(f"Invalid Info.plist XML ({exc})")
            info = {}
        if isinstance(info, dict):
            bundle_id = str(info.get("CFBundleIdentifier", ""))
            executable = str(info.get("CFBundleExecutable", ""))
            raw_archs = info.get("LSArchitecturePriority", [])
            if isinstance(raw_archs, list):
                archs = tuple(str(a) for a in raw_archs)
            mic_desc = str(info.get("NSMicrophoneUsageDescription", "")).strip()
            if bundle_id != EXPECTED_BUNDLE_ID:
                errors.append(
                    f"CFBundleIdentifier must be {EXPECTED_BUNDLE_ID!r}, got {bundle_id!r}"
                )
            if executable != EXPECTED_EXECUTABLE:
                errors.append(
                    f"CFBundleExecutable must be {EXPECTED_EXECUTABLE!r}, got {executable!r}"
                )
            if "arm64" not in archs:
                errors.append(f"LSArchitecturePriority must include 'arm64', got {archs!r}")
            if not mic_desc:
                errors.append("NSMicrophoneUsageDescription must be non-empty")
            if info.get("NSHighResolutionCapable") is not True:
                errors.append("NSHighResolutionCapable must be True")

    if not entitlements_path.is_file():
        errors.append(f"Missing entitlements.plist: {entitlements_path}")
    else:
        try:
            ent = plistlib.loads(entitlements_path.read_bytes())
        except Exception as exc:
            errors.append(f"Invalid entitlements.plist XML ({exc})")
            ent = {}
        if isinstance(ent, dict):
            entitlement_keys = tuple(sorted(k for k, v in ent.items() if v is True))
            if ent.get("com.apple.security.device.audio-input") is not True:
                errors.append(
                    "entitlements.plist must enable 'com.apple.security.device.audio-input'"
                )

    for filename in KIT_FILENAMES.values():
        wav_path = drums_dir / filename
        if wav_path.is_file() and wav_path.stat().st_size > 0:
            drum_files.append(filename)
        else:
            errors.append(f"Missing or empty drum sample: {wav_path}")

    native_present = cpp_path.is_file()
    if not native_present:
        errors.append(f"Missing native C++ source: {cpp_path}")

    return PackagingCheckReport(
        ok=len(errors) == 0,
        bundle_identifier=bundle_id,
        executable_name=executable,
        architectures=archs,
        microphone_description=mic_desc,
        entitlements=entitlement_keys,
        drum_assets=tuple(drum_files),
        native_source_present=native_present,
        errors=tuple(errors),
    )


def format_packaging_report(report: PackagingCheckReport) -> str:
    """Render a human-readable summary of `PackagingCheckReport`."""
    status = "OK" if report.ok else "FAILED"
    lines = [
        f"JamMate macOS Packaging Check: {status}",
        f"  Bundle ID:      {report.bundle_identifier}",
        f"  Executable:     {report.executable_name}",
        f"  Architectures:  {', '.join(report.architectures) or '--'}",
        f"  Mic permission: {report.microphone_description}",
        f"  Entitlements:   {', '.join(report.entitlements) or '--'}",
        (
            f"  Drum assets:    {len(report.drum_assets)}/{len(KIT_FILENAMES)} "
            f"({', '.join(report.drum_assets)})"
        ),
        f"  Native C++ src: {'present' if report.native_source_present else 'missing'}",
    ]
    if report.errors:
        lines.append("  Errors:")
        for err in report.errors:
            lines.append(f"    - {err}")
    return "\n".join(lines)


def _launcher_script(python_executable: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
CONTENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
export PYTHONPATH="$RESOURCES_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
export JAMMATE_RESOURCES_DIR="$RESOURCES_DIR"
exec "{python_executable}" -m jammate.main "$@"
"""


def build_app_bundle(
    output_dir: Path = REPO_ROOT / "dist",
    *,
    repo_root: Path = REPO_ROOT,
    sign: bool = True,
) -> Path:
    """Assemble `JamMate.app` in `output_dir` and optionally sign it with `entitlements.plist`."""
    report = validate_packaging_config(repo_root)
    if not report.ok:
        raise RuntimeError("Packaging validation failed: " + "; ".join(report.errors))

    cpp_src = repo_root / "src" / "jammate" / "native" / "_jammate_native.cpp"
    dylib_out = repo_root / "src" / "jammate" / "native" / "_jammate_native.dylib"
    if cpp_src.is_file() and not dylib_out.is_file():
        compile_native_extension(cpp_src, dylib_out)

    app_path = output_dir / APP_BUNDLE_NAME
    if app_path.exists():
        shutil.rmtree(app_path)

    contents_dir = app_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    drums_target_dir = resources_dir / "assets" / "drums"

    macos_dir.mkdir(parents=True, exist_ok=True)
    drums_target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Info.plist, PkgInfo, and entitlements.plist
    shutil.copy2(repo_root / "packaging" / "Info.plist", contents_dir / "Info.plist")
    (contents_dir / "PkgInfo").write_text("APPL????", encoding="utf-8")
    shutil.copy2(
        repo_root / "packaging" / "entitlements.plist",
        resources_dir / "entitlements.plist",
    )

    # 2. Bundle drum kit WAVs
    for filename in KIT_FILENAMES.values():
        shutil.copy2(repo_root / "assets" / "drums" / filename, drums_target_dir / filename)

    # 3. Copy jammate package + compiled _jammate_native.dylib into Contents/Resources/jammate
    shutil.copytree(
        repo_root / "src" / "jammate",
        resources_dir / "jammate",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    # 4. Write executable launcher in Contents/MacOS/JamMate
    executable_path = macos_dir / EXPECTED_EXECUTABLE
    executable_path.write_text(_launcher_script(sys.executable), encoding="utf-8")
    mode = executable_path.stat().st_mode
    executable_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # 5. Ad-hoc codesign with microphone/audio entitlements
    if sign and shutil.which("codesign") is not None:
        entitlements_path = repo_root / "packaging" / "entitlements.plist"
        subprocess.run(
            [
                "codesign",
                "--force",
                "--deep",
                "--sign",
                "-",
                "--entitlements",
                str(entitlements_path),
                str(app_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return app_path


def verify_built_bundle(app_path: Path) -> tuple[str, ...]:
    """Verify the directory structure, plist, executable permissions, and assets of `app_path`."""
    errors: list[str] = []
    info_path = app_path / "Contents" / "Info.plist"
    exe_path = app_path / "Contents" / "MacOS" / EXPECTED_EXECUTABLE
    drums_dir = app_path / "Contents" / "Resources" / "assets" / "drums"
    pkg_dir = app_path / "Contents" / "Resources" / "jammate"

    if not info_path.is_file():
        errors.append(f"Missing bundle Info.plist: {info_path}")
    if not exe_path.is_file():
        errors.append(f"Missing bundle executable: {exe_path}")
    elif not os.access(exe_path, os.X_OK):
        errors.append(f"Bundle executable is not executable: {exe_path}")
    for filename in KIT_FILENAMES.values():
        if not (drums_dir / filename).is_file():
            errors.append(f"Missing bundled drum sample: {drums_dir / filename}")
    if not (pkg_dir / "main.py").is_file():
        errors.append(f"Missing bundled package entry point: {pkg_dir / 'main.py'}")
    return tuple(errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate the standalone macOS JamMate.app bundle (Step 7.2).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate packaging configuration (Info.plist, entitlements, assets) and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist",
        help="directory where JamMate.app will be written (default: dist/)",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="skip ad-hoc codesign step",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = validate_packaging_config(REPO_ROOT)
    if args.check:
        print(format_packaging_report(report))
        return 0 if report.ok else 1

    if not report.ok:
        print(format_packaging_report(report), file=sys.stderr)
        return 1

    if CPP_SOURCE_PATH.is_file() and not DYLIB_PATH.is_file():
        compile_native_extension(CPP_SOURCE_PATH, DYLIB_PATH)

    app_path = build_app_bundle(args.output_dir, repo_root=REPO_ROOT, sign=not args.no_sign)
    bundle_errors = verify_built_bundle(app_path)
    if bundle_errors:
        for err in bundle_errors:
            print(f"Bundle verification error: {err}", file=sys.stderr)
        return 1

    print(f"Built {app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

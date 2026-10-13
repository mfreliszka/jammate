import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_placeholder_kit.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_placeholder_kit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_placeholder_kit"] = module
    spec.loader.exec_module(module)
    return module


make_placeholder_kit = _load_script()


def test_writes_all_five_kit_pieces_with_click_free_envelopes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert make_placeholder_kit.main(["--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    for name in ("kick", "snare", "closed_hat", "crash", "tom"):
        path = tmp_path / f"{name}.wav"
        assert path.is_file()
        assert f"{name}.wav" in out
        samples, sr = sf.read(path, dtype="float32")
        assert sr == 48_000
        assert samples.ndim == 1
        assert samples.size > 1_000
        assert samples[0] == 0.0
        assert samples[-1] == 0.0
        peak = float(np.max(np.abs(samples)))
        assert 0.5 <= peak <= 0.9


def test_refuses_to_overwrite_without_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert make_placeholder_kit.main(["--output-dir", str(tmp_path)]) == 0
    assert make_placeholder_kit.main(["--output-dir", str(tmp_path)]) == 1
    assert "already exists" in capsys.readouterr().err

    assert make_placeholder_kit.main(["--output-dir", str(tmp_path), "--overwrite"]) == 0


def test_generation_is_deterministic() -> None:
    kit_a = make_placeholder_kit.generate_kit(48_000)
    kit_b = make_placeholder_kit.generate_kit(48_000)
    for name in kit_a:
        np.testing.assert_array_equal(kit_a[name], kit_b[name])

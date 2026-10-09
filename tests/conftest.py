import pytest

from jammate.audio import engine as engine_module
from tests.fakes import FakeStream


@pytest.fixture
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> type[FakeStream]:
    """Replace `sd.Stream` in the engine with `FakeStream`."""
    FakeStream.reset()
    monkeypatch.setattr(engine_module.sd, "Stream", FakeStream)
    return FakeStream

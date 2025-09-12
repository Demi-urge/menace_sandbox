import sys
import types
import importlib
import pytest

# Ensure we import the real helper rather than the lightweight test stub
sys.modules.pop("context_builder_util", None)

# Stub heavy vector_service before importing the helper
sys.modules.setdefault(
    "vector_service.context_builder", types.SimpleNamespace(ContextBuilder=object)
)

cbu = importlib.import_module("context_builder_util")  # noqa: E402


def test_create_context_builder_paths(monkeypatch):
    monkeypatch.delenv("SANDBOX_DATA_DIR", raising=False)
    captured = {}

    class DummyBuilder:
        pass

    def fake_cb(*args):
        captured['args'] = args
        return DummyBuilder()

    monkeypatch.setattr(cbu._create_module, 'ContextBuilder', fake_cb)
    builder = cbu.create_context_builder()
    assert captured['args'] == ("bots.db", "code.db", "errors.db", "workflows.db")
    assert isinstance(builder, DummyBuilder)


def test_create_context_builder_requires_paths(monkeypatch):
    class DummyBuilder:
        def __init__(self):
            pass

    monkeypatch.setattr(cbu._create_module, "ContextBuilder", DummyBuilder)
    with pytest.raises(ValueError):
        cbu.create_context_builder()


def test_ensure_fresh_weights_invokes_builder():
    called = False

    class DummyBuilder:
        def refresh_db_weights(self):
            nonlocal called
            called = True

    cbu.ensure_fresh_weights(DummyBuilder())
    assert called


def test_ensure_fresh_weights_propagates_exception():
    class DummyBuilder:
        def refresh_db_weights(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cbu.ensure_fresh_weights(DummyBuilder())

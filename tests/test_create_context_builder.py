import sys
import types
import pytest

# Stub heavy ``vector_service`` before importing the helper
sys.modules.setdefault("vector_service", types.SimpleNamespace(ContextBuilder=object))

import context_builder_util as cbu  # noqa: E402


def test_create_context_builder_paths(monkeypatch):
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

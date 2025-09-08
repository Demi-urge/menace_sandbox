import sys
import types

# Stub heavy ``vector_service`` before importing the helper
sys.modules.setdefault("vector_service", types.SimpleNamespace(ContextBuilder=object))

import context_builder_util as cbu


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

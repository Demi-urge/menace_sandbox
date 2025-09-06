from __future__ import annotations

import sys
import types
from unittest import mock

from tests.test_self_improvement_logging import _load_engine


class DummyLogger:
    def info(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def test_replace_flag_triggers_refactor_and_event(monkeypatch):
    class DummyMRS:
        def __init__(self, *a, **k):
            pass

        def process_flags(self, flags):
            self.flags = flags

    stub = types.SimpleNamespace(ModuleRetirementService=DummyMRS)
    monkeypatch.setitem(sys.modules, "module_retirement_service", stub)

    sie = _load_engine()
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.logger = DummyLogger()
    engine.self_coding_engine = object()
    bus = mock.Mock()
    engine.event_bus = bus

    calls = []

    def fake_generate_patch(mod, sce, **kwargs):
        calls.append((mod, sce))
        return 123

    monkeypatch.setattr(sie, "generate_patch", fake_generate_patch)

    engine._handle_relevancy_flags({"foo": "replace", "bar": "retain"})

    assert calls == [("foo", engine.self_coding_engine)]
    bus.publish.assert_any_call(
        "relevancy:replace", {"module": "foo", "task_id": 123}
    )
    bus.publish.assert_any_call(
        "relevancy:scan", {"foo": "replace", "bar": "retain"}
    )

from __future__ import annotations

import sys
import types
from unittest import mock

from tests.test_self_improvement_logging import _load_engine


class DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def test_replace_flag_triggers_refactor_and_event(monkeypatch):
    class DummyMRS:
        def __init__(self, *a, **k):
            pass

        def process_flags(self, flags):
            self.flags = flags
            return flags

    stub = types.SimpleNamespace(ModuleRetirementService=DummyMRS)
    monkeypatch.setitem(sys.modules, "module_retirement_service", stub)
    ss_mod = sys.modules.setdefault("sandbox_settings", types.SimpleNamespace())
    ss_mod.DEFAULT_SEVERITY_SCORE_MAP = {}
    sys.modules["menace.sandbox_settings"] = ss_mod
    log_mod = sys.modules.setdefault("menace.logging_utils", types.ModuleType("menace.logging_utils"))
    log_mod.get_logger = lambda *a, **k: DummyLogger()
    log_mod.setup_logging = lambda *a, **k: None
    log_mod.log_record = lambda **kw: kw
    log_stub = types.ModuleType("menace_sandbox.logging_utils")
    log_stub.get_logger = lambda *a, **k: DummyLogger()
    log_stub.setup_logging = lambda *a, **k: None
    log_stub.log_record = lambda **kw: kw
    sys.modules["menace_sandbox.logging_utils"] = log_stub

    sie = _load_engine()
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.logger = DummyLogger()
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    engine.self_coding_engine = types.SimpleNamespace(context_builder=builder)
    bus = mock.Mock()
    engine.event_bus = bus

    calls = []

    def fake_generate_patch(mod, sce, **kwargs):
        calls.append((mod, sce, kwargs.get("context_builder")))
        return 123

    monkeypatch.setattr(sie, "generate_patch", fake_generate_patch)
    engine._patch_generator = sie.generate_patch

    engine._handle_relevancy_flags({"foo": "replace", "bar": "retain"})

    assert calls == [("foo", engine.self_coding_engine, builder)]
    bus.publish.assert_any_call(
        "relevancy:replace", {"module": "foo", "task_id": 123}
    )
    bus.publish.assert_any_call(
        "relevancy:scan", {"foo": "replace", "bar": "retain"}
    )

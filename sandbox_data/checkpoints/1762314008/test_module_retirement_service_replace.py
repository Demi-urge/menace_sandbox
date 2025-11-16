import sys
import types

scm_stub = types.ModuleType("self_coding_manager")
scm_stub.SelfCodingManager = type("SelfCodingManager", (), {})
sys.modules["self_coding_manager"] = scm_stub


class _DummyBuilder:
    def __init__(self, *a, **k):
        pass

    def refresh_db_weights(self):
        return None


vec_pkg = types.ModuleType("vector_service")
ctx_mod = types.ModuleType("vector_service.context_builder")
ctx_mod.ContextBuilder = _DummyBuilder
vec_pkg.context_builder = ctx_mod
sys.modules["vector_service"] = vec_pkg
sys.modules["vector_service.context_builder"] = ctx_mod

import module_retirement_service  # noqa: E402
from module_retirement_service import ModuleRetirementService  # noqa: E402


def _stub_build_graph(root):
    raise RuntimeError("skip graph build")


def test_replace_module(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"  # path-ignore
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    called = {}

    class DummyManager:
        def __init__(self):
            self.engine = None
            self.bot_name = "demo_bot"
            self.bot_registry = types.SimpleNamespace(update_bot=lambda *a, **k: None)
            self.evolution_orchestrator = types.SimpleNamespace(
                register_patch_cycle=lambda *a, **k: None, provenance_token="tok"
            )

        def generate_patch(self, path, *, context_builder, provenance_token, description=""):
            called["path"] = path
            return 1

    monkeypatch.setattr(module_retirement_service, "SelfCodingManager", DummyManager)

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "replaced_modules_total", gauge)

    mgr = DummyManager()
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    assert service.replace_module("demo")
    assert called["path"] == str(module)
    assert gauge.count == 1.0


def test_process_flags_replace(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"  # path-ignore
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    called = {}

    class DummyManager:
        def __init__(self):
            self.engine = None
            self.bot_name = "demo_bot"
            self.bot_registry = types.SimpleNamespace(update_bot=lambda *a, **k: None)
            self.evolution_orchestrator = types.SimpleNamespace(
                register_patch_cycle=lambda *a, **k: None, provenance_token="tok"
            )

        def generate_patch(self, path, *, context_builder, provenance_token, description=""):
            called["path"] = path
            return 1

    monkeypatch.setattr(module_retirement_service, "SelfCodingManager", DummyManager)

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "replaced_modules_total", gauge)

    captured = {}

    def fake_update(results):
        captured["results"] = results

    monkeypatch.setattr(module_retirement_service, "update_module_retirement_metrics", fake_update)

    mgr = DummyManager()
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    res = service.process_flags({"demo": "replace"})
    assert res == {"demo": "replaced"}
    assert called["path"] == str(module)
    assert captured["results"] == {"demo": "replaced"}
    assert gauge.count == 1.0


def test_process_flags_replace_skipped(monkeypatch, tmp_path, caplog):
    module = tmp_path / "demo.py"  # path-ignore
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    class DummyManager:
        def __init__(self):
            self.engine = None
            self.bot_name = "demo_bot"
            self.bot_registry = types.SimpleNamespace(update_bot=lambda *a, **k: None)
            self.evolution_orchestrator = types.SimpleNamespace(
                register_patch_cycle=lambda *a, **k: None, provenance_token="tok"
            )

        def generate_patch(self, path, *, context_builder, provenance_token, description=""):
            return None

    monkeypatch.setattr(module_retirement_service, "SelfCodingManager", DummyManager)

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "replaced_modules_total", gauge)

    captured = {}

    def fake_update(results):
        captured["results"] = results

    monkeypatch.setattr(module_retirement_service, "update_module_retirement_metrics", fake_update)

    mgr = DummyManager()
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    with caplog.at_level("INFO"):
        res = service.process_flags({"demo": "replace"})
    assert res == {"demo": "skipped"}
    assert captured["results"] == {"demo": "skipped"}
    assert gauge.count == 0.0
    assert "skipped demo" in caplog.text

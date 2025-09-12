import sys
import types

fake_qfe = types.ModuleType("quick_fix_engine")
fake_qfe.generate_patch = lambda path, *, context_builder, **kw: 1
sys.modules["quick_fix_engine"] = fake_qfe


class _DummyBuilder:
    def __init__(self, *a, **k):
        pass

    def refresh_db_weights(self):
        return None


sys.modules.setdefault("vector_service", types.SimpleNamespace(ContextBuilder=_DummyBuilder))

import module_retirement_service  # noqa: E402
from module_retirement_service import ModuleRetirementService  # noqa: E402


def _stub_build_graph(root):
    raise RuntimeError("skip graph build")


def test_replace_module(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"  # path-ignore
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    called = {}

    def fake_generate_patch(path, manager, *, context_builder, **kw):
        called["path"] = path
        return 1

    monkeypatch.setattr(module_retirement_service, "generate_patch", fake_generate_patch)

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "replaced_modules_total", gauge)

    mgr = types.SimpleNamespace(engine=None, register_patch_cycle=lambda *a, **k: None)
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

    def fake_generate_patch(path, manager, *, context_builder, **kw):
        called["path"] = path
        return 1

    monkeypatch.setattr(module_retirement_service, "generate_patch", fake_generate_patch)

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

    mgr = types.SimpleNamespace(engine=None, register_patch_cycle=lambda *a, **k: None)
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

    def fake_generate_patch(path, manager, *, context_builder, **kw):
        return None

    monkeypatch.setattr(module_retirement_service, "generate_patch", fake_generate_patch)

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

    mgr = types.SimpleNamespace(engine=None, register_patch_cycle=lambda *a, **k: None)
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    with caplog.at_level("INFO"):
        res = service.process_flags({"demo": "replace"})
    assert res == {"demo": "skipped"}
    assert captured["results"] == {"demo": "skipped"}
    assert gauge.count == 0.0
    assert "skipped demo" in caplog.text

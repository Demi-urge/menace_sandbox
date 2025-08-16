import sys
import types

fake_qfe = types.ModuleType("quick_fix_engine")
fake_qfe.generate_patch = lambda path: 1
sys.modules["quick_fix_engine"] = fake_qfe

import module_retirement_service
from module_retirement_service import ModuleRetirementService


def _stub_build_graph(root):
    raise RuntimeError("skip graph build")


def test_replace_module(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    called = {}

    def fake_generate_patch(path):
        called['path'] = path
        return 1

    monkeypatch.setattr(module_retirement_service, "generate_patch", fake_generate_patch)

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "replaced_modules_total", gauge)

    service = ModuleRetirementService(tmp_path)
    assert service.replace_module("demo")
    assert called['path'] == str(module)
    assert gauge.count == 1.0


def test_process_flags_replace(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    called = {}

    def fake_generate_patch(path):
        called['path'] = path
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
        captured['results'] = results

    monkeypatch.setattr(module_retirement_service, "update_module_retirement_metrics", fake_update)

    service = ModuleRetirementService(tmp_path)
    res = service.process_flags({"demo": "replace"})
    assert res == {"demo": "replaced"}
    assert called['path'] == str(module)
    assert captured['results'] == {"demo": "replaced"}
    assert gauge.count == 1.0


def test_retire_module_zero_impact(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

    import types
    import sys

    sandbox_stub = types.SimpleNamespace(
        SandboxSettings=lambda: types.SimpleNamespace(
            relevancy_threshold=10, relevancy_whitelist=[]
        )
    )
    monkeypatch.setitem(sys.modules, "sandbox_settings", sandbox_stub)

    import relevancy_radar

    monkeypatch.setattr(
        relevancy_radar, "_RELEVANCY_FLAGS_FILE", tmp_path / "flags.json"
    )

    flags = relevancy_radar.evaluate_relevancy({"demo": None}, {}, {"demo": 0.0})
    assert flags == {"demo": "retire"}

    captured = {}

    def fake_update(results):
        captured["results"] = results

    monkeypatch.setattr(
        module_retirement_service, "update_module_retirement_metrics", fake_update
    )

    class DummyGauge:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    gauge = DummyGauge()
    monkeypatch.setattr(module_retirement_service, "retired_modules_total", gauge)

    service = ModuleRetirementService(tmp_path)
    res = service.process_flags(flags)

    retired = tmp_path / "sandbox_data" / "retired_modules" / "demo.py"
    assert res == {"demo": "retired"}
    assert retired.exists()
    assert captured["results"] == {"demo": "retired"}
    assert gauge.count == 1.0

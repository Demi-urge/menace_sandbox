import sys
import types

fake_qfe = types.ModuleType("quick_fix_engine")
fake_qfe.generate_patch = lambda path, manager, *, context_builder, **kw: 1
sys.modules["quick_fix_engine"] = fake_qfe


class _DummyBuilder:
    def refresh_db_weights(self):
        return None


sys.modules.setdefault("vector_service", types.SimpleNamespace(ContextBuilder=_DummyBuilder))

import module_retirement_service  # noqa: E402
from module_retirement_service import ModuleRetirementService  # noqa: E402


def _stub_build_graph(root):
    raise RuntimeError("skip graph build")


def test_retire_module_zero_impact(monkeypatch, tmp_path):
    module = tmp_path / "demo.py"  # path-ignore
    module.write_text("print('hi')")

    monkeypatch.setattr(module_retirement_service, "build_import_graph", _stub_build_graph)

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

    mgr = types.SimpleNamespace(engine=None, register_patch_cycle=lambda *a, **k: None)
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    res = service.process_flags(flags)

    retired = tmp_path / "sandbox_data" / "retired_modules" / "demo.py"  # path-ignore
    assert res == {"demo": "retired"}
    assert retired.exists()
    assert captured["results"] == {"demo": "retired"}
    assert gauge.count == 1.0

import sys
import types

scm_stub = types.ModuleType("self_coding_manager")
scm_stub.SelfCodingManager = type("SelfCodingManager", (), {})
sys.modules["self_coding_manager"] = scm_stub


class _DummyBuilder:
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

    class DummyManager:
        def __init__(self):
            self.engine = None
            self.bot_name = "demo_bot"
            self.bot_registry = types.SimpleNamespace(update_bot=lambda *a, **k: None)
            self.evolution_orchestrator = types.SimpleNamespace(
                register_patch_cycle=lambda *a, **k: None, provenance_token="tok"
            )

        def generate_patch(self, *a, **k):
            return 1

    monkeypatch.setattr(module_retirement_service, "SelfCodingManager", DummyManager)
    mgr = DummyManager()
    service = ModuleRetirementService(
        tmp_path, context_builder=_DummyBuilder(), manager=mgr
    )
    res = service.process_flags(flags)

    retired = tmp_path / "sandbox_data" / "retired_modules" / "demo.py"  # path-ignore
    assert res == {"demo": "retired"}
    assert retired.exists()
    assert captured["results"] == {"demo": "retired"}
    assert gauge.count == 1.0

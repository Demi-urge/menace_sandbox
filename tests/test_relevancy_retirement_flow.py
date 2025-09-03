from __future__ import annotations

import sys
import types

# minimal sandbox settings stub
sandbox_settings = types.ModuleType("sandbox_settings")
sandbox_settings.SandboxSettings = lambda: types.SimpleNamespace(
    relevancy_threshold=10, relevancy_whitelist=[]
)
sys.modules.setdefault("sandbox_settings", sandbox_settings)


def test_relevancy_retirement_flow(tmp_path, monkeypatch):
    repo = tmp_path
    # create modules
    (repo / "used_module.py").write_text("def used():\n    return 42\n")
    (repo / "main.py").write_text("import used_module\n")
    (repo / "orphan.py").write_text("def orphan():\n    return 0\n")

    # ensure radar writes inside temp repo
    import relevancy_radar

    monkeypatch.setattr(relevancy_radar, "_BASE_DIR", repo, raising=False)
    monkeypatch.setattr(
        relevancy_radar, "_RELEVANCY_FLAGS_FILE", repo / "sandbox_data" / "flags.json"
    )
    monkeypatch.setattr(
        relevancy_radar,
        "_RELEVANCY_METRICS_FILE",
        repo / "sandbox_data" / "metrics.json",
    )
    monkeypatch.setattr(
        relevancy_radar, "update_relevancy_metrics", lambda flags: None
    )
    monkeypatch.setattr(
        relevancy_radar, "load_usage_stats", lambda: {"main": 30, "used_module": 30}
    )

    flags = relevancy_radar.RelevancyRadar.flag_unused_modules(
        ["main", "used_module", "orphan"]
    )
    assert flags == {"orphan": "retire"}

    fake_qfe = types.ModuleType("quick_fix_engine")
    fake_qfe.generate_patch = lambda path: 1
    monkeypatch.setitem(sys.modules, "quick_fix_engine", fake_qfe)

    import module_retirement_service

    monkeypatch.setattr(
        module_retirement_service, "update_module_retirement_metrics", lambda results: None
    )

    class DummyCounter:
        def __init__(self):
            self.count = 0

        def inc(self, amount: float = 1.0):
            self.count += amount

    dummy_counter = DummyCounter()
    monkeypatch.setattr(module_retirement_service, "retired_modules_total", dummy_counter)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    service = module_retirement_service.ModuleRetirementService(repo)
    service.process_flags(flags)

    archive = repo / "sandbox_data" / "retired_modules" / "orphan.py"
    assert archive.exists()
    assert not (repo / "orphan.py").exists()
    # non-flagged modules remain
    assert (repo / "used_module.py").exists()
    assert (repo / "main.py").exists()
    assert not (repo / "sandbox_data" / "retired_modules" / "used_module.py").exists()

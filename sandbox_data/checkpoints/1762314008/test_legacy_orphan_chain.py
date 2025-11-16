import json
import sys
import types
from dataclasses import dataclass as _dc
from pathlib import Path

import metrics_exporter
import orphan_analyzer
import pytest
from prometheus_client import REGISTRY

from sandbox_runner import cycle as cycle_mod
from tests.test_recursive_orphans import (
    _load_methods,
    DummyIndex,
    DummyLogger,
)

# ---------------------------------------------------------------------------
# Provide minimal stubs for optional dependencies used by sandbox_settings

pyd = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
sub.dataclass = _dc
pyd.BaseModel = type("BaseModel", (), {})
pyd.dataclasses = sub
sys.modules.setdefault("pydantic", pyd)
sys.modules.setdefault("pydantic.dataclasses", sub)

ps_mod = types.ModuleType("pydantic_settings")
ps_mod.BaseSettings = object
ps_mod.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", ps_mod)

yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
sys.modules.setdefault("jinja2", jinja_mod)

REGISTRY._names_to_collectors.clear()


# ---------------------------------------------------------------------------

def test_cycle_caches_legacy_modules(monkeypatch, tmp_path):
    (tmp_path / "a.py").write_text("import b\nimport old\n")  # path-ignore
    (tmp_path / "b.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "old.py").write_text("# legacy\nVALUE = 2\n")  # path-ignore

    def fake_discover(repo_path: str):
        return {
            "a": {"parents": [], "classification": "candidate"},
            "b": {"parents": ["a"], "classification": "candidate"},
            "old": {"parents": ["a"], "classification": "legacy"},
        }

    monkeypatch.setattr(cycle_mod, "discover_recursive_orphans", fake_discover)

    mod = types.ModuleType("scripts.discover_isolated_modules")
    mod.discover_isolated_modules = lambda root, *, recursive=True: []
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: "legacy" in p.name)

    calls: dict[str, object] = {}

    def fake_auto_include(mods, recursive=False, validate=False):
        calls["mods"] = list(mods)
        calls["recursive"] = recursive
        calls["validate"] = validate
        return object(), {"added": list(mods)}

    monkeypatch.setattr(cycle_mod, "auto_include_modules", fake_auto_include)

    cache: dict[str, object] = {}
    monkeypatch.setattr(
        cycle_mod,
        "append_orphan_cache",
        lambda repo, traces: cache.setdefault("traces", traces),
    )
    monkeypatch.setattr(
        cycle_mod,
        "append_orphan_classifications",
        lambda repo, classes: cache.setdefault("classes", classes),
    )
    monkeypatch.setattr(cycle_mod, "prune_orphan_cache", lambda *a, **k: None)

    class Settings:
        auto_include_isolated = True
        recursive_isolated = True

    ctx = types.SimpleNamespace(
        repo=tmp_path,
        settings=Settings(),
        module_map=set(),
        orphan_traces={},
    )

    cycle_mod.include_orphan_modules(ctx)

    assert sorted(calls["mods"]) == ["a.py", "b.py"]  # path-ignore
    assert calls["recursive"] and calls["validate"]
    assert ctx.module_map == {"a.py", "b.py"}  # path-ignore
    assert "old.py" not in calls["mods"]  # path-ignore
    assert "old.py" in cache["traces"]  # path-ignore
    assert cache["traces"]["old.py"]["classification"] == "legacy"  # path-ignore
    assert ctx.orphan_traces["old.py"]["classification"] == "legacy"  # path-ignore
    assert ctx.orphan_traces["old.py"]["redundant"] is True  # path-ignore


# ---------------------------------------------------------------------------

def test_update_orphan_modules_records_metrics(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import b\nimport old\n")  # path-ignore
    (repo / "b.py").write_text("VALUE = 1\n")  # path-ignore
    (repo / "old.py").write_text("# legacy\nVALUE = 2\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {}, "groups": {}}))
    (data_dir / "orphan_classifications.json").write_text(
        json.dumps({"old.py": {"classification": "legacy"}})  # path-ignore
    )

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_CLEAN_ORPHANS", "1")
    monkeypatch.chdir(repo)

    mod = types.ModuleType("scripts.discover_isolated_modules")
    mod.discover_isolated_modules = lambda root, *, recursive=True: ["a.py", "b.py", "old.py"]  # path-ignore
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    dep = types.ModuleType("sandbox_runner.dependency_utils")
    dep.collect_local_dependencies = lambda mods: mods
    monkeypatch.setitem(sys.modules, "sandbox_runner.dependency_utils", dep)

    _integrate_orphans, _update_orphans, _refresh_map, _test_orphans = _load_methods()

    calls: dict[str, list[list[str]]] = {"tested": [], "workflows": []}

    def fake_run_repo_section(repo_path, modules=None, return_details=False, **k):
        calls["tested"].append(sorted(modules or []))
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [0.0]},
        )
        details = {m: {"sec": [{"result": {"exit_code": 0}}]} for m in (modules or [])}
        return (tracker, details) if return_details else tracker

    sys.modules["sandbox_runner"].run_repo_section_simulations = fake_run_repo_section

    def fake_generate(mods):
        calls.setdefault("workflows", []).append(sorted(mods))
        return [1]

    def fake_try(mods):
        calls.setdefault("integrated", []).append(sorted(mods))

    def fake_auto_include(mods, recursive=False, validate=True):
        fake_generate(mods)
        fake_try(mods)
        return [1]

    env_mod = types.SimpleNamespace(
        auto_include_modules=fake_auto_include,
        try_integrate_into_workflows=fake_try,
    )
    _integrate_orphans.__globals__["environment"] = env_mod

    def fake_build_graph(root):
        class G:
            nodes = {"a", "b", "old"}

            def degree(self, n):
                return 1 if n in {"a", "b"} else 0

        return G()

    monkeypatch.setattr(orphan_analyzer, "build_import_graph", fake_build_graph)

    for g in (
        metrics_exporter.orphan_modules_reintroduced_total,
        metrics_exporter.orphan_modules_passed_total,
        metrics_exporter.orphan_modules_tested_total,
        metrics_exporter.orphan_modules_failed_total,
        metrics_exporter.orphan_modules_redundant_total,
        metrics_exporter.orphan_modules_legacy_total,
        metrics_exporter.orphan_modules_reclassified_total,
    ):
        g.set(0)

    for name in [
        "orphan_modules_reintroduced_total",
        "orphan_modules_passed_total",
        "orphan_modules_tested_total",
        "orphan_modules_failed_total",
        "orphan_modules_redundant_total",
        "orphan_modules_legacy_total",
        "orphan_modules_reclassified_total",
    ]:
        _test_orphans.__globals__[name] = getattr(metrics_exporter, name)
        _integrate_orphans.__globals__[name] = getattr(metrics_exporter, name)
        _update_orphans.__globals__[name] = getattr(metrics_exporter, name)

    index = DummyIndex(data_dir / "module_map.json")
    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
        _last_map_refresh=0.0,
        orphan_traces={},
    )
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._refresh_module_map = types.MethodType(_refresh_map, engine)
    engine._test_orphan_modules = types.MethodType(_test_orphans, engine)

    _update_orphans(engine)

    assert calls["tested"] and sorted(m[0] for m in calls["tested"]) == ["a.py", "b.py"]  # path-ignore
    assert calls["workflows"] and calls["workflows"][0] == ["a.py", "b.py"]  # path-ignore
    data = json.loads((data_dir / "module_map.json").read_text())
    assert set(data.get("modules", {})) == {"a.py", "b.py"}  # path-ignore
    assert "old.py" not in data.get("modules", {})  # path-ignore
    assert (data_dir / "orphan_modules.json").read_text() == "[]"
    classifications = json.loads((data_dir / "orphan_classifications.json").read_text())
    assert classifications["old.py"]["classification"] == "legacy"  # path-ignore

    assert metrics_exporter.orphan_modules_tested_total._value.get() == 2
    assert metrics_exporter.orphan_modules_reintroduced_total._value.get() == 4
    assert metrics_exporter.orphan_modules_failed_total._value.get() == 0
    assert metrics_exporter.orphan_modules_redundant_total._value.get() == 0
    assert metrics_exporter.orphan_modules_legacy_total._value.get() == 0
    assert metrics_exporter.orphan_modules_reclassified_total._value.get() == 0

import json
import sys
import types

import yaml
from dynamic_path_router import resolve_path


# ---------------------------------------------------------------------------

def test_orphan_cycle_integration(tmp_path, monkeypatch):
    resolve_path("sandbox_runner.py")
    repo = tmp_path
    (repo / "existing.py").write_text("X = 1\n")  # path-ignore
    (repo / "new_mod.py").write_text("Y = 2\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(
        json.dumps({"modules": {"existing": "existing.py"}, "groups": {}})  # path-ignore
    )

    calls: dict[str, object] = {}

    def fake_discover(root: str):
        calls["discover"] = True
        return {"new_mod": []}

    monkeypatch.setitem(
        sys.modules,
        "sandbox_runner.orphan_discovery",
        types.SimpleNamespace(discover_recursive_orphans=fake_discover),
    )

    def fake_run_tests():
        calls["tests"] = True

    def auto_include_modules(paths, recursive=True, router=None, context_builder=None):
        fake_run_tests()
        calls["auto_include"] = list(paths)
        return object(), {"added": list(paths)}

    def try_integrate_into_workflows(mods, router=None, context_builder=None):
        calls["workflows"] = list(mods)
        return mods

    env_mod = types.SimpleNamespace(
        auto_include_modules=auto_include_modules,
        try_integrate_into_workflows=try_integrate_into_workflows,
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    mg_mod = types.SimpleNamespace(
        ModuleSynergyGrapher=lambda root: types.SimpleNamespace(
            graph={},
            update_graph=lambda names: calls.setdefault("synergy", list(names)),
            build_graph=lambda repo: {},
        ),
        load_graph=lambda p: {},
    )
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    ic_mod = types.SimpleNamespace(
        IntentClusterer=lambda: types.SimpleNamespace(
            index_modules=lambda paths: calls.setdefault(
                "cluster", [str(p) for p in paths]
            )
        )
    )
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    from sandbox_runner.orphan_integration import post_round_orphan_scan

    added, syn_ok, cl_ok = post_round_orphan_scan(repo)

    assert calls["discover"] is True
    assert calls["tests"] is True
    assert calls["auto_include"] == [str(repo / "new_mod.py")]  # path-ignore
    assert calls["workflows"] == [str(repo / "new_mod.py")]  # path-ignore
    assert calls["synergy"] == ["new_mod"]
    assert calls["cluster"] == [str(repo / "new_mod.py")]  # path-ignore
    assert added == [str(repo / "new_mod.py")]  # path-ignore
    assert syn_ok and cl_ok

    metrics = yaml.safe_load((repo / "sandbox_metrics.yaml").read_text())
    assert metrics["extra_metrics"]["orphan_modules_added"] == 1.0

    log_path = repo / "sandbox_data" / "orphan_integration.log"
    log_entry = json.loads(log_path.read_text().splitlines()[-1])
    assert log_entry["modules"] == [str(repo / "new_mod.py")]  # path-ignore

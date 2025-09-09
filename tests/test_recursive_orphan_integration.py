import json
import sys
import types
from pathlib import Path


def test_recursive_orphan_integration(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "nested").mkdir()
    (repo / "nested/helper.py").write_text(  # path-ignore
        "def greet():\n    return 'hi'\n"
    )
    (repo / "orphan.py").write_text(  # path-ignore
        "from nested.helper import greet\n\n"
    )

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            classify_module=lambda p: "candidate",
            analyze_redundancy=lambda p: False,
        ),
    )

    from sandbox_runner.orphan_discovery import discover_recursive_orphans

    mapping = discover_recursive_orphans(str(repo))
    assert set(mapping) == {"orphan", "nested.helper"}
    assert mapping["nested.helper"]["parents"] == ["orphan"]

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    data_dir = repo / "sandbox_data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    (data_dir / "module_map.json").write_text("{}")

    import sandbox_runner.environment as env
    from context_builder_util import create_context_builder

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        def run_once(self):
            return {"failed": False}

    monkeypatch.setitem(
        sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=DummySTS)
    )

    calls: dict[str, list[str] | bool] = {}

    def fake_generate(mods, workflows_db="workflows.db", context_builder=None):
        calls["generate"] = list(mods)
        return [1]

    def fake_integrate(mods, context_builder=None):
        calls["integrate"] = list(mods)
        return [1]

    class DummyTracker:
        def __init__(self, mods):
            self.module_deltas = {m: [1.0] for m in mods}
            self.metrics_history = {"synergy_roi": [0.0]}
            self.cluster_map = {}

        def save_history(self, path: str) -> None:
            Path(path).write_text(json.dumps({"roi_history": [1.0]}))

    def fake_run():
        calls["run"] = True
        mods = calls.get("integrate", [])
        return DummyTracker(mods)

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)

    tracker, tested = env.auto_include_modules(
        ["orphan.py"],
        recursive=True,
        validate=True,
        context_builder=create_context_builder(),
    )  # path-ignore

    expected = {"orphan.py", "nested/helper.py"}  # path-ignore
    assert set(calls.get("generate", [])) == expected
    assert set(calls.get("integrate", [])) == expected
    assert calls.get("run") is True
    map_data = json.loads((data_dir / "module_map.json").read_text())
    modules_map = map_data.get("modules", map_data)
    assert expected.issubset(set(modules_map))
    assert (data_dir / "roi_history.json").exists()
    assert set(tracker.module_deltas) == expected
    assert set(tested["added"]) == expected
    assert tested["failed"] == []
    assert tested["redundant"] == []

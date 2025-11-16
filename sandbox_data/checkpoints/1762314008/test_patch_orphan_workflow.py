import json
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


def test_patch_introduces_new_module_included(tmp_path):
    """A repository patch adding a module is discovered next phase."""
    (tmp_path / "existing.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {"existing.py": 1}, "groups": {}}))  # path-ignore

    # Patch introduces a new module depending on existing one
    (tmp_path / "new_mod.py").write_text("import existing\n")  # path-ignore

    ROOT = Path(__file__).resolve().parents[2]
    os.environ["SANDBOX_DISCOVERY_WORKERS"] = "1"
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.orphan_discovery", ROOT / "sandbox_runner" / "orphan_discovery.py"  # path-ignore
    )
    od = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od)
    mapping = od.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))
    assert "new_mod" in mapping
    info = mapping["new_mod"]
    assert info.get("classification") == "candidate"
    assert info.get("redundant") is False


def test_update_orphan_modules_captures_import(tmp_path, monkeypatch):
    """Workflow evolution importing an orphan triggers immediate capture."""
    from tests.test_recursive_orphans import (
        _load_methods,
        DummyIndex,
        DummyLogger,
    )

    (tmp_path / "wf.py").write_text("import orphan\n")  # path-ignore
    (tmp_path / "orphan.py").write_text("VALUE = 1\n")  # path-ignore

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    _integrate_orphans, _update_orphan_modules, _refresh_module_map, _test_orphan_modules = _load_methods()

    ROOT = Path(__file__).resolve().parents[2]
    sr_mod = sys.modules["sandbox_runner"]
    sr_mod.__path__ = [str(ROOT / "sandbox_runner")]
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.dependency_utils", ROOT / "sandbox_runner" / "dependency_utils.py"  # path-ignore
    )
    dep_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dep_mod)
    sys.modules["sandbox_runner.dependency_utils"] = dep_mod

    from sandbox_runner.dependency_utils import collect_local_dependencies

    integrated: list[str] = []

    def fake_integrate(paths: list[str]) -> set[str]:
        integrated.extend(Path(p).name for p in paths)
        return {Path(p).name for p in paths}

    eng = types.SimpleNamespace(
        module_index=DummyIndex(),
        module_clusters={},
        logger=DummyLogger(),
    )
    eng._integrate_orphans = fake_integrate
    eng._refresh_module_map = types.MethodType(lambda self, modules=None: None, eng)
    eng._test_orphan_modules = lambda mods: set(mods)

    def simple_collect(mods):
        return collect_local_dependencies(mods)

    eng._collect_recursive_modules = types.MethodType(simple_collect, eng)

    calls: dict[str, list[str] | None] = {"auto": None}

    def fake_auto(mods, recursive=False, validate=False):
        calls["auto"] = list(mods)
        return types.SimpleNamespace(module_deltas={}), {"added": list(mods), "failed": [], "redundant": []}

    env = types.SimpleNamespace(auto_include_modules=fake_auto)
    _update_orphan_modules.__globals__["environment"] = env

    _update_orphan_modules(eng, [str(tmp_path / "wf.py")])  # path-ignore

    assert "orphan.py" in integrated  # path-ignore
    assert calls["auto"] and "wf.py" in calls["auto"] and "orphan.py" in calls["auto"]  # path-ignore

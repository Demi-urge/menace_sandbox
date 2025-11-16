import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans

ROOT = Path(__file__).resolve().parents[1]


def _load_auto_include(fake_generate, fake_try, fake_run):
    path = ROOT / "sandbox_runner" / "environment.py"  # path-ignore
    src = path.read_text()
    tree = ast.parse(src)
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "auto_include_modules"
    )
    mod_dict = {
        "generate_workflows_for_modules": fake_generate,
        "try_integrate_into_workflows": fake_try,
        "run_workflow_simulations": fake_run,
        "Iterable": Iterable,
        "Dict": Dict,
        "Any": Any,
        "ROITracker": type("ROITracker", (), {}),
        "__name__": "sandbox_runner.environment",
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0)
    module = ast.Module(body=[future, func], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), mod_dict)
    return mod_dict["auto_include_modules"]


def test_recursive_inclusion_autonomous(tmp_path, monkeypatch):
    # create modules: orphan -> helper -> nested; orphan -> legacy (redundant)
    (tmp_path / "orphan.py").write_text("import helper\nimport legacy\n")  # path-ignore
    (tmp_path / "helper.py").write_text("import nested\n")  # path-ignore
    (tmp_path / "nested.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "legacy.py").write_text("VALUE = 0\n")  # path-ignore

    # legacy module considered redundant
    def fake_analyze(path: Path) -> bool:
        return path.name == "legacy.py"  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", fake_analyze)

    # discovery should include orphan and its non-redundant deps
    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping == {
        "helper": {"parents": ["orphan"], "redundant": False},
        "legacy": {"parents": ["orphan"], "redundant": True},
        "nested": {"parents": ["helper"], "redundant": False},
        "orphan": {"parents": [], "redundant": False},
    }

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    generated: list[list[str]] = []
    integrated: list[list[str]] = []

    def fake_generate(mods):
        mods = sorted(mods)
        generated.append(mods)
        map_path = data_dir / "module_map.json"
        data = {"modules": {}, "groups": {}}
        if map_path.exists():
            data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][m] = 1
            data["groups"].setdefault("1", 1)
        map_path.write_text(json.dumps(data))
        return []

    def fake_try(mods):
        mods = sorted(mods)
        integrated.append(mods)
        return []

    def fake_run():
        class Tracker:
            module_deltas = {}
            metrics_history = {"synergy_roi": [0.0]}

            def save_history(self, path: str):
                Path(path).write_text("{}")

        return Tracker()

    auto_include = _load_auto_include(fake_generate, fake_try, fake_run)

    auto_include(["orphan.py"], recursive=True)  # path-ignore

    expected = ["helper.py", "nested.py", "orphan.py"]  # path-ignore
    assert generated and generated[0] == expected
    assert integrated and integrated[0] == expected

    map_data = json.loads((data_dir / "module_map.json").read_text())
    assert all(m in map_data["modules"] for m in expected)
    assert "legacy.py" not in map_data["modules"]  # path-ignore
    assert "legacy.py" not in generated[0]  # path-ignore

import ast
from pathlib import Path
from typing import Any, Dict, Iterable

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans

ROOT = Path(__file__).resolve().parents[1]


def _load_auto_include(fake_generate, fake_try, fake_run):
    path = ROOT / "sandbox_runner" / "environment.py"
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
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    module = ast.Module(body=[future, func], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), mod_dict)
    return mod_dict["auto_include_modules"]


def test_discover_recursive_orphans_includes_orphan_deps(monkeypatch, tmp_path):
    (tmp_path / "orphan1.py").write_text("import helper\nimport legacy\n")
    (tmp_path / "orphan2.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "legacy.py").write_text("VALUE = 1\n")

    def fake_analyze(path):
        return path.name == "legacy.py"

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", fake_analyze)

    result = discover_recursive_orphans(str(tmp_path))

    assert result == {
        "orphan1": [],
        "orphan2": [],
        "helper": ["orphan1", "orphan2"],
    }


def test_auto_include_modules_adds_orphan_deps(monkeypatch, tmp_path):
    (tmp_path / "orphan.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: False)
    mapping = discover_recursive_orphans(str(tmp_path))

    generated = []
    integrated = []
    simulated = []

    def fake_generate(mods):
        generated.append(list(mods))

    def fake_try(mods):
        integrated.append(list(mods))

    def fake_run():
        simulated.append(True)
        return object()

    auto_include = _load_auto_include(fake_generate, fake_try, fake_run)
    auto_include([f"{m}.py" for m in mapping.keys()])

    expected = sorted(f"{m}.py" for m in mapping.keys())
    assert generated and sorted(generated[0]) == expected
    assert integrated and sorted(integrated[0]) == expected
    assert simulated == [True]

import ast
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
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    module = ast.Module(body=[future, func], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), mod_dict)
    return mod_dict["auto_include_modules"]


def test_discover_recursive_orphans_includes_orphan_deps(monkeypatch, tmp_path):
    (tmp_path / "orphan1.py").write_text("import helper\nimport legacy\n")  # path-ignore
    (tmp_path / "orphan2.py").write_text("import helper\n")  # path-ignore
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "legacy.py").write_text("VALUE = 1\n")  # path-ignore

    def fake_classify(path):
        return "legacy" if path.name == "legacy.py" else "candidate"  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", fake_classify)

    result = discover_recursive_orphans(str(tmp_path))

    assert result == {
        "orphan1": {
            "parents": [],
            "classification": "candidate",
            "redundant": False,
        },
        "orphan2": {
            "parents": [],
            "classification": "candidate",
            "redundant": False,
        },
        "helper": {
            "parents": ["orphan1", "orphan2"],
            "classification": "candidate",
            "redundant": False,
        },
        "legacy": {
            "parents": ["orphan1"],
            "classification": "legacy",
            "redundant": True,
        },
    }


def test_auto_include_modules_adds_orphan_deps(monkeypatch, tmp_path):
    (tmp_path / "orphan.py").write_text("import helper\n")  # path-ignore
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore

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
    auto_include([f"{m}.py" for m in mapping.keys()])  # path-ignore

    expected = sorted(f"{m}.py" for m in mapping.keys())  # path-ignore
    assert generated and sorted(generated[0]) == expected
    assert integrated and sorted(integrated[0]) == expected
    assert simulated == [True]

import ast
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_test_method():
    path = ROOT / "self_improvement.py"  # path-ignore
    tree = ast.parse(path.read_text())
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_test_orphan_modules":
                    method = item
                    break
    assert method is not None
    mod_dict = {
        "os": os,
        "json": json,
        "math": __import__("math"),
        "Path": Path,
        "Iterable": __import__("typing").Iterable,
        "SandboxSettings": lambda: types.SimpleNamespace(
            test_redundant_modules=False
        ),
        "classify_module": lambda path, include_meta=True: ("candidate", {}),
        "analyze_redundancy": lambda p: False,
        "log_record": lambda **k: k,
        "environment": types.SimpleNamespace(
            auto_include_modules=lambda *a, **k: None
        ),
    }
    ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    code = ast.Module(body=[method], type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    return mod_dict["_test_orphan_modules"]


class DummyLogger:
    def info(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def test_module_fails_bad_scenario(monkeypatch, tmp_path):
    _test = _load_test_method()

    def fake_run(repo_path, modules=None, return_details=False, **k):
        m = modules[0]
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0]},
            metrics_history={"synergy_roi": [0.0]},
            scenario_synergy={
                "good": [{"synergy_roi": 0.5}],
                "bad": [{"synergy_roi": -0.1}],
            },
        )
        details = {
            m: {
                "good": [{"result": {"exit_code": 0}}],
                "bad": [{"result": {"exit_code": 0}}],
            }
        }
        return tracker, details

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    idx_mod = types.ModuleType("module_index_db")
    class DummyIndex:
        def __init__(self, *a, **k):
            pass
        def refresh(self, *a, **k):
            pass
        def get(self, name):
            return 1
    idx_mod.ModuleIndexDB = DummyIndex
    monkeypatch.setitem(sys.modules, "module_index_db", idx_mod)

    wf_mod = types.ModuleType("task_handoff_bot")
    class DummyWFDB:
        def __init__(self, path):
            pass
        def fetch(self, limit=1000):
            return []
    wf_mod.WorkflowDB = DummyWFDB
    monkeypatch.setitem(sys.modules, "task_handoff_bot", wf_mod)

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test, eng)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    res = eng._test_orphan_modules(["foo.py"])  # path-ignore
    assert res == set()
    assert eng.orphan_traces["foo.py"]["robustness"] == -0.1  # path-ignore
    scen = eng.orphan_traces["foo.py"]["scenarios"]  # path-ignore
    assert scen["good"]["roi"] == 0.5
    assert scen["bad"]["roi"] == -0.1
    assert scen["bad"]["failed"] is False


def test_module_records_worst_scenario(monkeypatch, tmp_path):
    _test = _load_test_method()

    def fake_run(repo_path, modules=None, return_details=False, **k):
        m = modules[0]
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0]},
            metrics_history={"synergy_roi": [0.0]},
            scenario_synergy={
                "s1": [{"synergy_roi": 0.5}],
                "s2": [{"synergy_roi": 0.2}],
            },
        )
        details = {
            m: {
                "s1": [{"result": {"exit_code": 0}}],
                "s2": [{"result": {"exit_code": 0}}],
            }
        }
        return tracker, details

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    idx_mod = types.ModuleType("module_index_db")
    class DummyIndex:
        def __init__(self, *a, **k):
            pass
        def refresh(self, *a, **k):
            pass
        def get(self, name):
            return 1
    idx_mod.ModuleIndexDB = DummyIndex
    monkeypatch.setitem(sys.modules, "module_index_db", idx_mod)

    wf_mod = types.ModuleType("task_handoff_bot")
    class DummyWFDB:
        def __init__(self, path):
            pass
        def fetch(self, limit=1000):
            return []
    wf_mod.WorkflowDB = DummyWFDB
    monkeypatch.setitem(sys.modules, "task_handoff_bot", wf_mod)

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test, eng)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    res = eng._test_orphan_modules(["bar.py"])  # path-ignore
    assert res == {"bar.py"}  # path-ignore
    assert eng.orphan_traces["bar.py"]["robustness"] == 0.2  # path-ignore
    scen = eng.orphan_traces["bar.py"]["scenarios"]  # path-ignore
    assert scen["s1"]["roi"] == 0.5
    assert scen["s2"]["roi"] == 0.2


def test_combined_scenario_split(monkeypatch, tmp_path):
    _test = _load_test_method()

    def fake_run(repo_path, modules=None, return_details=False, **k):
        m = modules[0]
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0]},
            metrics_history={"synergy_roi": [0.0]},
            scenario_synergy={
                "high_latency_api+hostile_input": [{"synergy_roi": 0.3}]
            },
        )
        details = {
            m: {
                "high_latency_api+hostile_input": [
                    {"result": {"exit_code": 0}}
                ]
            }
        }
        return tracker, details

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    idx_mod = types.ModuleType("module_index_db")
    class DummyIndex:
        def __init__(self, *a, **k):
            pass
        def refresh(self, *a, **k):
            pass
        def get(self, name):
            return 1
    idx_mod.ModuleIndexDB = DummyIndex
    monkeypatch.setitem(sys.modules, "module_index_db", idx_mod)

    wf_mod = types.ModuleType("task_handoff_bot")
    class DummyWFDB:
        def __init__(self, path):
            pass
        def fetch(self, limit=1000):
            return []
    wf_mod.WorkflowDB = DummyWFDB
    monkeypatch.setitem(sys.modules, "task_handoff_bot", wf_mod)

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test, eng)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    res = eng._test_orphan_modules(["baz.py"])  # path-ignore
    assert res == {"baz.py"}  # path-ignore
    scen = eng.orphan_traces["baz.py"]["scenarios"]  # path-ignore
    assert scen["high_latency_api"]["roi"] == 0.3
    assert scen["hostile_input"]["roi"] == 0.3

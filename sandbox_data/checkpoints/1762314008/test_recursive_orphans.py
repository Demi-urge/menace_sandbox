import ast
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_methods():
    path = ROOT / "self_improvement.py"  # path-ignore
    src = path.read_text()
    tree = ast.parse(src)
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_integrate_orphans",
                    "_update_orphan_modules",
                    "_refresh_module_map",
                    "_test_orphan_modules",
                }:
                    methods.append(item)
    def _run_repo_section_simulations(repo_path, modules=None, return_details=False, **k):
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [0.0]},
        )
        details = {m: {"sec": [{"result": {"exit_code": 0}}]} for m in (modules or [])}
        return (tracker, details) if return_details else tracker

    sr_mod = types.ModuleType("sandbox_runner")
    sr_mod.run_repo_section_simulations = _run_repo_section_simulations
    sr_mod.discover_recursive_orphans = lambda repo, module_map=None: {}
    sys.modules["sandbox_runner"] = sr_mod

    mod_dict = {
        "os": os,
        "json": json,
        "Path": Path,
        "time": __import__("time"),
        "datetime": __import__("datetime").datetime,
        "Iterable": __import__("typing").Iterable,
        "math": __import__("math"),
        "SandboxSettings": lambda: types.SimpleNamespace(
            test_redundant_modules=False
        ),
        "classify_module": lambda path, include_meta=True: ("candidate", {}),
        "build_module_map": lambda repo, ignore=None: {},
        "generate_workflows_for_modules": lambda mods, context_builder=None: None,
        "try_integrate_into_workflows": lambda mods, context_builder=None: None,
        "run_workflow_simulations": lambda *a, **k: None,
        "run_repo_section_simulations": _run_repo_section_simulations,
        "log_record": lambda **k: k,
        "analyze_redundancy": lambda p: False,
    }
    def auto_include_modules(mods, recursive=False, context_builder=None):
        generate_workflows_for_modules(mods, context_builder=context_builder)
        try_integrate_into_workflows(mods, context_builder=context_builder)
        run_workflow_simulations(context_builder=context_builder)
    mod_dict["auto_include_modules"] = auto_include_modules
    mod_dict["environment"] = types.SimpleNamespace(
        auto_include_modules=auto_include_modules
    )
    ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    code = ast.Module(body=methods, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    return (
        mod_dict["_integrate_orphans"],
        mod_dict["_update_orphan_modules"],
        mod_dict["_refresh_module_map"],
        mod_dict["_test_orphan_modules"],
    )


_integrate_orphans, _update_orphan_modules, _refresh_module_map, _test_orphan_modules = _load_methods()


class DummyIndex:
    def __init__(self, path: Path | None = None) -> None:
        self._map = {}
        self._groups = {}
        self.path = path

    def refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[Path(m).name] = 1
            self._groups[str(1)] = 1

    def get(self, name):
        return 1

    def save(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"modules": self._map, "groups": self._groups}, fh)


class DummyLogger:
    def exception(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


def test_update_orphan_modules_recursive(monkeypatch, tmp_path):
    called = {}
    sr = types.ModuleType("sandbox_runner")

    def discover(repo_path: str, module_map=None):
        called["used"] = True
        called["repo"] = repo_path
        called["map"] = module_map
        return {"foo.bar": []}

    sr.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    integrated: dict[str, list[str]] = {}

    def fake_integrate(paths: list[str]) -> set[str]:
        integrated["paths"] = list(paths)
        return {Path(p).name for p in paths}

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._integrate_orphans = fake_integrate
    eng._test_orphan_modules = lambda mods: set(mods)

    _update_orphan_modules(eng)

    assert called.get("used") is True
    assert Path(called["repo"]) == tmp_path
    assert Path(called["map"]).resolve() == (tmp_path / "module_map.json").resolve()
    assert integrated.get("paths") == [str(tmp_path / "foo/bar.py")]  # path-ignore
    data = json.loads((tmp_path / "orphan_modules.json").read_text())
    assert data == []


def test_clean_orphans_removes_passing(monkeypatch, tmp_path):
    sr = types.ModuleType("sandbox_runner")
    sr.discover_recursive_orphans = (
        lambda repo, module_map=None: {"foo": [], "bar": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_CLEAN_ORPHANS", "1")

    mods_path = tmp_path / "orphan_modules.json"
    mods_path.write_text(json.dumps(["foo.py", "bar.py"]))  # path-ignore

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._integrate_orphans = lambda paths: {Path(p).name for p in paths}

    def fake_test(mods: list[str]):
        return [m for m in mods if Path(m).stem == "foo"]

    eng._test_orphan_modules = fake_test

    _update_orphan_modules(eng)

    data = json.loads(mods_path.read_text())
    assert data == ["bar.py"]  # path-ignore


def test_update_orphan_modules_with_modules(monkeypatch, tmp_path):
    index = DummyIndex()
    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    calls = {"refresh": 0, "integrated": None, "auto": None}

    def fake_integrate(paths: list[str]) -> set[str]:
        calls["integrated"] = list(paths)
        return {Path(p).name for p in paths}

    def fake_refresh(self, modules=None):
        calls["refresh"] += 1

    eng._integrate_orphans = fake_integrate
    eng._refresh_module_map = types.MethodType(fake_refresh, eng)
    eng._test_orphan_modules = lambda mods: set(mods)
    eng._collect_recursive_modules = lambda mods: set(mods)

    env = types.SimpleNamespace(
        auto_include_modules=lambda mods, recursive=False, validate=False: calls.__setitem__(
            "auto", list(mods)
        )
    )
    _update_orphan_modules.__globals__["environment"] = env

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    (tmp_path / "orphan_modules.json").write_text(json.dumps(["foo.py"]))  # path-ignore

    _update_orphan_modules(eng, ["foo.py"])  # path-ignore

    assert calls["integrated"] == [str(tmp_path / "foo.py")]  # path-ignore
    assert calls["refresh"] == 1
    assert json.loads((tmp_path / "orphan_modules.json").read_text()) == []
    assert calls["auto"] == ["foo.py"]  # path-ignore


def test_refresh_module_map_triggers_update(monkeypatch, tmp_path):
    index = DummyIndex()
    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
        _last_map_refresh=0.0,
        auto_refresh_map=True,
        meta_logger=None,
        patch_db=None,
        data_bot=None,
    )

    calls = {"update": 0}

    def fake_update(self):
        calls["update"] += 1

    eng._update_orphan_modules = types.MethodType(fake_update, eng)
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)
    eng._collect_recursive_modules = lambda mods: set(mods)

    _refresh_module_map(eng, ["foo.py"])  # path-ignore

    # integration should occur via ``_integrate_orphans`` but orphan discovery
    # is postponed until explicitly triggered
    assert calls["update"] == 0


def test_isolated_modules_refresh_map(monkeypatch, tmp_path):
    index = DummyIndex(tmp_path / "module_map.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("pass\n")  # path-ignore

    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path, *, recursive=True):
        assert Path(path) == repo
        return ["iso.py"]  # path-ignore

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._refresh_module_map = types.MethodType(_refresh_module_map, eng)
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)
    eng._collect_recursive_modules = lambda mods: set(mods)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = (
        lambda repo_path, modules=None, return_details=False, **k: (
            (
                types.SimpleNamespace(
                    module_deltas={m: [1.0] for m in (modules or [])},
                    metrics_history={"synergy_roi": [0.0]},
                ),
                {m: {"sec": [{"result": {"exit_code": 0}}]} for m in modules or []},
            )
            if return_details
            else types.SimpleNamespace(
                module_deltas={m: [1.0] for m in (modules or [])},
                metrics_history={"synergy_roi": [0.0]},
            )
        )
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    _update_orphan_modules(eng)

    map_file = tmp_path / "module_map.json"
    assert map_file.exists()
    data = json.loads(map_file.read_text())
    assert data["modules"].get("iso.py") == 1  # path-ignore
    mods_path = tmp_path / "orphan_modules.json"
    assert json.loads(mods_path.read_text()) == []


def test_recursive_isolated(monkeypatch, tmp_path):
    index = DummyIndex(tmp_path / "module_map.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("pass\n")  # path-ignore

    called = {}
    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path, *, recursive=True):
        called["recursive"] = recursive
        assert Path(path) == repo
        return ["iso.py"]  # path-ignore

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._refresh_module_map = types.MethodType(_refresh_module_map, eng)
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)
    eng._collect_recursive_modules = lambda mods: set(mods)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    # disable recursion via environment variable
    monkeypatch.setenv("SANDBOX_RECURSIVE_ISOLATED", "0")

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = (
        lambda repo_path, modules=None, return_details=False, **k: (
            (
                types.SimpleNamespace(
                    module_deltas={m: [1.0] for m in (modules or [])},
                    metrics_history={"synergy_roi": [0.0]},
                ),
                {m: {"sec": [{"result": {"exit_code": 0}}]} for m in modules or []},
            )
            if return_details
            else types.SimpleNamespace(
                module_deltas={m: [1.0] for m in (modules or [])},
                metrics_history={"synergy_roi": [0.0]},
            )
        )
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    _update_orphan_modules(eng)

    map_file = tmp_path / "module_map.json"
    assert map_file.exists()
    data = json.loads(map_file.read_text())
    assert data["modules"].get("iso.py") == 1  # path-ignore
    mods_path = tmp_path / "orphan_modules.json"
    assert json.loads(mods_path.read_text()) == []
    assert called.get("recursive") is False


def test_refresh_map_skips_failing_modules(monkeypatch, tmp_path):
    index = DummyIndex()

    class LogLogger(DummyLogger):
        def __init__(self):
            self.info_calls: list[tuple[str, dict | None]] = []

        def info(self, msg, extra=None):
            self.info_calls.append((msg, extra))

    logger = LogLogger()
    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=logger,
        _last_map_refresh=0.0,
        auto_refresh_map=True,
        meta_logger=None,
        patch_db=None,
        data_bot=None,
    )
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._collect_recursive_modules = lambda mods: set(mods)
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)

    sr = types.ModuleType("sandbox_runner")

    def fake_run(repo_path, modules=None, return_details=False, **k):
        details = {
            m: {"sec": [{"result": {"exit_code": 1 if m == "bad.py" else 0}}]}  # path-ignore
            for m in modules or []
        }
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [0.0]},
        )
        return (tracker, details) if return_details else tracker

    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    _refresh_module_map(eng, ["good.py", "bad.py"])  # path-ignore

    # only good.py should be integrated  # path-ignore
    assert "good.py" in eng.module_clusters  # path-ignore
    assert "bad.py" not in eng.module_clusters  # path-ignore
    assert any("bad.py" in (extra or {}).get("module", "") for _, extra in logger.info_calls)  # path-ignore

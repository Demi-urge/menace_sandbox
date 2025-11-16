import ast
import json
import types
import sys
from pathlib import Path
from dynamic_path_router import resolve_path
import pytest


class DummyLogger:
    def __init__(self):
        self.records: list[tuple[str, str, dict | None]] = []

    def info(self, msg, *args, **kwargs):
        self.records.append(("info", msg % args if args else msg, kwargs.get("extra")))

    def error(self, msg, *args, **kwargs):
        self.records.append(("error", msg % args if args else msg, kwargs.get("extra")))

    def exception(self, msg, *args, **kwargs):
        self.records.append(("exception", msg % args if args else msg, kwargs.get("extra")))


def _load_integrator(repo: Path, data_dir: Path):
    src = resolve_path("self_improvement.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_integrate_orphans"
    )
    module = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), method],
        type_ignores=[],
    )
    module = ast.fix_missing_locations(module)
    from typing import Iterable  # local import to avoid global dependency
    import time

    ns: dict[str, object] = {
        "json": json,
        "Path": Path,
        "log_record": lambda **f: f,
        "classify_module": lambda path: "candidate",
        "orphan_modules_reintroduced_total": types.SimpleNamespace(inc=lambda n: None),
        "GLOBAL_ROUTER": None,
        "Iterable": Iterable,
        "time": time,
    }

    class _StubSettings:
        def __init__(self):
            self.sandbox_repo_path = str(repo)
            self.sandbox_data_dir = str(data_dir)
            self.test_redundant_modules = False

    ns["SandboxSettings"] = _StubSettings
    ns["_repo_path"] = lambda: repo
    ns["_data_dir"] = lambda: data_dir

    exec(compile(module, "<ast>", "exec"), ns)
    return ns["_integrate_orphans"]


def _build_engine(monkeypatch, tmp_path, metrics_db, tracker):
    repo = tmp_path / "repo"
    repo.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    mod_path = repo / resolve_path("a.py")
    mod_path.write_text("print('hi')\n")

    integrator = _load_integrator(repo, data_dir)

    monkeypatch.setitem(
        sys.modules,
        "scripts.discover_isolated_modules",
        types.SimpleNamespace(discover_isolated_modules=lambda *a, **k: set()),
    )
    monkeypatch.setitem(
        sys.modules,
        "sandbox_runner.dependency_utils",
        types.SimpleNamespace(collect_local_dependencies=lambda mods, **k: set(mods)),
    )

    logger = DummyLogger()
    engine = types.SimpleNamespace(
        module_index=types.SimpleNamespace(
            refresh=lambda *a, **k: None,
            save=lambda: None,
            get=lambda m: 1,
        ),
        module_clusters={},
        logger=logger,
        orphan_traces={},
        intent_clusterer=None,
        _sandbox_integrate=lambda repo_path, modules, logger, router: (
            tracker,
            len(modules),
            [],
            None,
            None,
        ),
        _update_orphan_modules=lambda: None,
        pre_roi_bot=None,
        data_bot=types.SimpleNamespace(metrics_db=metrics_db),
        tracker=tracker,
    )
    return integrator, engine, logger, mod_path


def test_metrics_db_failure(monkeypatch, tmp_path):
    class FailingDB:
        def log_eval(self, *a, **k):
            raise RuntimeError("db fail")

    tracker = types.SimpleNamespace(
        roi_history=[],
        register_metrics=lambda *a, **k: None,
        update=lambda *a, **k: None,
    )
    integrator, eng, logger, mod = _build_engine(monkeypatch, tmp_path, FailingDB(), tracker)

    with pytest.raises(RuntimeError) as exc:
        integrator(eng, [str(mod)])

    assert "db fail" in str(exc.value)
    assert any(
        "metrics db logging failed" in r[1] and str(resolve_path("a.py")) in r[1]
        for r in logger.records
    )


def test_tracker_failure(monkeypatch, tmp_path):
    class DummyDB:
        def log_eval(self, *a, **k):
            return None

    class FailingTracker:
        roi_history = []

        def register_metrics(self, *a, **k):
            raise RuntimeError("tracker down")

        def update(self, *a, **k):  # pragma: no cover - not expected
            pass

    tracker = FailingTracker()
    integrator, eng, logger, mod = _build_engine(monkeypatch, tmp_path, DummyDB(), tracker)

    with pytest.raises(RuntimeError) as exc:
        integrator(eng, [str(mod)])

    assert "tracker down" in str(exc.value)
    assert any("tracker metric update failed" in r[1] for r in logger.records)

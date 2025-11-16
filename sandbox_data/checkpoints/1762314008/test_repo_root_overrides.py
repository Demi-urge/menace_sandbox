import importlib
import importlib
import sys
import types
from pathlib import Path

import pytest
from context_builder_util import create_context_builder


class ContextBuilder:
    def build_context(self, *a, **k):
        return {}


def test_environment_respects_sandbox_repo_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "submodule"
    (sub / ".git").mkdir(parents=True)
    mod = sub / "helper.py"  # path-ignore
    mod.write_text("def hi():\n    return 'hi'\n")

    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    import dynamic_path_router as dpr
    dpr = importlib.reload(dpr)
    dpr.clear_cache()

    records = []

    class DummyWorkflowDB:
        def __init__(self, path, router=None):
            self.path = Path(path)
        def add(self, rec):
            records.append(rec)
            return len(records)

    class DummyRecord:
        def __init__(self, workflow, title, dependencies=None, reasons=None):
            self.workflow = workflow

    sys.modules["menace.task_handoff_bot"] = types.SimpleNamespace(
        WorkflowDB=DummyWorkflowDB, WorkflowRecord=DummyRecord
    )
    sys.modules["module_index_db"] = types.SimpleNamespace(
        ModuleIndexDB=lambda: types.SimpleNamespace(get=lambda self, m: 0)
    )

    class _Metric:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            return None

    class _Metrics(types.SimpleNamespace):
        def __getattr__(self, name):  # pragma: no cover - generic metric stub
            return _Metric()

    sys.modules["metrics_exporter"] = _Metrics(
        orphan_modules_side_effects_total=_Metric()
    )

    env = importlib.reload(importlib.import_module("sandbox_runner.environment"))
    called = []
    monkeypatch.setattr(env, "integrate_new_orphans", lambda repo, router=None: called.append(repo))

    env.try_integrate_into_workflows(
        [str(mod.resolve())], context_builder=create_context_builder()
    )

    assert called and called[0] == repo.resolve()
    assert records and records[0].workflow == ["submodule.helper"]


def test_self_coding_engine_respects_menace_root(tmp_path, monkeypatch):
    root = tmp_path / "alt_root"
    (root / ".git").mkdir(parents=True)
    (root / "chunk_summary_cache").mkdir()
    data = root / "sandbox_data"
    data.mkdir()

    monkeypatch.setenv("MENACE_ROOT", str(root))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))

    import dynamic_path_router as dpr
    dpr = importlib.reload(dpr)
    dpr.clear_cache()

    sce = importlib.reload(importlib.import_module("self_coding_engine"))

    class Dummy: ...

    eng = sce.SelfCodingEngine(Dummy(), Dummy(), context_builder=ContextBuilder())
    eng.failure_similarity_tracker.update(similarity=0.5)
    eng._save_state()

    assert eng.chunk_summary_cache_dir == root / "chunk_summary_cache"
    assert eng._state_path.parent == data
    assert eng._state_path.exists()


def test_cli_resolves_paths_under_repo(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    wf = repo / "workflows.db"
    wf.write_text("hello")

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    import dynamic_path_router as dpr
    dpr = importlib.reload(dpr)
    dpr.clear_cache()

    cli = importlib.reload(importlib.import_module("sandbox_runner.cli"))

    path = cli.resolve_path("workflows.db")
    assert path == wf
    assert path.read_text() == "hello"
    path.write_text("bye")
    assert wf.read_text() == "bye"

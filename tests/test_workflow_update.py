import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(monkeypatch=None):
    pkg = sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
    pkg.__path__ = [str(ROOT / "sandbox_runner")]
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment", ROOT / "sandbox_runner" / "environment.py"
    )
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    if monkeypatch is not None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    if monkeypatch is not None:
        monkeypatch.setattr(env, "purge_leftovers", lambda: None)
    return env


def _load_thb():
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        ROOT / "task_handoff_bot.py",
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    assert spec and spec.loader
    spec.loader.exec_module(thb)  # type: ignore[attr-defined]
    return thb


class StubIndex:
    def __init__(self, *a, **k):
        self._map = {"a.py": 1, "b.py": 1, "c.py": 2}
        self.tags = {}

    def get(self, name):
        return self._map.get(name, 0)

    def get_tags(self, name):
        return self.tags.get(name, [])

    def set_tags(self, name, tags):
        self.tags[name] = list(tags)


def test_try_integrate_into_workflows(tmp_path, monkeypatch):
    env = _load_env(monkeypatch)
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    wf_db = thb.WorkflowDB(db_path)
    wf1 = thb.WorkflowRecord(workflow=["a"], title="w1")
    wid1 = wf_db.add(wf1)
    wf2 = thb.WorkflowRecord(workflow=["c"], title="w2")
    wf_db.add(wf2)

    stub = types.ModuleType("module_index_db")
    stub.ModuleIndexDB = StubIndex
    monkeypatch.setitem(sys.modules, "module_index_db", stub)

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        async def _test_orphan_modules(self, paths):
            return set(paths), set(), {}

    sts_stub = types.ModuleType("self_test_service")
    sts_stub.SelfTestService = DummySTS
    monkeypatch.setitem(sys.modules, "self_test_service", sts_stub)

    class DummyClusterer:
        def __init__(self):
            root = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
            self.clusters = {
                str(root / "a.py"): [1],
                str(root / "b.py"): [1],
            }

    clusterer = DummyClusterer()

    updated = env.try_integrate_into_workflows(
        ["b.py"], workflows_db=db_path, intent_clusterer=clusterer
    )
    recs = {r.wid: r for r in wf_db.fetch(limit=10)}
    assert wid1 in updated
    assert "b" in recs[wid1].workflow
    assert "b" not in recs[wf2.wid].workflow


def test_try_integrate_no_match(tmp_path, monkeypatch):
    env = _load_env(monkeypatch)
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    wf_db = thb.WorkflowDB(db_path)
    wf_db.add(thb.WorkflowRecord(workflow=["a"], title="w1"))

    stub = types.ModuleType("module_index_db")
    stub.ModuleIndexDB = StubIndex
    monkeypatch.setitem(sys.modules, "module_index_db", stub)

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        async def _test_orphan_modules(self, paths):
            return set(paths), set(), {}

    sts_stub = types.ModuleType("self_test_service")
    sts_stub.SelfTestService = DummySTS
    monkeypatch.setitem(sys.modules, "self_test_service", sts_stub)

    updated = env.try_integrate_into_workflows(["d.py"], workflows_db=db_path)
    rec = wf_db.fetch(limit=10)[0]
    assert not updated
    assert rec.workflow == ["a"]


def test_try_integrate_duplicate_filenames(tmp_path, monkeypatch):
    env = _load_env(monkeypatch)
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    wf_db = thb.WorkflowDB(db_path)
    wf = thb.WorkflowRecord(workflow=["existing"], title="w")
    wid = wf_db.add(wf)

    class StubDupIndex:
        def __init__(self, *a, **k):
            self._map = {
                "existing.py": 1,
                "pkg1/orphan.py": 1,
                "pkg2/orphan.py": 1,
            }
            self.tags = {}

        def get(self, name):
            return self._map.get(name, 0)

        def get_tags(self, name):
            return self.tags.get(name, [])

        def set_tags(self, name, tags):
            self.tags[name] = list(tags)

    stub = types.ModuleType("module_index_db")
    stub.ModuleIndexDB = StubDupIndex
    monkeypatch.setitem(sys.modules, "module_index_db", stub)

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        async def _test_orphan_modules(self, paths):
            return set(paths), set(), {}

    sts_stub = types.ModuleType("self_test_service")
    sts_stub.SelfTestService = DummySTS
    monkeypatch.setitem(sys.modules, "self_test_service", sts_stub)

    mods = ["pkg1/orphan.py", "pkg2/orphan.py"]
    updated = env.try_integrate_into_workflows(mods, workflows_db=db_path)
    rec = {r.wid: r for r in wf_db.fetch(limit=10)}[wid]
    assert wid in updated
    assert set(rec.workflow) == {"existing", "pkg1.orphan", "pkg2.orphan"}


def test_try_integrate_intent_synergy(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    env = _load_env(monkeypatch)
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    wf_db = thb.WorkflowDB(db_path)
    wf = thb.WorkflowRecord(workflow=["a"], title="w")
    wid = wf_db.add(wf)

    (tmp_path / "a.py").write_text("def a():\n    pass\n")
    (tmp_path / "b.py").write_text("def b():\n    pass\n")

    stub = types.ModuleType("module_index_db")
    stub.ModuleIndexDB = StubIndex
    monkeypatch.setitem(sys.modules, "module_index_db", stub)

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        async def _test_orphan_modules(self, paths):
            return set(paths), set(), {}

    sts_stub = types.ModuleType("self_test_service")
    sts_stub.SelfTestService = DummySTS
    monkeypatch.setitem(sys.modules, "self_test_service", sts_stub)

    class DummyClusterer:
        def __init__(self):
            self.clusters = {
                str((tmp_path / "a.py").resolve()): [1],
                str((tmp_path / "b.py").resolve()): [2],
            }

    clusterer = DummyClusterer()
    monkeypatch.setattr(env, "_USE_MODULE_SYNERGY", True)
    monkeypatch.setattr(
        env,
        "get_synergy_cluster",
        lambda name, threshold=0.7, bfs=False: {name, "a"} if name == "b" else {name},
    )

    updated = env.try_integrate_into_workflows(
        ["b.py"],
        workflows_db=db_path,
        intent_clusterer=clusterer,
        intent_threshold=0.5,
    )
    recs = {r.wid: r for r in wf_db.fetch(limit=10)}
    assert wid in updated
    assert "b" in recs[wid].workflow


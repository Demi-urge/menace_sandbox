import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    pkg = sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
    pkg.__path__ = [str(ROOT / "sandbox_runner")]
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment", ROOT / "sandbox_runner" / "environment.py"
    )
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
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

    def get(self, name):
        return self._map.get(name, 0)


def test_try_integrate_into_workflows(tmp_path, monkeypatch):
    env = _load_env()
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

    updated = env.try_integrate_into_workflows(["b.py"], workflows_db=db_path)
    recs = {r.wid: r for r in wf_db.fetch(limit=10)}
    assert wid1 in updated
    assert "b" in recs[wid1].workflow
    assert "b" not in recs[wf2.wid].workflow


def test_try_integrate_no_match(tmp_path, monkeypatch):
    env = _load_env()
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    wf_db = thb.WorkflowDB(db_path)
    wf_db.add(thb.WorkflowRecord(workflow=["a"], title="w1"))

    stub = types.ModuleType("module_index_db")
    stub.ModuleIndexDB = StubIndex
    monkeypatch.setitem(sys.modules, "module_index_db", stub)
    updated = env.try_integrate_into_workflows(["d.py"], workflows_db=db_path)
    rec = wf_db.fetch(limit=10)[0]
    assert not updated
    assert rec.workflow == ["a"]


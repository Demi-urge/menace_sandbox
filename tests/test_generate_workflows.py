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


def test_generate_workflows(tmp_path):
    env = _load_env()
    thb = _load_thb()
    db_path = tmp_path / "wf.db"
    ids = env.generate_workflows_for_modules(["foo.py"], workflows_db=db_path)
    db = thb.WorkflowDB(db_path)
    recs = db.fetch()
    assert ids and ids[0] == recs[0].wid
    assert recs[0].workflow == ["foo"]

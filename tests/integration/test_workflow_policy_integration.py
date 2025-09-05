import os
import sys
import sqlite3
import pytest
import urllib.request
import tempfile
import importlib.util
import types

from dynamic_path_router import resolve_dir, resolve_path, repo_root

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dynamically load WorkflowSandboxRunner without importing the heavy package
package_path = resolve_dir("sandbox_runner")
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner", resolve_path("workflow_sandbox_runner.py")  # path-ignore
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner

from self_improvement_policy import SelfImprovementPolicy  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_sqlite(monkeypatch):
    orig = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_kw: orig(":memory:"))


def test_runner_and_policy_learning():
    captured: list[Path] = []
    original_open = open

    def record_open(path, mode="r", *a, **kw):
        p = Path(path)
        captured.append(p)
        return original_open(path, mode, *a, **kw)

    def allow_write_text(path, data, *a, **kw):
        return Path.write_text(path, data, *a, **kw)

    def llm_step():
        data = urllib.request.urlopen("http://llm.test/predict").read().decode()
        with open("artifact.txt", "w") as f:
            f.write("data")
        return float(data)

    def db_step():
        con = sqlite3.connect("sample.db")  # noqa: SQL001
        con.execute("CREATE TABLE t (v REAL)")
        con.execute("INSERT INTO t VALUES (1.0)")
        con.commit()
        val = con.execute("SELECT SUM(v) FROM t").fetchone()[0]
        con.close()
        return val

    runner = WorkflowSandboxRunner()
    metrics1 = runner.run(
        [llm_step, db_step],
        safe_mode=True,
        test_data={"http://llm.test/predict": "0.5"},
        fs_mocks={"open": record_open, "pathlib.Path.write_text": allow_write_text},
    )

    telemetry = runner.telemetry
    assert telemetry is not None
    assert set(telemetry["time_per_module"]) == {"llm_step", "db_step"}
    assert telemetry["crash_frequency"] == pytest.approx(0.0)
    assert not Path("artifact.txt").exists()
    assert captured and all(str(p).startswith(tempfile.gettempdir()) for p in captured)

    policy = SelfImprovementPolicy(alpha=1.0, gamma=0.0, epsilon=0.0)
    state = (0,)
    q1 = policy.score(state)
    reward1 = metrics1.modules[0].result + metrics1.modules[1].result
    policy.update(state, reward1)
    q2 = policy.score(state)

    metrics2 = runner.run(
        [llm_step, db_step],
        safe_mode=True,
        test_data={"http://llm.test/predict": "1.5"},
        fs_mocks={"open": record_open, "pathlib.Path.write_text": allow_write_text},
    )
    reward2 = metrics2.modules[0].result + metrics2.modules[1].result
    policy.update(state, reward2)
    q3 = policy.score(state)

    assert q2 > q1
    assert q3 > q2

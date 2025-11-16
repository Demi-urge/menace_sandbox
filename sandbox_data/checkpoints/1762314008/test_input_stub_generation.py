import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import importlib  # noqa: E402
import sandbox_runner.input_history_db as ih  # noqa: E402
import sandbox_runner.environment as env  # noqa: E402
import sandbox_runner as sr  # noqa: E402


def test_aggregate_history_stubs(tmp_path, monkeypatch):
    db = ih.InputHistoryDB(tmp_path / "hist.db")
    db.add({"a": 1, "mode": "x"})
    db.add({"a": 3, "mode": "x"})
    db.add({"a": 5, "mode": "y"})
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db.path))
    importlib.reload(env)
    agg = env.aggregate_history_stubs()
    assert agg == {"a": 3, "mode": "x"}


def test_generate_input_stubs_use_history(monkeypatch, tmp_path):
    db_path = tmp_path / "hist.db"
    db = ih.InputHistoryDB(db_path)
    db.add({"level": 1})
    db.add({"level": 3})
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "templates")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    importlib.reload(env)
    expected = env.aggregate_history_stubs()
    import sandbox_runner.generative_stub_provider as gsp  # noqa: E402

    def fail(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gsp, "generate_stubs", fail)
    monkeypatch.setattr(env, "_load_history", lambda p: [])
    stubs = env.generate_input_stubs(1)
    assert stubs == [expected]


def test_generate_input_stubs_misuse_env(monkeypatch, tmp_path):
    class DummyLogger:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(
        sys.modules, "error_logger", types.SimpleNamespace(ErrorLogger=DummyLogger)
    )
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "random")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    monkeypatch.setenv("SANDBOX_MISUSE_STUBS", "1")
    importlib.reload(env)
    stubs = env.generate_input_stubs(1)
    flat_vals = [str(v) for s in stubs for v in s.values()]
    assert any(
        "' OR '1'='1" in v or "<script>" in v or "../.." in v for v in flat_vals
    )

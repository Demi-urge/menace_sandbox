import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import importlib
import sandbox_runner.input_history_db as ih
import sandbox_runner.environment as env


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
    import sandbox_runner.generative_stub_provider as gsp
    def fail(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gsp, "generate_stubs", fail)
    monkeypatch.setattr(env, "_load_history", lambda p: [])
    stubs = env.generate_input_stubs(1)
    assert stubs == [expected]

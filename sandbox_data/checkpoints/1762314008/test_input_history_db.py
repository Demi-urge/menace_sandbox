import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sandbox_runner.input_history_db as ih


def test_input_history_roundtrip(tmp_path):
    db = ih.InputHistoryDB(tmp_path / "hist.db")
    db.add({"a": 1})
    db.add({"b": 2})
    samples = db.sample(2)
    assert any("a" in s for s in samples)


def test_input_history_recent(tmp_path):
    db = ih.InputHistoryDB(tmp_path / "hist.db")
    for i in range(5):
        db.add({"n": i})
    recents = db.recent(2)
    assert [r.get("n") for r in recents] == [4, 3]

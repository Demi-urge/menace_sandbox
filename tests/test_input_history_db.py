import sandbox_runner.input_history_db as ih


def test_input_history_roundtrip(tmp_path):
    db = ih.InputHistoryDB(tmp_path / "hist.db")
    db.add({"a": 1})
    db.add({"b": 2})
    samples = db.sample(2)
    assert any("a" in s for s in samples)

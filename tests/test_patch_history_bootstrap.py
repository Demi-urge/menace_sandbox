import code_database


def _stub_vector_db(counter):
    class _Stub:
        def __init__(self):
            counter.append("init")

    return _Stub


def test_patch_history_skips_vector_metrics_in_bootstrap(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(code_database, "VectorMetricsDB", _stub_vector_db(calls))

    db = code_database.PatchHistoryDB(tmp_path / "bootstrap.db", bootstrap=True)

    db.record_vector_metrics(
        "",
        [],
        patch_id=1,
        contribution=0.0,
        win=False,
        regret=False,
    )

    assert calls == []
    assert getattr(db, "_vector_metrics_db")() is None


def test_patch_history_initializes_vector_metrics_by_default(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(code_database, "VectorMetricsDB", _stub_vector_db(calls))

    code_database.PatchHistoryDB(tmp_path / "default.db")

    assert calls == ["init"]

import menace.code_database as cd


def test_patch_history_roundtrip(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    rec1 = cd.PatchRecord(
        "a.py",
        "desc",
        1.0,
        2.0,
        5,
        2,
        1.0,
        predicted_roi=2.5,
        predicted_errors=1.5,
        reverted=False,
        trending_topic="trend",
        source_bot="tester",
        version="1.0",
    )
    rec2 = cd.PatchRecord(
        "b.py",
        "bad",
        2.0,
        1.0,
        1,
        3,
        -1.0,
        predicted_roi=0.8,
        predicted_errors=3.2,
        reverted=True,
        source_bot="tester",
        version="1.0",
    )
    db.add(rec1)
    db.add(rec2)
    top = db.top_patches()
    assert top and top[0].filename == "a.py"
    rate = db.success_rate()
    assert rate == 0.5


def test_keyword_features(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    db.add(cd.PatchRecord("a.py", "topic ai development", 1.0, 2.0, trending_topic="AI"))
    count, recency = db.keyword_features()
    assert count > 0
    assert isinstance(recency, int)

import menace.code_database as cd


def test_patch_history_roundtrip(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    rec1 = cd.PatchRecord(
        "a.py",  # path-ignore
        "desc",
        1.0,
        2.0,
        5,
        2,
        0,
        0,
        1.0,
        predicted_roi=2.5,
        predicted_errors=1.5,
        reverted=False,
        trending_topic="trend",
        source_bot="tester",
        version="1.0",
    )
    rec2 = cd.PatchRecord(
        "b.py",  # path-ignore
        "bad",
        2.0,
        1.0,
        1,
        3,
        0,
        0,
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
    assert top and top[0].filename == "a.py"  # path-ignore
    rate = db.success_rate()
    assert rate == 0.5


def test_branch_retrieval(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    parent_id = db.add(cd.PatchRecord("a.py", "p", 1.0, 2.0))  # path-ignore
    db.add(
        cd.PatchRecord(
            "b.py",  # path-ignore
            "c",
            2.0,
            3.0,
            parent_patch_id=parent_id,
            reason="improve",
            trigger="auto",
        )
    )
    children = db.filter(parent_patch_id=parent_id)
    assert len(children) == 1
    assert children[0].reason == "improve"


def test_keyword_features(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    db.add(cd.PatchRecord("a.py", "topic ai development", 1.0, 2.0, trending_topic="AI"))  # path-ignore
    count, recency = db.keyword_features()
    assert count > 0
    assert isinstance(recency, int)


def test_get_patch_record(tmp_path):
    db = cd.PatchHistoryDB(tmp_path / "p.db")
    pid = db.add(cd.PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
    rec = db.get(pid)
    assert rec is not None
    assert rec.filename == "a.py"  # path-ignore
    assert db.get(9999) is None

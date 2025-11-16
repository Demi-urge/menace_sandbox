import menace.neuroplasticity as neu


def test_log_and_query(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    rec1 = neu.PathwayRecord(
        actions="A->B",
        inputs="i",
        outputs="o",
        exec_time=1.0,
        resources="r",
        outcome=neu.Outcome.SUCCESS,
        roi=2.0,
    )
    pid = db.log(rec1)
    rec2 = neu.PathwayRecord(
        actions="A->B",
        inputs="i2",
        outputs="o2",
        exec_time=2.0,
        resources="r",
        outcome=neu.Outcome.FAILURE,
        roi=1.0,
    )
    db.log(rec2)
    top = db.top_pathways(limit=1)
    assert top[0][0] == pid
    cur = db.conn.execute("SELECT frequency, success_rate FROM metadata WHERE pathway_id=?", (pid,))
    freq, suc = cur.fetchone()
    assert freq == 1
    assert 0 <= suc <= 1


def test_reinforce(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    p1 = db.log(neu.PathwayRecord(actions="X", inputs="", outputs="", exec_time=0.5, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    p2 = db.log(neu.PathwayRecord(actions="Y", inputs="", outputs="", exec_time=0.5, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    db.reinforce_link(p1, p2)
    db.reinforce_link(p1, p2, 0.5)
    row = db.conn.execute("SELECT weight FROM links WHERE from_id=? AND to_id=?", (p1, p2)).fetchone()
    assert row[0] == 1.5


def test_sequence_and_decay(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db", half_life_days=1)
    rec = neu.PathwayRecord(
        actions="A",
        inputs="",
        outputs="",
        exec_time=1.0,
        resources="",
        outcome=neu.Outcome.SUCCESS,
        roi=1.0,
        ts="2020-01-01T00:00:00",
    )
    pid = db.log(rec)
    later = neu.PathwayRecord(
        actions="A",
        inputs="",
        outputs="",
        exec_time=1.0,
        resources="",
        outcome=neu.Outcome.SUCCESS,
        roi=1.0,
        ts="2020-01-03T00:00:00",
    )
    db._update_meta(pid, later)
    row = db.conn.execute(
        "SELECT myelination_score FROM metadata WHERE pathway_id=?", (pid,)
    ).fetchone()
    assert row[0] < 1.0

    p2 = db.log(neu.PathwayRecord(actions="B", inputs="", outputs="", exec_time=0.1, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    db.record_sequence([pid, p2])
    assert db.next_pathway(pid) == p2


def test_similar_and_merge(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    p1 = db.log(neu.PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=0.5, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    p2 = db.log(neu.PathwayRecord(actions="B->C", inputs="", outputs="", exec_time=0.5, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    sim = db.similar_actions("A", limit=1)
    assert sim and sim[0][0] == p1
    db.reinforce_link(p1, p2, weight=5)
    db.merge_macro_pathways(weight_threshold=4)
    cur = db.conn.execute("SELECT COUNT(*) FROM pathways").fetchone()
    assert cur[0] == 3


def test_ngram_tracking(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    ids = [
        db.log(
            neu.PathwayRecord(actions=str(i), inputs="", outputs="", exec_time=1.0, resources="", outcome=neu.Outcome.SUCCESS, roi=1)
        )
        for i in range(3)
    ]
    db.record_sequence(ids)
    seq = "-".join(str(i) for i in ids)
    row = db.conn.execute("SELECT weight FROM ngrams WHERE n=3 AND seq=?", (seq,)).fetchone()
    assert row and row[0] >= 1.0

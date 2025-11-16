import logging

import menace.chatgpt_enhancement_bot as ceb  # noqa: E402


def test_add_and_link(tmp_path):
    db = ceb.EnhancementDB(tmp_path / "e.db")
    db._embed = lambda text: [1.0]

    # insert a local enhancement
    db.conn.execute(
        "INSERT INTO enhancements(idea, rationale, summary, source_menace_id) VALUES (?,?,?,?)",
        ("i", "r", "s", db.router.menace_id),
    )
    db.conn.commit()

    # insert cross-instance enhancement and links
    db.conn.execute(
        "INSERT INTO enhancements(idea, rationale, source_menace_id) VALUES (?,?,?)",
        ("j", "r2", "other"),
    )
    db.conn.commit()
    cross_id = db.conn.execute(
        "SELECT id FROM enhancements WHERE source_menace_id=?", ("other",)
    ).fetchone()[0]
    db.link_model(cross_id, 4)
    db.link_bot(cross_id, 5)
    db.link_workflow(cross_id, 6)

    items_local = db.fetch(scope="local")
    assert {e.idea for e in items_local} == {"i"}
    items_global = db.fetch(scope="global")
    assert {e.idea for e in items_global} == {"j"}
    items_all = db.fetch(scope="all")
    assert {e.idea for e in items_all} == {"i", "j"}

    # ensure scoped lookups respect menace boundaries
    assert db.models_for(cross_id, scope="local") == []
    assert db.models_for(cross_id, scope="global") == [4]
    assert db.bots_for(cross_id, scope="local") == []
    assert db.bots_for(cross_id, scope="global") == [5]
    assert db.workflows_for(cross_id, scope="local") == []
    assert db.workflows_for(cross_id, scope="global") == [6]


def test_duplicate_insert(tmp_path, caplog):
    db = ceb.EnhancementDB(tmp_path / "e.db")
    # prevent vector backend interactions
    db.add_embedding = lambda *a, **k: None

    enh = ceb.Enhancement(
        idea="same",
        rationale="why",
        before_code="a",
        after_code="b",
        description="desc",
    )

    with caplog.at_level(logging.WARNING):
        first = db.add(enh)
        second = db.add(enh)

    assert first == second
    with db._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0]
    assert count == 1
    assert any("duplicate enhancement" in r.message for r in caplog.records)

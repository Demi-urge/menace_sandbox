import menace.error_bot as eb
from menace.db_router import DBRouter


def test_bot_error_types_scope_and_filters(tmp_path):
    shared = tmp_path / "shared.db"
    router_a = DBRouter("one", str(tmp_path / "a.db"), str(shared))
    db = eb.ErrorDB(tmp_path / "err.db", router=router_a)
    db.conn.execute(
        "INSERT INTO telemetry(bot_id, error_type, source_menace_id) VALUES (?,?,?)",
        ("bot1", "L", "one"),
    )
    db.conn.commit()

    router_b = DBRouter("two", str(tmp_path / "b.db"), str(shared))
    conn_b = router_b.get_connection("telemetry")
    conn_b.execute(
        "INSERT INTO telemetry(bot_id, error_type, source_menace_id) VALUES (?,?,?)",
        ("bot1", "G", "two"),
    )
    conn_b.execute(
        "INSERT INTO telemetry(bot_id, error_type, source_menace_id) VALUES (?,?,?)",
        ("bot2", "X", "two"),
    )
    conn_b.commit()

    assert db.get_bot_error_types("bot1", scope="local") == ["L"]
    assert db.get_bot_error_types("bot1", scope="global") == ["G"]
    assert set(db.get_bot_error_types("bot1", scope="all")) == {"L", "G"}

    # bot2 has only global entries
    assert db.get_bot_error_types("bot2", scope="local") == []
    assert db.get_bot_error_types("bot2", scope="global") == ["X"]

    # missing bot_id returns no results
    assert db.get_bot_error_types("missing", scope="all") == []


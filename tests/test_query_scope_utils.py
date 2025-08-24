from query_scope_utils import build_scope_clause, append_scope_clause


def test_build_scope_clause():
    assert build_scope_clause("t", "local", "1") == (
        "WHERE t.source_menace_id=?",
        ("1",),
    )
    assert build_scope_clause("t", "global", "1") == (
        "WHERE t.source_menace_id!=?",
        ("1",),
    )
    assert build_scope_clause("t", "all", "1") == ("", ())


def test_append_scope_clause():
    clause = build_scope_clause("bots", "local", "1")
    sql, params = append_scope_clause("SELECT * FROM bots", clause)
    assert sql == "SELECT * FROM bots WHERE bots.source_menace_id=?"
    assert params == ("1",)

    clause = build_scope_clause("bots", "global", "2")
    sql, params = append_scope_clause(
        "SELECT * FROM bots WHERE bots.type=?",
        clause,
        ("test",),
    )
    assert sql == "SELECT * FROM bots WHERE bots.type=? AND bots.source_menace_id!=?"
    assert params == ("test", "2")

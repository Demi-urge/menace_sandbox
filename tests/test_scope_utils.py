from scope_utils import build_scope_clause, apply_scope


def test_build_scope_clause():
    assert build_scope_clause("t", "local", "1") == (
        "WHERE t.source_menace_id=?",
        ["1"],
    )
    assert build_scope_clause("t", "global", "1") == (
        "WHERE t.source_menace_id!=?",
        ["1"],
    )
    assert build_scope_clause("t", "all", "1") == ("", [])


def test_apply_scope():
    sql, params = apply_scope("SELECT * FROM bots", "local", "1")
    assert sql == "SELECT * FROM bots WHERE bots.source_menace_id=?"
    assert params == ["1"]

    sql, params = apply_scope(
        "SELECT * FROM bots WHERE bots.type=?",
        "global",
        "2",
        params=["test"],
    )
    assert sql == "SELECT * FROM bots WHERE bots.type=? AND bots.source_menace_id!=?"
    assert params == ["test", "2"]

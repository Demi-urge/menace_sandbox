from scope_utils import Scope, build_scope_clause, apply_scope_to_query


def test_build_scope_clause():
    assert build_scope_clause("t", Scope.LOCAL, "1") == (
        "t.source_menace_id = ?",
        ["1"],
    )
    assert build_scope_clause("t", Scope.GLOBAL, "1") == (
        "t.source_menace_id <> ?",
        ["1"],
    )
    assert build_scope_clause("t", Scope.ALL, "1") == ("", [])


def test_apply_scope_to_query():
    sql, params = apply_scope_to_query("SELECT * FROM bots", Scope.LOCAL, "1")
    assert sql == "SELECT * FROM bots WHERE bots.source_menace_id = ?"
    assert params == ["1"]

    sql, params = apply_scope_to_query(
        "SELECT * FROM bots WHERE bots.type=?",
        Scope.GLOBAL,
        "2",
        params=["test"],
    )
    assert sql == "SELECT * FROM bots WHERE bots.type=? AND bots.source_menace_id <> ?"
    assert params == ["test", "2"]

from db_scope import Scope, build_scope_clause


def test_build_scope_clause():
    assert build_scope_clause("t", Scope.LOCAL, "1") == (
        "WHERE t.source_menace_id = ?",
        ["1"],
    )
    assert build_scope_clause("t", Scope.GLOBAL, "1") == (
        "WHERE t.source_menace_id != ?",
        ["1"],
    )
    assert build_scope_clause("t", Scope.ALL, "1") == ("", [])

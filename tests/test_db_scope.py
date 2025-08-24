import pytest
from scope_utils import Scope, build_scope_clause, apply_scope


@pytest.mark.parametrize(
    "scope, expected_clause",
    [
        (Scope.LOCAL, "bots.source_menace_id = ?"),
        (Scope.GLOBAL, "bots.source_menace_id <> ?"),
        (Scope.ALL, ""),
    ],
)
def test_build_scope_clause(scope, expected_clause):
    clause, params = build_scope_clause("bots", scope, "alpha")
    assert clause == expected_clause
    if scope is Scope.ALL:
        assert params == []
    else:
        assert params == ["alpha"]


def test_apply_scope_adds_prefixes():
    clause, _ = build_scope_clause("bots", Scope.LOCAL, "alpha")
    assert (
        apply_scope("SELECT * FROM bots", clause)
        == "SELECT * FROM bots WHERE bots.source_menace_id = ?"
    )

    clause, _ = build_scope_clause("bots", Scope.GLOBAL, "alpha")
    assert (
        apply_scope("SELECT * FROM bots WHERE active = 1", clause)
        == "SELECT * FROM bots WHERE active = 1 AND bots.source_menace_id <> ?"
    )

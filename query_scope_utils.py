"""Helpers for constructing menace scope-aware SQL clauses."""

from __future__ import annotations

from typing import Literal, Any

ScopeLiteral = Literal["local", "global", "all"]


def build_scope_clause(table_name: str, scope: ScopeLiteral, menace_id: str) -> tuple[str, tuple[Any, ...]]:
    """Return a ``WHERE`` fragment and parameters for the given menace ``scope``.

    Parameters
    ----------
    table_name:
        Name or alias of the table being queried.
    scope:
        One of ``"local"``, ``"global"``, or ``"all"`` determining how records
        are filtered relative to ``menace_id``.
    menace_id:
        Identifier for the current menace instance.
    """
    if scope == "local":
        return f"WHERE {table_name}.source_menace_id=?", (menace_id,)
    if scope == "global":
        return f"WHERE {table_name}.source_menace_id!=?", (menace_id,)
    if scope == "all":
        return "", ()
    raise ValueError(f"unknown scope: {scope}")


def append_scope_clause(sql: str, clause: tuple[str, tuple[Any, ...]], params: tuple[Any, ...] = ()) -> tuple[str, tuple[Any, ...]]:
    """Append a scope clause to an existing SQL statement.

    ``sql`` may already contain a ``WHERE`` clause. Parameters for the returned
    query consist of any existing ``params`` combined with those from
    ``clause``.
    """
    fragment, clause_params = clause
    if not fragment:
        return sql, params

    upper_sql = sql.upper()
    if "WHERE" in upper_sql:
        sql = f"{sql} AND {fragment.split('WHERE ', 1)[1]}"
    else:
        sql = f"{sql} {fragment}"
    return sql, params + clause_params


__all__ = ["build_scope_clause", "append_scope_clause"]

"""Helpers for constructing menace scope-aware SQL clauses."""

from __future__ import annotations

import re
from typing import Any, Literal, Sequence

ScopeLiteral = Literal["local", "global", "all"]


def build_scope_clause(
    table_name: str, scope: ScopeLiteral, menace_id: str
) -> tuple[str, list[Any]]:
    """Return a ``WHERE`` fragment and parameters enforcing ``scope``.

    Parameters
    ----------
    table_name:
        Name or alias of the table being queried.
    scope:
        Visibility selector: ``"local"``, ``"global"`` or ``"all"``.
    menace_id:
        Identifier for the current Menace instance.
    """

    if scope == "local":
        return f"WHERE {table_name}.source_menace_id=?", [menace_id]
    if scope == "global":
        return f"WHERE {table_name}.source_menace_id!=?", [menace_id]
    if scope == "all":
        return "", []
    raise ValueError(f"unknown scope: {scope}")


def apply_scope(
    query: str,
    scope: ScopeLiteral,
    menace_id: str,
    table_alias: str | None = None,
    params: Sequence[Any] | None = None,
) -> tuple[str, list[Any]]:
    """Append menace scope filtering to ``query``.

    ``table_alias`` specifies the table or alias whose ``source_menace_id``
    column should be filtered. If omitted, the first table name following the
    ``FROM`` keyword is used.
    """

    alias = table_alias
    if alias is None:
        match = re.search(r"FROM\s+([^\s,]+)(?:\s+AS\s+(\w+)|\s+(\w+))?", query, re.IGNORECASE)
        if not match:
            raise ValueError("table alias could not be inferred; specify table_alias")
        alias = match.group(1)
        if match.group(2):
            alias = match.group(2)
        elif match.group(3) and match.group(3).upper() not in {"WHERE", "JOIN", "ON"}:
            alias = match.group(3)

    clause, scope_params = build_scope_clause(alias, scope, menace_id)
    sql = query
    params_list = list(params) if params is not None else []
    if clause:
        if "WHERE" in sql.upper():
            sql = f"{sql} AND {clause.split('WHERE ', 1)[1]}"
        else:
            sql = f"{sql} {clause}"
        params_list.extend(scope_params)
    return sql, params_list


__all__ = ["build_scope_clause", "apply_scope"]

from __future__ import annotations

"""Helpers for constructing menace scope-aware SQL clauses.

The :class:`Scope` enum defines three visibility levels for cross-instance
queries:

- ``"local"`` – records created by the current menace
- ``"global"`` – records from other menace instances
- ``"all"`` – no menace ID filtering

Use :func:`build_scope_clause` to generate a filter fragment enforcing the
selected scope. Examples::

    >>> build_scope_clause("bots", Scope.LOCAL, "alpha")
    ('bots.source_menace_id = ?', ['alpha'])
    >>> build_scope_clause("bots", Scope.GLOBAL, "alpha")
    ('bots.source_menace_id <> ?', ['alpha'])
    >>> build_scope_clause("bots", Scope.ALL, "alpha")
    ('', [])

Combine with :func:`apply_scope` to prefix clauses as needed::

    >>> clause, params = build_scope_clause("bots", Scope.LOCAL, "alpha")
    >>> apply_scope("SELECT * FROM bots", clause)
    'SELECT * FROM bots WHERE bots.source_menace_id = ?'

This replaces the deprecated ``include_cross_instance`` and ``all_instances``
flags with a single ``scope`` parameter.
"""

from enum import Enum
import re
from typing import Any, List, Tuple, Sequence


class Scope(str, Enum):
    """Database scope selector for queries."""

    LOCAL = "local"
    GLOBAL = "global"
    ALL = "all"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def build_scope_clause(
    table_name: str, scope: Scope | str, menace_id: Any
) -> Tuple[str, List[Any]]:
    """Return a SQL clause and parameter list enforcing ``scope``.

    The returned clause does **not** include a ``WHERE``/``AND`` prefix, making it
    easy to splice into existing queries. ``scope`` can be passed either as a
    :class:`Scope` value or one of the strings ``"local"``, ``"global"`` or
    ``"all"``.
    """

    scope = Scope(scope)
    if scope is Scope.LOCAL:
        return f"{table_name}.source_menace_id = ?", [menace_id]
    if scope is Scope.GLOBAL:
        return f"{table_name}.source_menace_id <> ?", [menace_id]
    return "", []


def apply_scope(query: str, clause: str) -> str:
    """Prepend ``clause`` to ``query`` with ``WHERE`` or ``AND`` as needed."""

    if not clause:
        return query
    if "where" in query.lower():
        return f"{query} AND {clause}"
    return f"{query} WHERE {clause}"


def apply_scope_to_query(
    query: str,
    scope: Scope | str,
    menace_id: Any,
    table_alias: str | None = None,
    params: Sequence[Any] | None = None,
) -> Tuple[str, List[Any]]:
    """Append menace scope filtering to ``query``.

    ``table_alias`` specifies the table or alias whose ``source_menace_id`` column
    should be filtered. If omitted, the first table name following ``FROM`` is
    used. ``params`` contains any existing parameter values that should be
    extended with the scope parameters.
    """

    alias = table_alias
    if alias is None:
        match = re.search(
            r"FROM\s+([^\s,]+)(?:\s+AS\s+(\w+)|\s+(\w+))?",
            query,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("table alias could not be inferred; specify table_alias")
        alias = match.group(1)
        if match.group(2):
            alias = match.group(2)
        elif match.group(3) and match.group(3).upper() not in {"WHERE", "JOIN", "ON"}:
            alias = match.group(3)

    clause, scope_params = build_scope_clause(alias, scope, menace_id)
    sql = apply_scope(query, clause)
    params_list = list(params) if params is not None else []
    params_list.extend(scope_params)
    return sql, params_list


__all__ = ["Scope", "build_scope_clause", "apply_scope", "apply_scope_to_query"]

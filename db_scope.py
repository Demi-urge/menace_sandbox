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

This replaces the deprecated ``include_cross_instance`` and ``all_instances``
flags with a single ``scope`` parameter.
"""

from enum import Enum
from typing import Any, List, Tuple


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


__all__ = ["Scope", "build_scope_clause", "apply_scope"]

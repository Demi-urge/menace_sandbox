from __future__ import annotations

"""Helpers for constructing menace scope-aware SQL clauses.

The :class:`Scope` enum defines three visibility levels for cross-instance
queries:

- ``"local"`` – records created by the current menace
- ``"global"`` – records from other menace instances
- ``"all"`` – no menace ID filtering

Use :func:`build_scope_clause` to generate a ``WHERE`` fragment enforcing the
selected scope. Examples::

    >>> build_scope_clause("bots", Scope.LOCAL, "alpha")
    ('WHERE bots.source_menace_id = ?', ['alpha'])
    >>> build_scope_clause("bots", Scope.GLOBAL, "alpha")
    ('WHERE bots.source_menace_id != ?', ['alpha'])
    >>> build_scope_clause("bots", Scope.ALL, "alpha")
    ('', [])

This replaces the deprecated ``include_cross_instance`` and ``all_instances``
flags with a single ``scope`` parameter.
"""

from enum import Enum
from typing import List, Tuple


class Scope(str, Enum):
    """Database scope selector for queries."""

    LOCAL = "local"
    GLOBAL = "global"
    ALL = "all"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def build_scope_clause(
    table_alias: str, scope: Scope | str, menace_id: str
) -> Tuple[str, List[str]]:
    """Return SQL fragment and parameters enforcing ``scope`` for ``menace_id``.

    ``scope`` controls menace visibility:

    - ``Scope.LOCAL`` – only rows from ``menace_id``
    - ``Scope.GLOBAL`` – rows from other Menace instances
    - ``Scope.ALL`` – no menace ID filtering
    """

    scope = Scope(scope)
    if scope is Scope.LOCAL:
        return f"WHERE {table_alias}.source_menace_id = ?", [menace_id]
    if scope is Scope.GLOBAL:
        return f"WHERE {table_alias}.source_menace_id != ?", [menace_id]
    return "", []


__all__ = ["Scope", "build_scope_clause"]

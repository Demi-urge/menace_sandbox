from __future__ import annotations

"""Helpers for constructing menace scope-aware SQL clauses."""

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
    """Return SQL fragment and parameters enforcing ``scope`` for ``menace_id``."""

    scope = Scope(scope)
    if scope is Scope.LOCAL:
        return f"WHERE {table_alias}.source_menace_id = ?", [menace_id]
    if scope is Scope.GLOBAL:
        return f"WHERE {table_alias}.source_menace_id != ?", [menace_id]
    return "", []


__all__ = ["Scope", "build_scope_clause"]


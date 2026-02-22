from __future__ import annotations

"""Listener that updates a Neo4j graph with bot events."""

from typing import Optional
from .unified_event_bus import UnifiedEventBus

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional
    GraphDatabase = None  # type: ignore


class Neo4jGraphListener:
    """Push bot events into a Neo4j database."""

    def __init__(
        self,
        event_bus: UnifiedEventBus,
        uri: str,
        user: str,
        password: str,
    ) -> None:
        if GraphDatabase is None:
            raise ImportError("neo4j is required for Neo4jGraphListener")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        event_bus.subscribe("bot:new", self._on_new)
        event_bus.subscribe("bot:interaction", self._on_interaction)

    # ------------------------------------------------------------------
    def _on_new(self, _: str, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        name = payload.get("name")
        if not name:
            return
        with self._driver.session() as session:
            session.run(
                "MERGE (b:Bot {name: $name})",
                name=str(name),
            )

    def _on_interaction(self, _: str, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        f = payload.get("from")
        t = payload.get("to")
        if not f or not t:
            return
        weight = float(payload.get("weight", 1.0))
        with self._driver.session() as session:
            session.run(
                """
                MERGE (a:Bot {name: $from})
                MERGE (b:Bot {name: $to})
                MERGE (a)-[r:INTERACTS_WITH]->(b)
                ON CREATE SET r.weight = $w
                ON MATCH SET r.weight = r.weight + $w
                """,
                {"from": str(f), "to": str(t), "w": weight},
            )


__all__ = ["Neo4jGraphListener"]

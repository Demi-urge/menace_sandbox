from __future__ import annotations

"""Graph based registry capturing bot interactions."""

from typing import Iterable, List, Tuple, Union, Optional, Dict
from pathlib import Path
import sqlite3
import time

try:
    from .databases import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore
try:
    from .neuroplasticity import PathwayDB
except Exception:  # pragma: no cover - optional dependency
    PathwayDB = None  # type: ignore

import networkx as nx
import logging

logger = logging.getLogger(__name__)

from .unified_event_bus import UnifiedEventBus


class BotRegistry:
    """Store connections between bots using a directed graph."""

    def __init__(self, *, persist: Optional[Path | str] = None, event_bus: Optional["UnifiedEventBus"] = None) -> None:
        self.graph = nx.DiGraph()
        self.persist_path = Path(persist) if persist else None
        self.event_bus = event_bus
        self.heartbeats: Dict[str, float] = {}
        self.interactions_meta: List[Dict[str, object]] = []
        if self.persist_path and self.persist_path.exists():
            try:
                self.load(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to load bot registry from %s: %s", self.persist_path, exc
                )

    def register_bot(self, name: str) -> None:
        """Ensure *name* exists in the graph."""
        self.graph.add_node(name)
        if self.event_bus:
            try:
                self.event_bus.publish("bot:new", {"name": name})
            except Exception as exc:
                logger.error("Failed to publish bot:new event: %s", exc)
        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to save bot registry to %s: %s", self.persist_path, exc
                )

    def register_interaction(self, from_bot: str, to_bot: str, weight: float = 1.0) -> None:
        """Record that *from_bot* interacted with *to_bot*."""
        self.register_bot(from_bot)
        self.register_bot(to_bot)
        if self.graph.has_edge(from_bot, to_bot):
            self.graph[from_bot][to_bot]["weight"] += weight
        else:
            self.graph.add_edge(from_bot, to_bot, weight=weight)
        self.interactions_meta.append({"from": from_bot, "to": to_bot, "weight": weight, "ts": time.time()})
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:interaction", {"from": from_bot, "to": to_bot, "weight": weight}
                )
            except Exception as exc:
                logger.error("Failed to publish bot:interaction event: %s", exc)
        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to save bot registry to %s: %s", self.persist_path, exc
                )

    def connections(self, bot: str, depth: int = 1) -> List[Tuple[str, float]]:
        """Return outgoing connections up to *depth* hops."""
        results: List[Tuple[str, float]] = []
        if bot not in self.graph:
            return results
        for nbr in self.graph.successors(bot):
            w = float(self.graph[bot][nbr].get("weight", 1.0))
            results.append((nbr, w))
            if depth > 1:
                results.extend(self.connections(nbr, depth - 1))
        return results

    # ------------------------------------------------------------------
    def record_heartbeat(self, name: str) -> None:
        """Update last seen timestamp for *name*."""
        self.heartbeats[name] = time.time()
        if self.event_bus:
            try:
                self.event_bus.publish("bot:heartbeat", {"name": name})
            except Exception as exc:
                logger.error("Failed to publish bot:heartbeat event: %s", exc)
                try:
                    self.event_bus.publish(
                        "bot:heartbeat_error", {"name": name, "error": str(exc)}
                    )
                except Exception:
                    logger.exception("Failed publishing heartbeat error")

    def active_bots(self, timeout: float = 60.0) -> Dict[str, float]:
        """Return bots seen within ``timeout`` seconds."""
        now = time.time()
        return {n: ts for n, ts in self.heartbeats.items() if now - ts <= timeout}

    def record_interaction_metadata(
        self,
        from_bot: str,
        to_bot: str,
        *,
        duration: float,
        success: bool,
        resources: str = "",
    ) -> None:
        """Store detailed metadata for an interaction."""
        self.interactions_meta.append(
            {
                "from": from_bot,
                "to": to_bot,
                "duration": duration,
                "success": success,
                "resources": resources,
                "ts": time.time(),
            }
        )

    def aggregate_statistics(self) -> Dict[str, float]:
        """Return simple aggregate metrics about interactions."""
        if not self.interactions_meta:
            return {"count": 0, "success_rate": 0.0, "avg_duration": 0.0}
        count = len(self.interactions_meta)
        successes = sum(1 for rec in self.interactions_meta if rec.get("success"))
        total_dur = sum(float(rec.get("duration", 0.0)) for rec in self.interactions_meta)
        return {
            "count": count,
            "success_rate": successes / count,
            "avg_duration": total_dur / count,
        }

    # ------------------------------------------------------------------
    def save(self, dest: Union[Path, str, "MenaceDB", "PathwayDB"]) -> None:
        """Persist the current graph to a SQLite-backed database."""
        if isinstance(dest, str) or isinstance(dest, Path):
            path = Path(dest)
            conn = sqlite3.connect(path)
            close_conn = True
        elif MenaceDB is not None and isinstance(dest, MenaceDB):
            conn = dest.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(dest, PathwayDB):
            conn = dest.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported destination for save")

        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS bot_nodes(name TEXT PRIMARY KEY)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_edges(
                from_bot TEXT,
                to_bot TEXT,
                weight REAL,
                PRIMARY KEY(from_bot, to_bot)
            )
            """
        )
        for node in self.graph.nodes:
            cur.execute("INSERT OR IGNORE INTO bot_nodes(name) VALUES(?)", (node,))
        for u, v, data in self.graph.edges(data=True):
            cur.execute(
                "REPLACE INTO bot_edges(from_bot,to_bot,weight) VALUES(?,?,?)",
                (u, v, float(data.get("weight", 1.0))),
            )
        conn.commit()
        if close_conn:
            conn.close()

    # ------------------------------------------------------------------
    def load(self, src: Union[Path, str, "MenaceDB", "PathwayDB"]) -> None:
        """Populate ``self.graph`` from ``src`` tables."""
        if isinstance(src, (str, Path)):
            path = Path(src)
            conn = sqlite3.connect(path)
            close_conn = True
        elif MenaceDB is not None and isinstance(src, MenaceDB):
            conn = src.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(src, PathwayDB):  # pragma: no cover - rarely used
            conn = src.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported source for load")

        self.graph.clear()
        cur = conn.cursor()
        try:
            node_rows = cur.execute("SELECT name FROM bot_nodes").fetchall()
        except Exception:
            node_rows = []
        for (name,) in node_rows:
            self.graph.add_node(name)

        try:
            edge_rows = cur.execute(
                "SELECT from_bot, to_bot, weight FROM bot_edges"
            ).fetchall()
        except Exception:
            edge_rows = []
        for u, v, w in edge_rows:
            self.graph.add_edge(u, v, weight=float(w))

        if close_conn:
            conn.close()


__all__ = ["BotRegistry"]

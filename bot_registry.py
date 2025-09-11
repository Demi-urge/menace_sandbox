from __future__ import annotations

"""Graph based registry capturing bot interactions.

The registry persists bot connections to a database and allows bots to be
hot swapped at runtime. Updating a bot's backing module via ``update_bot``
broadcasts a ``bot:updated`` event so other components can react to the
change.
"""

from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
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

from .unified_event_bus import UnifiedEventBus
import db_router
from db_router import DBRouter, init_db_router

logger = logging.getLogger(__name__)


class BotRegistry:
    """Store connections between bots using a directed graph."""

    def __init__(
        self,
        *,
        persist: Optional[Path | str] = None,
        event_bus: Optional["UnifiedEventBus"] = None,
    ) -> None:
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

    def update_bot(
        self,
        name: str,
        module_path: str,
        *,
        patch_id: int | None = None,
        commit: str | None = None,
    ) -> None:
        """Update stored module path for ``name`` and emit ``bot:updated``.

        ``patch_id`` and ``commit`` are included in the emitted event so
        consumers can trace the exact change applied to the bot.
        """

        # Ensure the bot exists in the graph.
        self.register_bot(name)
        node = self.graph.nodes[name]
        node["module"] = module_path
        node["version"] = int(node.get("version", 0)) + 1
        if patch_id is not None:
            node["patch_id"] = patch_id
        if commit is not None:
            node["commit"] = commit

        if self.event_bus:
            try:
                payload = {
                    "name": name,
                    "module": module_path,
                    "version": node["version"],
                }
                if patch_id is not None:
                    payload["patch_id"] = patch_id
                if commit is not None:
                    payload["commit"] = commit
                self.event_bus.publish("bot:updated", payload)
            except Exception as exc:
                logger.error("Failed to publish bot:updated event: %s", exc)
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
        self.interactions_meta.append(
            {"from": from_bot, "to": to_bot, "weight": weight, "ts": time.time()}
        )
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

    def record_validation(self, bot: str, module: str, passed: bool) -> None:
        """Record patch validation outcome for ``bot``."""
        self.interactions_meta.append(
            {"bot": bot, "module": module, "passed": bool(passed), "ts": time.time()}
        )
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:patch_validation",
                    {"bot": bot, "module": module, "passed": bool(passed)},
                )
            except Exception as exc:
                logger.error("Failed to publish bot:patch_validation event: %s", exc)

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
    def save(
        self, dest: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Persist the current graph to a SQLite-backed database."""
        if isinstance(dest, (str, Path)):
            path = Path(dest)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(dest, DBRouter):
            conn = dest.get_connection("bots")
            close_conn = False
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
            "CREATE TABLE IF NOT EXISTS bot_nodes(" "name TEXT PRIMARY KEY, "
            "module TEXT, "
            "version INTEGER)"
        )
        # Ensure columns exist for databases created before they were introduced.
        try:  # pragma: no cover - only executed on legacy schemas
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
            if "module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN module TEXT")
            if "version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN version INTEGER")
        except Exception:
            pass
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
            data = self.graph.nodes[node]
            module = data.get("module")
            version = data.get("version")
            cur.execute(
                "INSERT OR REPLACE INTO bot_nodes(name, module, version) VALUES(?, ?, ?)",
                (node, module, version),
            )
        for u, v, data in self.graph.edges(data=True):
            cur.execute(
                "REPLACE INTO bot_edges(from_bot,to_bot,weight) VALUES(?,?,?)",
                (u, v, float(data.get("weight", 1.0))),
            )
        conn.commit()
        if close_conn:
            conn.close()

    # ------------------------------------------------------------------
    def load(
        self, src: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Populate ``self.graph`` from ``src`` tables."""
        if isinstance(src, (str, Path)):
            path = Path(src)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(src, DBRouter):
            conn = src.get_connection("bots")
            close_conn = False
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
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
        except Exception:
            cols = []

        module_col = "module" in cols
        version_col = "version" in cols
        try:
            if module_col and version_col:
                node_rows = cur.execute(
                    "SELECT name, module, version FROM bot_nodes"
                ).fetchall()
            elif module_col:
                node_rows = cur.execute(
                    "SELECT name, module FROM bot_nodes"
                ).fetchall()
            elif version_col:
                node_rows = cur.execute(
                    "SELECT name, version FROM bot_nodes"
                ).fetchall()
            else:
                node_rows = cur.execute("SELECT name FROM bot_nodes").fetchall()
        except Exception:  # pragma: no cover - corrupted table
            node_rows = []

        for row in node_rows:
            name = row[0]
            module = None
            version = None
            if module_col and version_col and len(row) >= 3:
                _, module, version = row
            elif module_col:
                _, module = row
            elif version_col:
                _, version = row
            self.graph.add_node(name)
            if module is not None:
                self.graph.nodes[name]["module"] = module
            if version is not None:
                self.graph.nodes[name]["version"] = int(version)

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

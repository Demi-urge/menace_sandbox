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
import importlib
import importlib.util
import sys
import subprocess
import json

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

        ``patch_id`` and ``commit`` are expected from the ``SelfCodingManager``
        so changes can be traced back to their origin.  If either piece of
        metadata is missing this method attempts to retrieve it from
        :mod:`patch_provenance` and retries once.  If the metadata still cannot
        be determined a :class:`RuntimeError` is raised.
        """

        if patch_id is None or commit is None:
            logger.warning(
                "update_bot called without provenance for %s (patch_id=%s commit=%s)",
                name,
                patch_id,
                commit,
            )
            if patch_id is not None:
                try:
                    from .patch_provenance import PatchProvenanceService

                    service = PatchProvenanceService()
                    rec = service.db.get(patch_id)
                    if rec and getattr(rec, "summary", None):
                        try:
                            commit = json.loads(rec.summary).get("commit")
                        except Exception:  # pragma: no cover - best effort
                            commit = None
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to fetch patch provenance for %s: %s", patch_id, exc
                    )
            if patch_id is None or commit is None:
                raise RuntimeError("patch provenance required")

        # Ensure the bot exists in the graph.
        self.register_bot(name)
        node = self.graph.nodes[name]
        prev_state = dict(node)
        node["module"] = module_path
        node["version"] = int(node.get("version", 0)) + 1
        node["patch_id"] = patch_id
        node["commit"] = commit

        if self.event_bus:
            try:
                payload = {
                    "name": name,
                    "module": module_path,
                    "version": node["version"],
                    "patch_id": patch_id,
                    "commit": commit,
                }
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
        self.hot_swap_bot(name)
        self.health_check_bot(name, prev_state)

    def hot_swap_bot(self, name: str) -> None:
        """Import or reload the module backing ``name`` and refresh references."""

        node = self.graph.nodes.get(name)
        if not node or "module" not in node:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node["module"]
        commit = node.get("commit")
        patch_id = node.get("patch_id")
        prev_module = node.get("last_good_module")
        prev_version = node.get("last_good_version")
        prev_commit = node.get("last_good_commit")
        prev_patch = node.get("last_good_patch_id")
        if not commit or patch_id is None:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "missing_provenance"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "missing_provenance",
                        },
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:update_blocked event: %s", exc)
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: missing provenance metadata")

        stored_commit: str | None = None
        try:
            from .patch_provenance import PatchProvenanceService

            service = PatchProvenanceService()
            rec = service.db.get(patch_id)
            if rec and getattr(rec, "summary", None):
                try:
                    stored_commit = json.loads(rec.summary).get("commit")
                except Exception:  # pragma: no cover - best effort
                    stored_commit = None
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to fetch patch provenance for %s: %s", patch_id, exc)
        if stored_commit != commit:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "provenance_mismatch"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "provenance_mismatch",
                            "expected": stored_commit,
                            "actual": commit,
                        },
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to publish bot:update_blocked event: %s", exc
                    )
            manager = node.get("selfcoding_manager") or node.get("manager")
            if manager and hasattr(manager, "register_patch_cycle"):
                try:
                    manager.register_patch_cycle(
                        f"manual change detected for {name}",
                        {
                            "reason": "provenance_mismatch",
                            "module": module_path,
                        },
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to notify SelfCodingManager for %s: %s", name, exc
                    )
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: provenance mismatch")

        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain", module_path]
            ).decode()
            if status.strip() and self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "uncommitted_changes"},
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:manual_change event: %s", exc)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to check manual changes for %s: %s", module_path, exc)
        try:
            path_obj = Path(module_path)
            if path_obj.suffix == ".py" or "/" in module_path:
                module_name = path_obj.stem
                spec = importlib.util.spec_from_file_location(
                    module_name, module_path
                )
                if not spec or not spec.loader:
                    raise ImportError(f"cannot load module from {module_path}")
                importlib.invalidate_caches()
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    module.__file__ = module_path
                    module.__spec__ = spec
                    spec.loader.exec_module(module)
                else:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            else:
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)
            node["last_good_module"] = module_path
            node["last_good_version"] = node.get("version")
            node["last_good_commit"] = commit
            node["last_good_patch_id"] = patch_id
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(
                "Failed to hot swap bot %s from %s: %s", name, module_path, exc
            )
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise

    def health_check_bot(self, name: str, prev_state: Optional[Dict[str, object]] = None) -> None:
        """Import the bot module and record a heartbeat to verify health."""

        node = self.graph.nodes.get(name)
        if not node or "module" not in node:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node["module"]
        try:
            path_obj = Path(module_path)
            if path_obj.suffix == ".py" or "/" in module_path:
                module_name = path_obj.stem
            else:
                module_name = module_path
            importlib.invalidate_caches()
            importlib.import_module(module_name)
            self.record_heartbeat(name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Health check failed for bot %s: %s", name, exc)
            if prev_state is not None:
                node.clear()
                node.update(prev_state)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s",
                        self.persist_path,
                        save_exc,
                    )
            raise

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
            "version INTEGER, "
            "last_good_module TEXT, "
            "last_good_version INTEGER)"
        )
        # Ensure columns exist for databases created before they were introduced.
        try:  # pragma: no cover - only executed on legacy schemas
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
            if "module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN module TEXT")
            if "version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN version INTEGER")
            if "last_good_module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_module TEXT")
            if "last_good_version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_version INTEGER")
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
            last_mod = data.get("last_good_module")
            last_ver = data.get("last_good_version")
            cur.execute(
                """
                INSERT OR REPLACE INTO bot_nodes(
                    name, module, version, last_good_module, last_good_version
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (node, module, version, last_mod, last_ver),
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
        last_mod_col = "last_good_module" in cols
        last_ver_col = "last_good_version" in cols
        select_cols = [
            c for c in ["module", "version", "last_good_module", "last_good_version"] if c in cols
        ]
        col_sql = ", ".join(select_cols)
        try:
            if col_sql:
                node_rows = cur.execute(
                    f"SELECT name, {col_sql} FROM bot_nodes"
                ).fetchall()
            else:
                node_rows = cur.execute("SELECT name FROM bot_nodes").fetchall()
        except Exception:  # pragma: no cover - corrupted table
            node_rows = []

        for row in node_rows:
            name = row[0]
            self.graph.add_node(name)
            idx = 1
            if module_col:
                module = row[idx]
                idx += 1
                if module is not None:
                    self.graph.nodes[name]["module"] = module
            if version_col:
                version = row[idx]
                idx += 1
                if version is not None:
                    self.graph.nodes[name]["version"] = int(version)
            if last_mod_col:
                last_mod = row[idx]
                idx += 1
                if last_mod is not None:
                    self.graph.nodes[name]["last_good_module"] = last_mod
            if last_ver_col:
                last_ver = row[idx]
                idx += 1
                if last_ver is not None:
                    self.graph.nodes[name]["last_good_version"] = int(last_ver)

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

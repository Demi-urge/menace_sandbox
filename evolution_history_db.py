from __future__ import annotations

"""Store history of evolution cycles and their outcomes."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope


@dataclass
class EvolutionEvent:
    """Record of a single evolution cycle."""

    action: str
    before_metric: float
    after_metric: float
    roi: float
    predicted_roi: float = 0.0
    confidence: float = 0.0
    efficiency: float = 0.0
    bottleneck: float = 0.0
    patch_id: int | None = None
    workflow_id: int | None = None
    ts: str = datetime.utcnow().isoformat()
    trending_topic: str | None = None
    reason: str = ""
    trigger: str = ""
    performance: float = 0.0
    parent_event_id: int | None = None
    predicted_class: str = ""
    actual_class: str = ""


@dataclass
class WorkflowEvolutionRecord:
    """Audit entry for a workflow variant evaluation."""

    workflow_id: int
    variant: str
    baseline_roi: float
    variant_roi: float
    roi_delta: float
    baseline_synergy: float = 0.0
    variant_synergy: float = 0.0
    synergy_delta: float = 0.0
    mutation_id: int | None = None
    ts: str = datetime.utcnow().isoformat()


class EvolutionHistoryDB:
    """SQLite-backed store for evolution history."""

    def __init__(
        self,
        path: Path | str = "evolution_history.db",
        router: DBRouter | None = None,
    ) -> None:
        self.router = router or GLOBAL_ROUTER
        if self.router is None:
            self.router = init_db_router("evolution_history", str(path), str(path))
        self.conn = self.router.get_connection("evolution_history")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_history(
                source_menace_id TEXT NOT NULL,
                action TEXT,
                before_metric REAL,
                after_metric REAL,
                roi REAL,
                predicted_roi REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                efficiency REAL DEFAULT 0,
                bottleneck REAL DEFAULT 0,
                patch_id INTEGER,
                workflow_id INTEGER,
                ts TEXT,
                trending_topic TEXT,
                reason TEXT,
                "trigger" TEXT,
                performance REAL DEFAULT 0,
                parent_event_id INTEGER,
                predicted_class TEXT,
                actual_class TEXT
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(evolution_history)").fetchall()]
        if "source_menace_id" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN "
                "source_menace_id TEXT NOT NULL DEFAULT ''",
            )
            self.conn.execute(
                "UPDATE evolution_history SET source_menace_id='' "
                "WHERE source_menace_id IS NULL",
            )
        if "efficiency" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN efficiency REAL DEFAULT 0")
        if "bottleneck" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN bottleneck REAL DEFAULT 0")
        if "predicted_roi" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN predicted_roi REAL DEFAULT 0"
            )
        if "confidence" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN confidence REAL DEFAULT 0"
            )
        if "patch_id" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN patch_id INTEGER")
        if "workflow_id" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN workflow_id INTEGER")
        if "trending_topic" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN trending_topic TEXT")
        if "reason" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN reason TEXT")
        if "trigger" not in cols:
            self.conn.execute('ALTER TABLE evolution_history ADD COLUMN "trigger" TEXT')
        if "performance" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN performance REAL DEFAULT 0")
        if "parent_event_id" not in cols:
            if "parent_id" in cols:
                self.conn.execute(
                    "ALTER TABLE evolution_history RENAME COLUMN parent_id TO parent_event_id"
                )
            else:
                self.conn.execute(
                    "ALTER TABLE evolution_history ADD COLUMN parent_event_id INTEGER"
                )
        if "predicted_class" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN predicted_class TEXT"
            )
        if "actual_class" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN actual_class TEXT"
            )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_evolution(
                workflow_id INTEGER,
                variant TEXT,
                baseline_roi REAL,
                variant_roi REAL,
                roi_delta REAL,
                baseline_synergy REAL DEFAULT 0,
                variant_synergy REAL DEFAULT 0,
                synergy_delta REAL DEFAULT 0,
                mutation_id INTEGER,
                ts TEXT
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(workflow_evolution)").fetchall()]
        if "mutation_id" not in cols:
            self.conn.execute("ALTER TABLE workflow_evolution ADD COLUMN mutation_id INTEGER")
        if "ts" not in cols:
            self.conn.execute("ALTER TABLE workflow_evolution ADD COLUMN ts TEXT")
        if "baseline_synergy" not in cols:
            self.conn.execute(
                "ALTER TABLE workflow_evolution ADD COLUMN baseline_synergy REAL DEFAULT 0"
            )
        if "variant_synergy" not in cols:
            self.conn.execute(
                "ALTER TABLE workflow_evolution ADD COLUMN variant_synergy REAL DEFAULT 0"
            )
        if "synergy_delta" not in cols:
            self.conn.execute(
                "ALTER TABLE workflow_evolution ADD COLUMN synergy_delta REAL DEFAULT 0"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_evolution_history_source_menace_id ON evolution_history("
            "source_menace_id)",
        )
        self.conn.commit()

    def _current_menace_id(self, source_menace_id: str | None) -> str:
        return source_menace_id or (self.router.menace_id if self.router else "")

    def add(self, event: EvolutionEvent, *, source_menace_id: str | None = None) -> int:
        menace_id = self._current_menace_id(source_menace_id)
        cur = self.conn.execute(
            "INSERT INTO evolution_history("
            "source_menace_id, action, before_metric, after_metric, roi, "
            "predicted_roi, confidence, efficiency, bottleneck, patch_id, workflow_id, ts, "
            'trending_topic, reason, "trigger", performance, parent_event_id, '
            "predicted_class, actual_class) VALUES(" + ",".join("?" for _ in range(19)) + ")",
            (
                menace_id,
                event.action,
                event.before_metric,
                event.after_metric,
                event.roi,
                event.predicted_roi,
                event.confidence,
                event.efficiency,
                event.bottleneck,
                event.patch_id,
                event.workflow_id,
                event.ts,
                event.trending_topic,
                event.reason,
                event.trigger,
                event.performance,
                event.parent_event_id,
                event.predicted_class,
                event.actual_class,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    def log_workflow_evolution(
        self,
        *,
        workflow_id: int,
        variant: str,
        baseline_roi: float,
        variant_roi: float,
        roi_delta: float,
        mutation_id: int | None = None,
        baseline_synergy: float = 0.0,
        variant_synergy: float = 0.0,
        synergy_delta: float | None = None,
    ) -> int:
        """Persist details of a workflow variant evaluation."""
        if synergy_delta is None:
            synergy_delta = float(variant_synergy) - float(baseline_synergy)
        cur = self.conn.execute(
            "INSERT INTO workflow_evolution(" \
            "workflow_id, variant, baseline_roi, variant_roi, roi_delta, " \
            "baseline_synergy, variant_synergy, synergy_delta, mutation_id, ts) " \
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                int(workflow_id),
                variant,
                float(baseline_roi),
                float(variant_roi),
                float(roi_delta),
                float(baseline_synergy),
                float(variant_synergy),
                float(synergy_delta),
                mutation_id,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_cycle(self, rowid: int, efficiency: float, bottleneck: float) -> None:
        self.conn.execute(
            "UPDATE evolution_history SET efficiency=?, bottleneck=? WHERE rowid=?",
            (efficiency, bottleneck, rowid),
        )
        self.conn.commit()

    def fetch(
        self,
        limit: int = 50,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[
        Tuple[
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
            str,
            str,
        ]
    ]:
        base = (
            "SELECT action, before_metric, after_metric, roi, predicted_roi, "
            "efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, "
            'reason, "trigger", performance, parent_event_id, predicted_class, '
            "actual_class FROM evolution_history"
        )
        menace_id = self._current_menace_id(source_menace_id)
        clause, scope_params = build_scope_clause(
            "evolution_history", Scope(scope), menace_id
        )
        base = apply_scope(base, clause)
        base += " ORDER BY ts DESC LIMIT ?"
        params = [*scope_params, limit]
        cur = self.conn.execute(base, params)
        return cur.fetchall()

    def fetch_children(
        self,
        parent_event_id: int,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[
        Tuple[
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
            str,
            str,
        ]
    ]:
        menace_id = self._current_menace_id(source_menace_id)
        clause, scope_params = build_scope_clause(
            "evolution_history", Scope(scope), menace_id
        )
        query = (
            "SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, "
            "efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, "
            'reason, "trigger", performance, parent_event_id, predicted_class, actual_class '
            "FROM evolution_history WHERE parent_event_id=?"
        )
        query = apply_scope(query, clause)
        params = [parent_event_id, *scope_params]
        cur = self.conn.execute(query, params)
        return cur.fetchall()

    # ------------------------------------------------------------------
    def children(
        self,
        parent_event_id: int,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[
        Tuple[
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
            str,
            str,
        ]
    ]:
        """Alias for :meth:`fetch_children` for backwards compatibility."""

        return self.fetch_children(
            parent_event_id, scope=scope, source_menace_id=source_menace_id
        )

    # ------------------------------------------------------------------
    def ancestors(
        self,
        event_id: int,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[
        Tuple[
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
            str,
            str,
        ]
    ]:
        """Return lineage from root to ``event_id``."""

        menace_id = self._current_menace_id(source_menace_id)
        clause, scope_params = build_scope_clause(
            "evolution_history", Scope(scope), menace_id
        )
        query = (
            "SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, "
            "efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, "
            'reason, "trigger", performance, parent_event_id, predicted_class, actual_class '
            "FROM evolution_history WHERE rowid=?"
        )
        query = apply_scope(query, clause)

        rows = []
        row = self.conn.execute(query, (event_id, *scope_params)).fetchone()
        while row:
            rows.append(row)
            parent = row[-1]
            if parent is None:
                break
            row = self.conn.execute(query, (parent, *scope_params)).fetchone()
        return list(reversed(rows))

    # ------------------------------------------------------------------
    def subtree(
        self,
        event_id: int,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> dict | None:
        """Return the mutation subtree rooted at ``event_id``."""

        fields = [
            "rowid",
            "action",
            "before_metric",
            "after_metric",
            "roi",
            "predicted_roi",
            "efficiency",
            "bottleneck",
            "patch_id",
            "workflow_id",
            "ts",
            "trending_topic",
            "reason",
            "trigger",
            "performance",
            "parent_event_id",
            "predicted_class",
            "actual_class",
        ]

        def build(row: Tuple) -> dict:
            node = dict(zip(fields, row))
            kids = self.fetch_children(
                row[0], scope=scope, source_menace_id=source_menace_id
            )
            node["children"] = [build(child) for child in kids]
            return node

        menace_id = self._current_menace_id(source_menace_id)
        clause, scope_params = build_scope_clause(
            "evolution_history", Scope(scope), menace_id
        )
        query = (
            "SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, "
            "efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, "
            'reason, "trigger", performance, parent_event_id, predicted_class, actual_class '
            "FROM evolution_history WHERE rowid=?"
        )
        query = apply_scope(query, clause)
        row = self.conn.execute(query, (event_id, *scope_params)).fetchone()
        if not row:
            return None
        return build(row)

    # ------------------------------------------------------------------
    def spawn_variant(
        self,
        parent_event_id: int,
        action: str,
        *,
        reason: str = "spawn_variant",
        trigger: str = "spawn_variant",
    ) -> int:
        """Create a new evolution event branching from ``parent_event_id``."""

        row = self.conn.execute(
            "SELECT workflow_id FROM evolution_history WHERE rowid=?",
            (parent_event_id,),
        ).fetchone()
        workflow_id = row[0] if row else None
        event = EvolutionEvent(
            action=action,
            before_metric=0.0,
            after_metric=0.0,
            roi=0.0,
            workflow_id=workflow_id,
            reason=reason,
            trigger=trigger,
            parent_event_id=parent_event_id,
        )
        return self.add(event)

    def record_outcome(
        self,
        event_id: int,
        *,
        after_metric: float,
        roi: float,
        performance: float,
    ) -> None:
        """Update outcome metrics for an evolution event."""

        self.conn.execute(
            "UPDATE evolution_history SET after_metric=?, roi=?, performance=? WHERE rowid=?",
            (after_metric, roi, performance, event_id),
        )
        self.conn.commit()

    def lineage_tree(
        self,
        workflow_id: int,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[dict]:
        menace_id = self._current_menace_id(source_menace_id)
        clause, scope_params = build_scope_clause(
            "evolution_history", Scope(scope), menace_id
        )
        query = (
            "SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, "
            "efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, "
            'reason, "trigger", performance, parent_event_id, predicted_class, actual_class '
            "FROM evolution_history WHERE workflow_id=?"
        )
        query = apply_scope(query, clause)
        cur = self.conn.execute(query, (workflow_id, *scope_params))
        rows = cur.fetchall()
        by_parent: dict[int | None, list] = {}
        for row in rows:
            by_parent.setdefault(row[15], []).append(row)

        def build(pid: int | None) -> List[dict]:
            children: List[dict] = []
            for r in by_parent.get(pid, []):
                children.append(
                    {
                        "rowid": r[0],
                        "action": r[1],
                        "before_metric": r[2],
                        "after_metric": r[3],
                        "roi": r[4],
                        "predicted_roi": r[5],
                        "efficiency": r[6],
                        "bottleneck": r[7],
                        "patch_id": r[8],
                        "workflow_id": r[9],
                        "ts": r[10],
                        "trending_topic": r[11],
                        "reason": r[12],
                        "trigger": r[13],
                        "performance": r[14],
                        "parent_event_id": r[15],
                        "predicted_class": r[16],
                        "actual_class": r[17],
                        "children": build(r[0]),
                    }
                )
            return children

        return build(None)

    def summary(
        self,
        limit: int = 50,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> dict[str, float]:
        """Return simple stats for recent evolution events."""
        rows = self.fetch(limit, scope=scope, source_menace_id=source_menace_id)
        if not rows:
            return {"count": 0, "avg_roi": 0.0, "avg_delta": 0.0}
        count = len(rows)
        avg_roi = sum(r[3] for r in rows) / count
        avg_delta = sum(r[2] - r[1] for r in rows) / count
        return {"count": count, "avg_roi": avg_roi, "avg_delta": avg_delta}

    def averages(self, limit: int = 5) -> dict[str, float]:
        """Return average ROI delta and efficiency for recent cycles."""
        rows = self.fetch(limit)
        if not rows:
            return {"avg_roi_delta": 0.0, "avg_efficiency": 0.0}
        count = len(rows)
        avg_roi_delta = sum(r[2] - r[1] for r in rows) / count
        avg_eff = sum(r[5] for r in rows) / count
        return {"avg_roi_delta": avg_roi_delta, "avg_efficiency": avg_eff}


__all__ = [
    "EvolutionEvent",
    "EvolutionHistoryDB",
    "WorkflowEvolutionRecord",
]

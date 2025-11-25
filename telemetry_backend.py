"""Simple telemetry backend for ROI predictions and drift metrics.

Most fetch helpers accept a ``scope`` argument controlling menace visibility:
``"local"`` returns only records from the current menace, ``"global"`` exposes
entries from other menaces and ``"all"`` disables filtering entirely.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_router import (
    DBRouter,
    GLOBAL_ROUTER,
    LOCAL_TABLES,
    init_db_router,
    _configure_sqlite_connection,
)
from .metrics_exporter import Gauge
from .scope_utils import Scope, build_scope_clause, apply_scope

ROI_EVENTS_DB = "roi_events.db"

_BOOTSTRAP_WATCHDOG_ENV = "BOOTSTRAP_WATCHDOG_ACTIVE"
_DEFAULT_BOOTSTRAP_TIMEOUT_S = 1.0


logger = logging.getLogger(__name__)


_TABLE_ACCESS = Gauge(
    "table_access_total",
    "Count of table accesses",
    ["menace", "table", "operation"],
)

# In-memory collector for table usage grouped by menace and operation
_TABLE_ACCESS_COUNTS: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
    lambda: defaultdict(dict)
)


def _connection_path(conn: sqlite3.Connection) -> Path | None:
    """Return the file backing ``conn`` or ``None`` for in-memory databases."""

    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:
        return None
    for _, name, filename in rows:
        if name == "main" and filename:
            try:
                return Path(filename).resolve()
            except Exception:
                return None
    return None


def _select_router(
    db_path: str,
    *,
    bootstrap_safe: bool = False,
    init_timeout_s: float | None = None,
) -> DBRouter:
    """Return a router bound to ``db_path`` creating one when required."""

    desired = Path(db_path).expanduser().resolve()
    existing = GLOBAL_ROUTER
    if existing is not None:
        if existing.menace_id != "telemetry":
            existing = None
        else:
            local_path = _connection_path(existing.local_conn)
            shared_path = _connection_path(existing.shared_conn)
            if local_path != desired or shared_path != desired:
                existing = None
    if existing is None:
        return init_db_router(
            "telemetry",
            str(desired),
            str(desired),
            bootstrap_mode=bootstrap_safe,
            init_deadline_s=init_timeout_s,
        )
    return existing


def record_table_access(
    menace_id: str, table_name: str, operation: str, count: int = 1
) -> None:
    """Increment telemetry count for a table access.

    Parameters
    ----------
    menace_id:
        Identifier of the Menace instance reporting the metric.
    table_name:
        Table being accessed.
    operation:
        Operation type or table category (for example ``"read"``,
        ``"write"``, ``"shared"`` or ``"local"``).
    count:
        Number of accesses to record.  Defaults to ``1``.
    """

    try:
        _TABLE_ACCESS.labels(
            menace=menace_id, table=table_name, operation=operation
        ).inc(count)
    except Exception:  # pragma: no cover - best effort
        pass

    counts = _TABLE_ACCESS_COUNTS[menace_id][operation]
    counts[table_name] = counts.get(table_name, 0.0) + float(count)


def record_shared_table_access(table_name: str) -> None:  # pragma: no cover - legacy
    """Backward compatible wrapper around :func:`record_table_access`."""

    record_table_access("shared", table_name, "unknown")


def get_table_access_counts(*, flush: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return collected table access metrics grouped by menace and operation.

    The returned mapping has the structure `{menace: {operation: {table: count}}}`.
    Counts are stored as floats to align with the Prometheus client API. When
    `flush` is true the internal collector is cleared after the snapshot is
    taken.
    """

    snapshot: Dict[str, Dict[str, Dict[str, float]]] = {
        menace: {op: dict(tables) for op, tables in ops.items()}
        for menace, ops in _TABLE_ACCESS_COUNTS.items()
    }
    if flush:
        _TABLE_ACCESS_COUNTS.clear()
    return snapshot



class TelemetryBackend:
    """Persist ROI prediction telemetry using :class:`DBRouter`."""

    def __init__(
        self,
        db_path: str = "telemetry.db",
        *,
        router: DBRouter | None = None,
        bootstrap_safe: bool | None = None,
        init_timeout_s: float | None = None,
    ) -> None:
        self.db_path = db_path
        env_bootstrap = os.getenv(_BOOTSTRAP_WATCHDOG_ENV)
        env_flag = env_bootstrap is not None and env_bootstrap.strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }
        self.bootstrap_safe = env_flag if bootstrap_safe is None else bool(bootstrap_safe)
        self.init_timeout_s = (
            _DEFAULT_BOOTSTRAP_TIMEOUT_S
            if self.bootstrap_safe and init_timeout_s is None
            else init_timeout_s
        )
        self._noop_enabled = False
        self._noop_events: list[dict[str, Any]] = []

        if router is not None:
            chosen = router
        else:
            chosen = _select_router(
                db_path,
                bootstrap_safe=self.bootstrap_safe,
                init_timeout_s=self.init_timeout_s,
            )
        self.router = chosen
        LOCAL_TABLES.add("roi_telemetry")
        LOCAL_TABLES.add("roi_prediction_events")
        try:
            if self.bootstrap_safe and not self._configure_with_deadline(chosen):
                raise TimeoutError("telemetry bootstrap configuration deadline exceeded")
            self._init_db()
        except sqlite3.OperationalError as exc:
            if self.bootstrap_safe:
                self._enable_noop(f"operational-error: {exc}")
            else:
                raise
        except TimeoutError as exc:
            self._enable_noop(str(exc))

    # ------------------------------------------------------------------
    def _configure_with_deadline(self, router: DBRouter) -> bool:
        timeout = self.init_timeout_s
        if timeout is None or timeout <= 0:
            _configure_sqlite_connection(router.local_conn)
            _configure_sqlite_connection(router.shared_conn)
            return True

        def _run_config(conn: sqlite3.Connection) -> tuple[bool, Exception | None]:
            result: dict[str, Exception | None] = {"exc": None}

            def _target() -> None:
                try:
                    _configure_sqlite_connection(conn)
                except Exception as exc:  # pragma: no cover - defensive capture
                    result["exc"] = exc

            thread = threading.Thread(target=_target, daemon=True)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                return False, None
            return True, result["exc"]

        for name, conn in (("local", router.local_conn), ("shared", router.shared_conn)):
            ok, exc = _run_config(conn)
            if not ok:
                logger.warning(
                    "telemetry.bootstrap.configure-timeout", extra={"connection": name, "timeout_s": timeout}
                )
                return False
            if exc is not None:
                raise exc
        return True

    # ------------------------------------------------------------------
    def _enable_noop(self, reason: str) -> None:
        self._noop_enabled = True
        self.router = None  # type: ignore[assignment]
        logger.warning(
            "telemetry.bootstrap.fallback",
            extra={"reason": reason, "db_path": self.db_path, "bootstrap_safe": self.bootstrap_safe},
        )

    # ------------------------------------------------------------------
    def _record_noop_prediction(
        self,
        workflow_id: str | None,
        predicted: float | None,
        actual: float | None,
        confidence: float | None,
        scenario_deltas: Optional[Dict[str, float]],
        drift_flag: bool,
        readiness: float | None,
        scenario: str | None,
        ts: str | None,
    ) -> None:
        timestamp = ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = {
            "workflow_id": workflow_id,
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "scenario": scenario,
            "scenario_deltas": scenario_deltas or {},
            "drift_flag": bool(drift_flag),
            "readiness": readiness,
            "ts": timestamp,
        }
        self._noop_events.append(record)

    # ------------------------------------------------------------------
    def _filter_noop_history(
        self,
        workflow_id: str | None,
        *,
        scenario: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rec in self._noop_events:
            if workflow_id is not None and rec["workflow_id"] != workflow_id:
                continue
            if scenario is not None and rec["scenario"] != scenario:
                continue
            if start_ts is not None and rec["ts"] < start_ts:
                continue
            if end_ts is not None and rec["ts"] > end_ts:
                continue
            results.append(dict(rec))
        results.sort(key=lambda r: r.get("ts") or "")
        return results

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        conn = self.router.get_connection("roi_telemetry")
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roi_telemetry (
                    workflow_id TEXT,
                        predicted REAL,
                        actual REAL,
                        confidence REAL,
                        scenario TEXT,
                        scenario_deltas TEXT,
                        drift_flag INTEGER,
                        readiness REAL,
                        ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source_menace_id TEXT NOT NULL
                    )
                    """
                )
            for stmt in (
                "ALTER TABLE roi_telemetry ADD COLUMN readiness REAL",
                "ALTER TABLE roi_telemetry ADD COLUMN scenario TEXT",
                "ALTER TABLE roi_telemetry ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_roi_telemetry_source_menace_id ON roi_telemetry(source_menace_id)"
            )

    # ------------------------------------------------------------------
    def log_prediction(
        self,
        workflow_id: str | None,
        predicted: float | None,
        actual: float | None,
        confidence: float | None,
        scenario_deltas: Optional[Dict[str, float]],
        drift_flag: bool,
        readiness: float | None,
        scenario: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Store a prediction outcome and associated metrics."""

        if self._noop_enabled or self.router is None:
            self._record_noop_prediction(
                workflow_id,
                predicted,
                actual,
                confidence,
                scenario_deltas,
                drift_flag,
                readiness,
                scenario,
                ts,
            )
            return

        conn = self.router.get_connection("roi_telemetry")
        with conn:
            params = (
                workflow_id,
                None if predicted is None else float(predicted),
                None if actual is None else float(actual),
                None if confidence is None else float(confidence),
                scenario,
                json.dumps(scenario_deltas or {}),
                int(bool(drift_flag)),
                None if readiness is None else float(readiness),
                self.router.menace_id,
            )
            if ts is None:
                conn.execute(
                    "INSERT INTO roi_telemetry (workflow_id, predicted, actual, confidence, scenario, scenario_deltas, drift_flag, readiness, source_menace_id) VALUES (?,?,?,?,?,?,?,?,?)",
                    params,
                )
            else:
                conn.execute(
                    "INSERT INTO roi_telemetry (workflow_id, predicted, actual, confidence, scenario, scenario_deltas, drift_flag, readiness, source_menace_id, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    params + (ts,),
                )

    # ------------------------------------------------------------------
    def fetch_history(
        self,
        workflow_id: str | None = None,
        *,
        scenario: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return logged prediction history filtered by workflow or scenario.

        ``scope`` controls menace visibility: ``"local"`` returns records created
        by this menace, ``"global"`` shows entries from other menaces and
        ``"all"`` disables filtering entirely. ``source_menace_id`` overrides the
        current menace identifier when querying shared databases.
        """

        if self._noop_enabled or self.router is None:
            return self._filter_noop_history(
                workflow_id,
                scenario=scenario,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        conn = self.router.get_connection("roi_telemetry")
        cur = conn.cursor()
        base = (
            "SELECT workflow_id, predicted, actual, confidence, scenario, "
            "scenario_deltas, drift_flag, readiness, ts FROM roi_telemetry"
        )
        menace_id = source_menace_id or self.router.menace_id
        clause, scope_params = build_scope_clause(
            "roi_telemetry", scope, menace_id
        )
        base = apply_scope(base, clause)
        params: List[Any] = [*scope_params]
        if workflow_id is not None:
            base = apply_scope(base, "workflow_id = ?")
            params.append(workflow_id)
        if scenario is not None:
            base = apply_scope(base, "scenario = ?")
            params.append(scenario)
        if start_ts is not None:
            base = apply_scope(base, "ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            base = apply_scope(base, "ts <= ?")
            params.append(end_ts)
        base += " ORDER BY ts"
        cur.execute(base, params)
        rows = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for wf, pred, act, conf, scen, deltas, drift, ready, ts in rows:
            result.append(
                {
                    "workflow_id": wf,
                    "predicted": pred,
                    "actual": act,
                    "confidence": conf,
                    "scenario": scen,
                    "scenario_deltas": json.loads(deltas) if deltas else {},
                    "drift_flag": bool(drift),
                    "readiness": ready,
                    "ts": ts,
                }
            )
        return result

    # ------------------------------------------------------------------
    def fetch_drift_metrics(
        self,
        workflow_id: str | None = None,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> Dict[str, Any]:
        """Return summary drift statistics for ``workflow_id``.

        ``scope`` controls menace visibility while ``source_menace_id`` allows
        querying records created by other menaces.
        """

        if self._noop_enabled or self.router is None:
            flags = [
                bool(rec["drift_flag"])
                for rec in self._noop_events
                if workflow_id is None or rec["workflow_id"] == workflow_id
            ]
            total = len(flags)
            drifted = sum(flags)
            rate = drifted / total if total else 0.0
            return {"total": total, "drifted": drifted, "rate": rate}

        conn = self.router.get_connection("roi_telemetry")
        cur = conn.cursor()
        base = "SELECT drift_flag FROM roi_telemetry"
        menace_id = source_menace_id or self.router.menace_id
        clause, scope_params = build_scope_clause(
            "roi_telemetry", scope, menace_id
        )
        base = apply_scope(base, clause)
        params: List[Any] = [*scope_params]
        if workflow_id is not None:
            base = apply_scope(base, "workflow_id = ?")
            params.append(workflow_id)
        cur.execute(base, params)
        flags = [bool(x[0]) for x in cur.fetchall()]
        total = len(flags)
        drifted = sum(flags)
        rate = drifted / total if total else 0.0
        return {"total": total, "drifted": drifted, "rate": rate}


# ---------------------------------------------------------------------------
def _init_roi_events_db(router: DBRouter | None = None) -> None:
    """Ensure the ``roi_prediction_events`` table exists."""

    router = router or GLOBAL_ROUTER
    if router is None:  # pragma: no cover - defensive
        raise RuntimeError("DBRouter is not initialised")
    LOCAL_TABLES.add("roi_prediction_events")
    conn = router.get_connection("roi_prediction_events")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roi_prediction_events (
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    workflow_id TEXT,
                    predicted_roi REAL,
                    actual_roi REAL,
                    confidence REAL,
                    scenario_deltas TEXT,
                    drift_flag INTEGER,
                    predicted_class TEXT,
                    actual_class TEXT,
                    predicted_horizons TEXT,
                    actual_horizons TEXT,
                    predicted_categories TEXT,
                    actual_categories TEXT,
                    source_menace_id TEXT NOT NULL
                )
                """
            )
        for stmt in (
            "ALTER TABLE roi_prediction_events ADD COLUMN predicted_class TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN actual_class TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN predicted_horizons TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN actual_horizons TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN predicted_categories TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN actual_categories TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN scenario_deltas TEXT",
            "ALTER TABLE roi_prediction_events ADD COLUMN drift_flag INTEGER",
            "ALTER TABLE roi_prediction_events ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_roi_prediction_events_source_menace_id ON roi_prediction_events(source_menace_id)"
        )


def log_roi_event(
    predicted_roi: float | None,
    actual_roi: float | None,
    confidence: float | None,
    scenario_deltas: Optional[Dict[str, float]],
    drift_flag: bool,
    *,
    workflow_id: str | None = None,
    predicted_class: str | None = None,
    actual_class: str | None = None,
    predicted_horizons: Optional[List[float]] = None,
    actual_horizons: Optional[List[float]] = None,
    predicted_categories: Optional[List[str]] = None,
    actual_categories: Optional[List[str]] = None,
    router: DBRouter | None = None,
) -> None:
    """Persist a ROI prediction event to ``roi_prediction_events``."""

    router = router or GLOBAL_ROUTER
    if router is None:  # pragma: no cover - defensive
        raise RuntimeError("DBRouter is not initialised")
    _init_roi_events_db(router)
    conn = router.get_connection("roi_prediction_events")
    with conn:
        conn.execute(
            "INSERT INTO roi_prediction_events (workflow_id, predicted_roi, actual_roi, confidence, scenario_deltas, drift_flag, predicted_class, actual_class, predicted_horizons, actual_horizons, predicted_categories, actual_categories, source_menace_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                workflow_id,
                None if predicted_roi is None else float(predicted_roi),
                None if actual_roi is None else float(actual_roi),
                None if confidence is None else float(confidence),
                json.dumps(scenario_deltas or {}),
                int(bool(drift_flag)),
                predicted_class,
                actual_class,
                json.dumps([float(x) for x in (predicted_horizons or [])]),
                json.dumps([float(x) for x in (actual_horizons or [])]),
                json.dumps(predicted_categories or []),
                json.dumps(actual_categories or []),
                router.menace_id,
            ),
        )


def fetch_roi_events(
    workflow_id: str | None = None,
    *,
    start_ts: str | None = None,
    end_ts: str | None = None,
    limit: int | None = None,
    router: DBRouter | None = None,
    scope: Scope | str = "local",
    source_menace_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Return persisted ROI prediction events for dashboards.

    ``source_menace_id`` allows querying records created by a different menace
    when combined with an appropriate ``scope``.
    """

    router = router or GLOBAL_ROUTER
    if router is None:  # pragma: no cover - defensive
        raise RuntimeError("DBRouter is not initialised")
    _init_roi_events_db(router)
    conn = router.get_connection("roi_prediction_events")
    cur = conn.cursor()
    base = (
        "SELECT ts, workflow_id, predicted_roi, actual_roi, confidence, scenario_deltas, drift_flag "
        "FROM roi_prediction_events"
    )
    menace_id = source_menace_id or router.menace_id
    clause, scope_params = build_scope_clause(
        "roi_prediction_events", scope, menace_id
    )
    base = apply_scope(base, clause)
    params: List[Any] = [*scope_params]
    if workflow_id is not None:
        base = apply_scope(base, "workflow_id = ?")
        params.append(workflow_id)
    if start_ts is not None:
        base = apply_scope(base, "ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        base = apply_scope(base, "ts <= ?")
        params.append(end_ts)
    base += " ORDER BY ts"
    if limit is not None:
        base += " LIMIT ?"
        params.append(int(limit))
    cur.execute(base, params)
    rows = cur.fetchall()
    return [
        {
            "ts": ts,
            "workflow_id": wf,
            "predicted_roi": pred,
            "actual_roi": act,
            "confidence": conf,
            "scenario_deltas": json.loads(deltas) if deltas else {},
            "drift_flag": bool(flag),
        }
        for ts, wf, pred, act, conf, deltas, flag in rows
    ]

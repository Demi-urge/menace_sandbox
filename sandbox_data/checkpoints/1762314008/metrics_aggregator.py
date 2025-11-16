from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict
import os
import uuid

try:  # pragma: no cover - allow running as script
    from .db_router import init_db_router  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from db_router import init_db_router  # type: ignore
try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sns = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # type: ignore

try:
    from .data_bot import MetricsDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from data_bot import MetricsDB  # type: ignore

_RETRIEVER_WIN_GAUGE = _me.Gauge(
    "retriever_win_rate", "Win rate of retrieved patches", ["origin_db"]
)
_RETRIEVER_REGRET_GAUGE = _me.Gauge(
    "retriever_regret_rate", "Regret rate of retrieved patches", ["origin_db"]
)
_RETRIEVER_STALE_GAUGE = _me.Gauge(
    "retriever_stale_cost_seconds",
    "Penalty seconds for stale embeddings beyond threshold",
    ["origin_db"],
)
_RETRIEVER_ROI_GAUGE = _me.Gauge(
    "retriever_roi_total",
    "Total ROI contribution of retrieved patches",
    ["origin_db"],
)

_EMBED_MEAN_GAUGE = _me.Gauge(
    "embedding_metric_mean",
    "Aggregated mean of embedding metrics",
    ["metric", "period"],
)
_EMBED_COUNT_GAUGE = _me.Gauge(
    "embedding_metric_count",
    "Aggregated count of embedding metrics",
    ["metric", "period"],
)
_RETR_MEAN_GAUGE = _me.Gauge(
    "retrieval_metric_mean",
    "Aggregated mean of retrieval metrics",
    ["metric", "period"],
)
_RETR_COUNT_GAUGE = _me.Gauge(
    "retrieval_metric_count",
    "Aggregated count of retrieval metrics",
    ["metric", "period"],
)


router = GLOBAL_ROUTER


def compute_retriever_stats(
    metrics_db: Path | str = "metrics.db", roi_db: Path | str = "roi.db"
) -> Dict[str, Dict[str, float]]:
    """Join retrieval metrics with ROI outcomes and log KPIs per database."""

    if pd is None:
        raise RuntimeError("pandas is required for retriever stats")

    try:
        conn = router.get_connection("patch_outcomes")
        outcomes = pd.read_sql(
            "SELECT patch_id, origin_db, success, reverted FROM patch_outcomes", conn
        )
        stale_df = pd.read_sql(
            "SELECT origin_db, stale_seconds FROM embedding_staleness", conn
        )
    except Exception:
        return {}

    roi_df = pd.DataFrame(columns=["patch_id", "revenue", "api_cost", "roi"])
    try:
        conn = router.get_connection("action_roi")
        roi_df = pd.read_sql(
            "SELECT action AS patch_id, revenue, api_cost FROM action_roi", conn
        )
        roi_df["roi"] = roi_df["revenue"].fillna(0) - roi_df["api_cost"].fillna(0)
    except Exception:
        pass

    merged = outcomes.merge(roi_df, on="patch_id", how="left")
    metrics: Dict[str, Dict[str, float]] = {}
    threshold = float(os.getenv("EMBEDDING_STALE_THRESHOLD_SECONDS", "86400"))

    for origin, grp in merged.groupby("origin_db"):
        total = len(grp)
        wins = int(grp.get("success", pd.Series()).sum()) if total else 0
        regrets = total - wins
        win_rate = wins / total if total else 0.0
        regret_rate = regrets / total if total else 0.0
        roi_total = float(grp.get("roi", pd.Series()).fillna(0).sum())

        stale_cost = 0.0
        if not stale_df.empty:
            s = stale_df[stale_df["origin_db"] == origin]
            if not s.empty:
                excess = s["stale_seconds"] - threshold
                stale_cost = float(excess.where(excess > 0, 0).sum())

        try:
            _RETRIEVER_WIN_GAUGE.labels(origin_db=origin).set(win_rate)
            _RETRIEVER_REGRET_GAUGE.labels(origin_db=origin).set(regret_rate)
            _RETRIEVER_STALE_GAUGE.labels(origin_db=origin).set(stale_cost)
            _RETRIEVER_ROI_GAUGE.labels(origin_db=origin).set(roi_total)
        except Exception:
            pass

        if MetricsDB is not None:
            try:
                MetricsDB(metrics_db).log_retriever_kpi(
                    origin, win_rate, regret_rate, stale_cost, roi_total, total
                )
            except Exception:
                pass

        metrics[origin] = {
            "win_rate": win_rate,
            "regret_rate": regret_rate,
            "stale_cost": stale_cost,
            "roi": roi_total,
            "sample_count": float(total),
        }

    return metrics


class MetricsAggregator:
    """Aggregate raw metrics and produce visualisations."""

    def __init__(
        self, db_path: Path | str = "metrics.db", out_dir: Path | str = "analytics"
    ) -> None:
        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir = self.out_dir / "heatmaps"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        # Ensure required tables exist in the metrics database.  Older
        # deployments may not have the ``embedding_stats`` or
        # ``retrieval_stats`` tables or may miss newly added columns such as
        # ``patch_id`` and ``db_source``.  Creating them here keeps the
        # aggregator self‑contained and avoids import order issues.
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create metrics tables if they do not already exist.

        The schema mirrors the one used by :class:`data_bot.MetricsDB` but is
        duplicated here so the aggregator can operate independently when run as
        a standalone script.
        """

        with self._connect("embedding_stats") as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_stats(
                    db_name TEXT,
                    tokens INTEGER,
                    wall_ms REAL,
                    store_ms REAL,
                    patch_id TEXT,
                    db_source TEXT,
                    ts TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_stats(
                    session_id TEXT,
                    origin_db TEXT,
                    record_id TEXT,
                    rank INTEGER,
                    hit INTEGER,
                    hit_rate REAL,
                    tokens_injected INTEGER,
                    contribution REAL,
                    patch_id TEXT,
                    db_source TEXT,
                    ts TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_stats_ts ON embedding_stats(ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_retrieval_stats_ts ON retrieval_stats(ts)"
            )

            cols = [r[1] for r in conn.execute(
                "PRAGMA table_info(embedding_stats)"
            ).fetchall()]
            if "patch_id" not in cols:
                conn.execute("ALTER TABLE embedding_stats ADD COLUMN patch_id TEXT")
            if "db_source" not in cols:
                conn.execute("ALTER TABLE embedding_stats ADD COLUMN db_source TEXT")
            if "ts" not in cols:
                conn.execute(
                    "ALTER TABLE embedding_stats ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP"
                )

            cols = [r[1] for r in conn.execute(
                "PRAGMA table_info(retrieval_stats)"
            ).fetchall()]
            migrations = {
                "patch_id": "ALTER TABLE retrieval_stats ADD COLUMN patch_id TEXT",
                "db_source": "ALTER TABLE retrieval_stats ADD COLUMN db_source TEXT",
                "hit_rate": "ALTER TABLE retrieval_stats ADD COLUMN hit_rate REAL",
                "tokens_injected": "ALTER TABLE retrieval_stats ADD COLUMN tokens_injected INTEGER",
                "contribution": "ALTER TABLE retrieval_stats ADD COLUMN contribution REAL",
                "ts": "ALTER TABLE retrieval_stats ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP",
            }
            for column, stmt in migrations.items():
                if column not in cols:
                    conn.execute(stmt)

    def _connect(self, table: str) -> sqlite3.Connection:
        return router.get_connection(table)

    def _aggregate_table(self, table: str, cols: List[str], period: str) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for aggregation")
        query = f"SELECT {', '.join(cols)}, ts FROM {table}"
        with self._connect(table) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            df = pd.read_sql(query, conn, parse_dates=["ts"])
        if df.empty:
            return df
        if period == "week":
            df["period"] = df["ts"].dt.to_period("W").apply(lambda r: r.start_time.date())
        elif period == "hour":
            df["period"] = df["ts"].dt.floor("H")
        else:
            df["period"] = df["ts"].dt.date
        agg = df.groupby("period")[cols].agg(["mean", "count"])
        agg.columns = ["_".join(c) for c in agg.columns.to_flat_index()]
        agg.reset_index(inplace=True)
        return agg

    def _export_csv(self, df: "pd.DataFrame", name: str) -> Path:
        path = self.out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def _heatmap(self, df: "pd.DataFrame", name: str) -> Path | None:
        if df.empty or plt is None:
            return None
        pivot = df.set_index("period")
        plt.figure(figsize=(max(6, len(pivot)), 2 + len(pivot.columns)))
        if sns:
            sns.heatmap(pivot.T, annot=True, fmt=".2f")
        else:  # pragma: no cover - simple fallback
            plt.imshow(pivot.T, aspect="auto", cmap="viridis")
            plt.colorbar()
            plt.xticks(range(len(pivot.index)), pivot.index, rotation=45, ha="right")
            plt.yticks(range(len(pivot.columns)), pivot.columns)
        plt.tight_layout()
        out = self.heatmap_dir / f"{name}.png"
        plt.savefig(out)
        plt.close()
        return out

    def _store_aggregate(self, df: "pd.DataFrame", table: str) -> None:
        if df.empty:
            return
        with self._connect(table) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)

    def run(self, period: str = "day") -> Dict[str, Path | None]:
        results: Dict[str, Path | None] = {}
        # Update retriever KPIs before aggregation
        try:
            compute_retriever_stats(self.db_path)
        except Exception:
            pass
        emb_cols = ["tokens", "wall_ms", "store_ms"]
        emb = self._aggregate_table("embedding_stats", emb_cols, period)
        results["embedding_csv"] = self._export_csv(emb, f"embedding_stats_{period}")
        results["embedding_heatmap"] = self._heatmap(emb, f"embedding_stats_{period}")
        self._store_aggregate(emb, f"embedding_stats_agg_{period}")
        if not emb.empty:
            latest = emb.iloc[-1]
            for col, val in latest.items():
                if col == "period":
                    continue
                if col.endswith("_mean"):
                    metric = col[:-5]
                    _EMBED_MEAN_GAUGE.labels(metric=metric, period=period).set(float(val))
                elif col.endswith("_count"):
                    metric = col[:-6]
                    _EMBED_COUNT_GAUGE.labels(metric=metric, period=period).set(float(val))
        ret_cols = ["rank", "hit", "tokens_injected", "contribution", "hit_rate"]
        ret = self._aggregate_table("retrieval_stats", ret_cols, period)
        results["retrieval_csv"] = self._export_csv(ret, f"retrieval_stats_{period}")
        results["retrieval_heatmap"] = self._heatmap(ret, f"retrieval_stats_{period}")
        self._store_aggregate(ret, f"retrieval_stats_agg_{period}")
        if not ret.empty:
            latest = ret.iloc[-1]
            for col, val in latest.items():
                if col == "period":
                    continue
                if col.endswith("_mean"):
                    metric = col[:-5]
                    _RETR_MEAN_GAUGE.labels(metric=metric, period=period).set(float(val))
                elif col.endswith("_count"):
                    metric = col[:-6]
                    _RETR_COUNT_GAUGE.labels(metric=metric, period=period).set(float(val))
        kpi_cols = ["win_rate", "regret_rate", "stale_penalty", "sample_count"]
        kpi = self._aggregate_table("retriever_kpi", kpi_cols, period)
        results["retriever_kpi_csv"] = self._export_csv(kpi, f"retriever_kpi_{period}")
        results["retriever_kpi_heatmap"] = self._heatmap(kpi, f"retriever_kpi_{period}")
        self._store_aggregate(kpi, f"retriever_kpi_agg_{period}")
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and generate heatmaps")
    parser.add_argument("--db", default="metrics.db", help="Path to metrics database")
    parser.add_argument(
        "--out-dir", default="analytics", help="Directory for CSV exports and heatmaps"
    )
    parser.add_argument(
        "--period",
        choices=["hour", "day", "week"],
        default="day",
        help="Aggregation period",
    )
    args = parser.parse_args()
    agg = MetricsAggregator(args.db, args.out_dir)
    agg.run(args.period)


if __name__ == "__main__":
    main()

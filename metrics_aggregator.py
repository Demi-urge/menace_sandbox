from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict

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
    MetricsDB = None  # type: ignore

_RETRIEVER_WIN_GAUGE = _me.Gauge(
    "retriever_win_rate", "Win rate of retrieved patches", ["origin_db"]
)
_RETRIEVER_REGRET_GAUGE = _me.Gauge(
    "retriever_regret_rate", "Regret rate of retrieved patches", ["origin_db"]
)
_RETRIEVER_STALE_GAUGE = _me.Gauge(
    "retriever_stale_penalty_hours",
    "Average hours since embedding last use",
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


def compute_retriever_stats(
    metrics_db: Path | str = "metrics.db", patch_db: Path | str = "patch_history.db"
) -> Dict[str, float]:
    """Join retrieval metrics with patch outcomes and log KPIs.

    Parameters
    ----------
    metrics_db:
        Path to the metrics database containing ``retrieval_metrics`` and
        ``embedding_metrics`` tables.
    patch_db:
        Path to the patch history database used for outcome lookup.
    """

    if pd is None:
        raise RuntimeError("pandas is required for retriever stats")

    m_path = Path(metrics_db)
    if not m_path.exists():
        return {"win_rate": 0.0, "regret_rate": 0.0, "stale_penalty": 0.0}

    with sqlite3.connect(m_path) as conn:
        def _load(name: str, cols: str) -> "pd.DataFrame":
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            return pd.read_sql(
                f"SELECT {cols}, ts FROM {name}", conn, parse_dates=["ts"]
            )

        rm = _load("retrieval_metrics", "origin_db, record_id, hit")
        emb = _load("embedding_metrics", "record_id")

    win_rate = regret_rate = stale_penalty = 0.0

    # Compute win/regret rate from patch outcomes
    patch_path = Path(patch_db)
    if not rm.empty and patch_path.exists():
        with sqlite3.connect(patch_path) as conn:
            patches = pd.read_sql(
                "SELECT id, roi_delta, reverted FROM patch_history", conn
            )
        hit_df = rm[(rm["hit"] == 1) & rm["origin_db"].str.contains("patch", na=False)]
        join = hit_df.merge(patches, left_on="record_id", right_on="id", how="left")
        total = len(join)
        if total:
            wins = ((join["roi_delta"] > 0) & (join["reverted"] == 0)).sum()
            regrets = total - int(wins)
            win_rate = wins / total
            regret_rate = regrets / total

    # Stale embedding penalty: time since last use of embedding
    if not rm.empty and not emb.empty:
        df = rm.merge(emb, on="record_id", suffixes=("_ret", "_emb"))
        if not df.empty:
            df.sort_values(["record_id", "ts_ret"], inplace=True)
            df["prev_ts"] = df.groupby("record_id")["ts_ret"].shift(1)
            df["prev_ts"].fillna(df["ts_emb"], inplace=True)
            df["age_hours"] = (
                df["ts_ret"] - df["prev_ts"]
            ).dt.total_seconds() / 3600.0
            stale_penalty = float(df["age_hours"].mean())

    origin = "patch_history"
    try:  # best effort metrics
        _RETRIEVER_WIN_GAUGE.labels(origin_db=origin).set(win_rate)
        _RETRIEVER_REGRET_GAUGE.labels(origin_db=origin).set(regret_rate)
        _RETRIEVER_STALE_GAUGE.labels(origin_db=origin).set(stale_penalty)
    except Exception:
        pass

    if MetricsDB is not None:
        try:
            MetricsDB(m_path).log_retriever_kpi(
                origin, win_rate, regret_rate, stale_penalty
            )
        except Exception:
            pass

    return {
        "win_rate": win_rate,
        "regret_rate": regret_rate,
        "stale_penalty": stale_penalty,
    }


class MetricsAggregator:
    """Aggregate raw metrics and produce visualisations."""

    def __init__(self, db_path: Path | str = "metrics.db", out_dir: Path | str = "analytics") -> None:
        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir = self.out_dir / "heatmaps"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _aggregate_table(self, table: str, cols: List[str], period: str) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for aggregation")
        query = f"SELECT {', '.join(cols)}, ts FROM {table}"
        with self._connect() as conn:
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
        with self._connect() as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)

    def run(self, period: str = "day") -> Dict[str, Path | None]:
        results: Dict[str, Path | None] = {}
        # Update retriever KPIs before aggregation
        try:
            compute_retriever_stats(self.db_path)
        except Exception:
            pass
        emb_cols = ["tokens", "wall_time", "index_latency"]
        emb = self._aggregate_table("embedding_metrics", emb_cols, period)
        results["embedding_csv"] = self._export_csv(emb, f"embedding_{period}")
        results["embedding_heatmap"] = self._heatmap(emb, f"embedding_{period}")
        self._store_aggregate(emb, f"embedding_agg_{period}")
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
        ret_cols = ["rank", "hit", "tokens", "score"]
        ret = self._aggregate_table("retrieval_metrics", ret_cols, period)
        results["retrieval_csv"] = self._export_csv(ret, f"retrieval_{period}")
        results["retrieval_heatmap"] = self._heatmap(ret, f"retrieval_{period}")
        self._store_aggregate(ret, f"retrieval_agg_{period}")
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
        kpi_cols = ["win_rate", "regret_rate", "stale_penalty"]
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

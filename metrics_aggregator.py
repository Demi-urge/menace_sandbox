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

    def run(self, period: str = "day") -> Dict[str, Path | None]:
        results: Dict[str, Path | None] = {}
        emb_cols = ["tokens", "wall_time", "index_latency"]
        emb = self._aggregate_table("embedding_metrics", emb_cols, period)
        results["embedding_csv"] = self._export_csv(emb, f"embedding_{period}")
        results["embedding_heatmap"] = self._heatmap(emb, f"embedding_{period}")
        ret_cols = ["rank", "hit", "tokens", "score"]
        ret = self._aggregate_table("retrieval_metrics", ret_cols, period)
        results["retrieval_csv"] = self._export_csv(ret, f"retrieval_{period}")
        results["retrieval_heatmap"] = self._heatmap(ret, f"retrieval_{period}")
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and generate heatmaps")
    parser.add_argument("--db", default="metrics.db", help="Path to metrics database")
    parser.add_argument(
        "--out-dir", default="analytics", help="Directory for CSV exports and heatmaps"
    )
    parser.add_argument(
        "--period", choices=["day", "week"], default="day", help="Aggregation period"
    )
    args = parser.parse_args()
    agg = MetricsAggregator(args.db, args.out_dir)
    agg.run(args.period)


if __name__ == "__main__":
    main()

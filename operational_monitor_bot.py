"""Operational Monitoring Bot for system-wide metrics and anomalies."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    from pyod.models.iforest import IForest  # type: ignore
except Exception:  # pragma: no cover - optional
    IForest = None  # type: ignore

from .data_bot import DataBot, MetricsDB, MetricRecord
from .database_steward_bot import ESIndex
from .splunk_logger import SplunkHEC
from .report_generation_bot import ReportGenerationBot, ReportOptions
from .db_router import DBRouter
from .admin_bot_base import AdminBotBase
from .advanced_error_management import (
    AnomalyEnsembleDetector,
    PlaybookGenerator,
)
from .autoscaler import Autoscaler
from .unified_event_bus import UnifiedEventBus


@dataclass
class AnomalyRecord:
    """Detected anomaly information."""

    bot: str
    metric: str
    value: float
    severity: float
    ts: str = datetime.utcnow().isoformat()


class AnomalyDB:
    """SQLite-backed store for anomalies."""

    def __init__(self, path: Path | str = "anomalies.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anomalies(
                bot TEXT,
                metric TEXT,
                value REAL,
                severity REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: AnomalyRecord) -> None:
        self.conn.execute(
            "INSERT INTO anomalies(bot, metric, value, severity, ts) VALUES(?,?,?,?,?)",
            (rec.bot, rec.metric, rec.value, rec.severity, rec.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, str, float, float, str]]:
        cur = self.conn.execute(
            "SELECT bot, metric, value, severity, ts FROM anomalies"
        )
        return cur.fetchall()


class OperationalMonitoringBot(AdminBotBase):
    """Collect metrics, detect anomalies and log to Elasticsearch or Splunk."""

    def __init__(
        self,
        metrics_db: MetricsDB | None = None,
        es: ESIndex | None = None,
        splunk: SplunkHEC | None = None,
        anomaly_db: AnomalyDB | None = None,
        reporter: ReportGenerationBot | None = None,
        db_router: DBRouter | None = None,
        autoscaler: Autoscaler | None = None,
        *,
        event_bus: UnifiedEventBus | None = None,
        severity_threshold: float = 5.0,
    ) -> None:
        super().__init__(db_router=db_router)
        self.db = metrics_db or MetricsDB()
        self.data_bot = DataBot(self.db)
        self.es = es or ESIndex()
        self.splunk = splunk
        self.anomaly_db = anomaly_db or AnomalyDB()
        self.reporter = reporter or ReportGenerationBot(self.db)
        self.detector = AnomalyEnsembleDetector(self.db)
        self.playbook_generator = PlaybookGenerator()
        self.autoscaler = autoscaler
        self.event_bus = event_bus
        self.severity_threshold = severity_threshold
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("OperationalMonitor")

    def collect_and_export(
        self, bot: str, response_time: float = 0.0, errors: int = 0
    ) -> MetricRecord:
        """Collect metrics via DataBot and log to Elasticsearch."""
        self.query(bot)
        rec = self.data_bot.collect(bot, response_time, errors)
        self.es.add(rec.ts, rec.__dict__)
        if self.splunk:
            self.splunk.add(rec.ts, rec.__dict__)
        return rec

    def detect_anomalies(self, bot: str, limit: int = 50) -> List[AnomalyRecord]:
        """Detect anomalies using PyOD if available, fallback to simple threshold."""
        self.query(bot)
        df = self.db.fetch(limit)
        if hasattr(df, "empty"):
            df = df[df["bot"] == bot]
            if df.empty:
                return []
        else:
            df = [r for r in df if r.get("bot") == bot]
            if not df:
                return []
        anomalies: List[AnomalyRecord] = []
        if IForest and hasattr(df, "empty"):
            model = IForest(contamination=0.1)
            X = df[["cpu", "memory", "response_time", "errors"]]
            model.fit(X)
            preds = model.predict(X)
            scores = model.decision_function(X)
            for idx, flag in zip(df.index, preds):
                if flag == 1:
                    row = df.loc[idx]
                    sev = float(-scores[list(df.index).index(idx)])
                    rec = AnomalyRecord(
                        bot=row["bot"], metric="overall", value=sev, severity=sev, ts=row["ts"]
                    )
                    anomalies.append(rec)
        else:  # pragma: no cover - fallback
            idxs = DataBot.detect_anomalies(
                df, "cpu", threshold=2.0, metrics_db=self.db
            )
            for idx in idxs:
                row = df.iloc[idx] if hasattr(df, "iloc") else df[idx]
                rec = AnomalyRecord(
                    bot=row["bot"],
                    metric="cpu",
                    value=float(row["cpu"]),
                    severity=float(row["cpu"]),
                    ts=row["ts"],
                )
                anomalies.append(rec)
        for a in anomalies:
            self.anomaly_db.add(a)
            self.es.add(a.ts, a.__dict__)
            if self.splunk:
                self.splunk.add(a.ts, a.__dict__)
        severe = [a for a in anomalies if a.severity >= self.severity_threshold]
        if severe:
            if self.autoscaler:
                try:
                    self.autoscaler.scale_up(len(severe))
                except Exception:
                    self.logger.exception("autoscaler notification failed")
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "autoscale:request",
                        {
                            "bot": bot,
                            "count": len(severe),
                            "max_severity": max(a.severity for a in severe),
                        },
                    )
                except Exception:
                    self.logger.exception("event bus publish failed")
        ensemble = self.detector.detect()
        if ensemble:
            pb = self.playbook_generator.generate(ensemble)
            self.logger.error("detected %s -> %s", ensemble, pb)
        return anomalies

    def generate_anomaly_report(self) -> Path:
        """Compile a report of recent anomalies."""
        self.query("anomaly")
        data = self.anomaly_db.fetch()
        df = pd.DataFrame(data, columns=["bot", "metric", "value", "severity", "ts"])
        if df.empty:
            return self.reporter.compile_report(
                ReportOptions(metrics=[], title="Anomaly Report"), limit=0
            )
        df.to_csv(self.reporter.reports_dir / "anomalies.csv", index=False)
        return self.reporter.compile_report(
            ReportOptions(metrics=["value", "severity"], title="Anomaly Report"),
            limit=len(df),
        )


__all__ = ["AnomalyRecord", "AnomalyDB", "OperationalMonitoringBot"]

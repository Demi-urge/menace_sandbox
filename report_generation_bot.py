"""Report Generation Bot for compiling and emailing metrics reports."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

import logging
import json

from .coding_bot_interface import self_coding_managed
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, List, Optional

registry = BotRegistry()
data_bot = DataBot(start_server=False)
logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except Exception:
    plt = None  # type: ignore
    HAVE_MATPLOTLIB = False
    logger.warning(
        "matplotlib is not installed; chart generation will be skipped"
    )
try:
    from jinja2 import Template
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback
    class Template:  # type: ignore[misc]
        """Minimal stand-in for :class:`jinja2.Template` when dependency is missing."""

        def __init__(self, text: str) -> None:
            self._text = text

        def render(self, **context: object) -> str:
            # Simple placeholder replacement for the handful of fields we use.
            output = self._text
            for key, value in context.items():
                output = output.replace(f"{{{{{key}}}}}", str(value))
            return output

try:
    import tabulate  # noqa: F401
    HAVE_TABULATE = True
except Exception:  # pragma: no cover - optional
    HAVE_TABULATE = False

from .data_bot import MetricsDB

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency
    Celery = None  # type: ignore

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - optional dependency
    BackgroundScheduler = None  # type: ignore

import threading


@dataclass
class ReportOptions:
    """Customisation options for a report."""

    metrics: Iterable[str]
    title: str = "System Report"
    recipients: List[str] | None = None


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class ReportGenerationBot:
    """Create formatted reports from Data Bot metrics and send them via email."""

    def __init__(self, db: MetricsDB | None = None, reports_dir: str | Path = "reports", smtp_server: str = "localhost") -> None:
        self.db = db or MetricsDB()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.smtp_server = smtp_server
        self.last_report_file = self.reports_dir / "last_report.txt"
        self.app: Optional[Celery] = None
        self.scheduler: Optional[BackgroundScheduler] = None
        self.timer: Optional[threading.Timer] = None

    def _chart_path(self, name: str) -> Path:
        return self.reports_dir / f"{name}.png"

    def _generate_charts(self, df: pd.DataFrame, metrics: Iterable[str]) -> List[Path]:
        if not HAVE_MATPLOTLIB:
            logger.debug("Skipping chart generation because matplotlib is unavailable")
            return []
        paths = []
        for metric in metrics:
            if metric not in df:
                continue
            fig = plt.figure()
            df.plot(y=metric, kind="line", title=metric, ax=plt.gca())
            path = self._chart_path(metric)
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
        return paths

    def compile_report(
        self,
        options: ReportOptions,
        limit: int | None = 100,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> Path:
        df = self.db.fetch(limit=limit, start=start, end=end)
        if pd is None or not hasattr(df, "__class__"):
            logger.warning(
                "pandas is unavailable; generating plain-text report without DataFrame support"
            )
            charts: list[Path] = []
            summary = json.dumps(df, indent=2, default=str)
        else:
            charts = self._generate_charts(df, options.metrics)
            desc = df[options.metrics].describe()
            summary = desc.to_markdown() if HAVE_TABULATE else desc.to_string()
        template = Template("""# {{title}}

Metrics summary:
{{summary}}
""")
        text = template.render(title=options.title, summary=summary)
        report_path = self.reports_dir / "report.txt"
        report_path.write_text(text)
        self.last_report_file.write_text(end or datetime.utcnow().isoformat())
        return report_path

    def send_report(self, report: Path, recipients: Iterable[str]) -> None:
        msg = EmailMessage()
        msg["Subject"] = report.stem
        msg["From"] = "noreply@example.com"
        msg["To"] = ", ".join(recipients)
        msg.set_content(report.read_text())
        with smtplib.SMTP(self.smtp_server) as s:
            s.send_message(msg)

    # ------------------------------------------------------------------
    def last_report_ts(self) -> str | None:
        """Return the timestamp of the last generated report if available."""
        if self.last_report_file.exists():
            return self.last_report_file.read_text().strip()
        return None

    def schedule(self, options: ReportOptions, interval: int = 60 * 60 * 24 * 7) -> None:
        """Schedule periodic report generation via Celery beat or APScheduler."""

        def _run() -> None:
            self.compile_report(
                options,
                limit=None,
                start=self.last_report_ts(),
                end=datetime.utcnow().isoformat(),
            )

        if Celery:
            self.app = Celery("reporter", broker="memory://", backend="rpc://")

            @self.app.task(name="reporter.generate")
            def _task() -> None:  # pragma: no cover - celery path
                _run()

            self.app.conf.beat_schedule = {
                "weekly-report": {"task": "reporter.generate", "schedule": interval}
            }
        elif BackgroundScheduler:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(_run, "interval", seconds=interval)
            self.scheduler.start()
        else:  # pragma: no cover - fallback
            def _loop() -> None:
                _run()
                self.timer = threading.Timer(interval, _loop)
                self.timer.daemon = True
                self.timer.start()

            _loop()


__all__ = ["ReportOptions", "ReportGenerationBot"]
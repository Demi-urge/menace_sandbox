from __future__ import annotations

"""System health watchdog.

This module depends on :class:`vector_service.ContextBuilder` for gathering
diagnostic context.  Importing this module without the ``vector_service``
package installed results in an :class:`ImportError` to make the dependency
explicit.
"""

import json
import logging
import os
import smtplib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import Iterable, Callable, TYPE_CHECKING

try:
    from . import RAISE_ERRORS
except Exception:  # pragma: no cover - fallback when package missing
    RAISE_ERRORS = False
from db_router import GLOBAL_ROUTER
from .scope_utils import Scope, build_scope_clause, apply_scope
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency
    Celery = None  # type: ignore

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - optional dependency
    BackgroundScheduler = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    from .auto_escalation_manager import AutoEscalationManager
except Exception:  # pragma: no cover - gracefully degrade in tests
    AutoEscalationManager = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .error_bot import ErrorDB
except Exception:  # pragma: no cover - gracefully degrade in tests
    ErrorDB = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .resource_allocation_optimizer import ROIDB
except Exception:  # pragma: no cover - gracefully degrade in tests
    ROIDB = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .data_bot import MetricsDB
except Exception:  # pragma: no cover - gracefully degrade in tests
    MetricsDB = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .chaos_tester import ChaosTester
except Exception:  # pragma: no cover - gracefully degrade in tests
    ChaosTester = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - gracefully degrade in tests
    KnowledgeGraph = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .advanced_error_management import SelfHealingOrchestrator, PlaybookGenerator
except Exception:  # pragma: no cover - gracefully degrade in tests
    SelfHealingOrchestrator = PlaybookGenerator = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .bot_registry import BotRegistry
except Exception:  # pragma: no cover - gracefully degrade in tests
    BotRegistry = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .escalation_protocol import EscalationProtocol, EscalationLevel
except Exception:  # pragma: no cover - gracefully degrade in tests
    EscalationProtocol = EscalationLevel = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - gracefully degrade in tests
    UnifiedEventBus = None  # type: ignore
from .retry_utils import retry

try:
    from vector_service import ContextBuilder
except Exception as exc:  # pragma: no cover - fail fast when dependency missing
    raise ImportError(
        "watchdog requires vector_service.ContextBuilder; install the"
        " vector_service package to enable context retrieval"
    ) from exc

if TYPE_CHECKING:
    from .replay_engine import ReplayValidator
    from .menace_orchestrator import MenaceOrchestrator


def _default_auto_handler(builder: ContextBuilder) -> AutoEscalationManager | None:
    try:
        builder.refresh_db_weights()
        if AutoEscalationManager is None:
            return None
        return AutoEscalationManager(context_builder=builder)
    except Exception:
        logging.exception("auto handler init failed")
        return None


@dataclass
class Thresholds:
    """Alert thresholds for the watchdog."""

    consecutive_failures: int = 3
    roi_loss_percent: float = 10.0
    downtime_hours: float = 2.0
    error_trend: float = 5.0


@dataclass
class Notifier:
    """Send alerts via Slack, Telegram or email."""

    slack_webhook: str | None = None
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    recipients: Iterable[str] | None = None
    smtp_server: str = "localhost"
    auto_handler: AutoEscalationManager | None = None

    def notify(self, message: str, attachments: Iterable[str] | None = None) -> None:
        if self.slack_webhook:
            try:
                text = message
                if attachments:
                    for path in attachments:
                        try:
                            with open(path) as f:
                                text += "\n\n" + f"Attachment: {path}\n" + f.read()
                        except Exception as exc:
                            logging.error("Failed to read attachment %s: %s", path, exc)

                @retry(Exception, attempts=3)
                def _post() -> object:
                    return requests.post(
                        self.slack_webhook,
                        data=json.dumps({"text": text}).encode(),
                        headers={"Content-Type": "application/json"},
                        timeout=3,
                    )

                _post()
            except Exception:  # pragma: no cover - network issues
                logging.error("Failed to post to slack")
            return
        if self.telegram_token and self.telegram_chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            try:
                text = message
                if attachments:
                    for path in attachments:
                        try:
                            with open(path) as f:
                                text += "\n\n" + f"Attachment: {path}\n" + f.read()
                        except Exception as exc:
                            logging.error("Failed to read attachment %s: %s", path, exc)

                @retry(Exception, attempts=3)
                def _post() -> object:
                    return requests.post(
                        url,
                        data={"chat_id": self.telegram_chat_id, "text": text},
                        timeout=3,
                    )

                _post()
            except Exception:  # pragma: no cover - network issues
                logging.error("Failed to post to telegram")
            return
        if self.recipients:
            msg = EmailMessage()
            msg["Subject"] = "Menace Watchdog Alert"
            msg["From"] = "watchdog@example.com"
            msg["To"] = ", ".join(self.recipients)
            msg.set_content(message)
            if attachments:
                for path in attachments:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        maintype = "application"
                        subtype = "octet-stream"
                        msg.add_attachment(
                            data,
                            maintype=maintype,
                            subtype=subtype,
                            filename=os.path.basename(path),
                        )
                    except Exception:
                        logging.error("Failed to attach %s", path)
            try:
                with smtplib.SMTP(self.smtp_server) as s:
                    s.send_message(msg)
            except Exception:  # pragma: no cover - smtp issues
                logging.error("Failed to send email")
            return
        logging.warning("No notification channel configured")

    def escalate(self, message: str, attachments: Iterable[str] | None = None) -> None:
        """Escalate issues automatically when possible."""
        handled = False
        if self.auto_handler:
            try:
                self.auto_handler.handle(message, attachments)
                handled = True
            except Exception:
                logging.exception("auto escalation failed")
        if not handled:
            urgent = f"URGENT: {message}"
            self.notify(urgent, attachments)


class Watchdog:
    """Monitor failure metrics and summon a human when thresholds are breached."""

    def __init__(
        self,
        error_db: ErrorDB,
        roi_db: ROIDB,
        metrics_db: MetricsDB,
        *,
        context_builder: ContextBuilder,
        registry: BotRegistry | None = None,
        thresholds: Thresholds | None = None,
        notifier: Notifier | None = None,
        healer_backend: str | None = None,
        event_bus: "UnifiedEventBus" | None = None,
        restart_log: str = "watchdog_restart.log",
        failover_hosts: Iterable[str] | None = None,
    ) -> None:
        self.error_db = error_db
        self.roi_db = roi_db
        self.metrics_db = metrics_db
        self.context_builder = context_builder
        self.thresholds = thresholds or Thresholds()
        self.notifier = notifier or Notifier()
        if self.notifier.auto_handler is None:
            self.notifier.auto_handler = _default_auto_handler(self.context_builder)
        self.registry = registry or BotRegistry()
        self.event_bus = event_bus
        self.restart_log = restart_log
        env_hosts = os.getenv("WATCHDOG_FAILOVER_HOSTS", "").split(",")
        f_hosts = [h.strip() for h in env_hosts if h.strip()]
        add_hosts = [h.strip() for h in (failover_hosts or []) if h.strip()]
        self.failover_hosts: list[str] = f_hosts or add_hosts
        self.healer = SelfHealingOrchestrator(
            KnowledgeGraph(), backend=healer_backend
        )
        self.protocol = EscalationProtocol(
            [
                EscalationLevel("primary", self.notifier),
                EscalationLevel("secondary", self.notifier),
            ]
        )
        self.logger = logging.getLogger("Watchdog")
        self.app: Celery | None = None
        self.scheduler: BackgroundScheduler | None = None
        self.synthetic_faults: list[dict[str, object]] = []
        self.failed_workflows: list[str] = []
        self.heartbeats: dict[str, float] = {}

    def _log_restart(self, message: str) -> None:
        """Append *message* with timestamp to the restart log."""
        try:
            with open(self.restart_log, "a", encoding="utf-8") as f:
                f.write(f"{time.time()}: {message}\n")
        except Exception as exc:
            self.logger.error("failed writing restart log: %s", exc)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "watchdog:log_error",
                        {"message": message, "error": str(exc)},
                    )
                except Exception:
                    self.logger.exception("failed publishing log error")

    # ------------------------------------------------------------------
    def schedule(self, interval: int = 60) -> None:
        """Begin periodic monitoring using Celery beat or APScheduler."""
        if Celery:
            self.app = Celery("watchdog", broker="memory://", backend="rpc://")

            @self.app.task(name="watchdog.check")
            def _task() -> None:
                self.check()

            self.app.conf.beat_schedule = {
                "watchdog-check": {"task": "watchdog.check", "schedule": interval}
            }
        elif BackgroundScheduler:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.check, "interval", seconds=interval)
            self.scheduler.start()
        else:  # pragma: no cover - fallback
            import threading

            def _loop() -> None:
                self.check()
                timer = threading.Timer(interval, _loop)
                timer.daemon = True
                timer.start()

            _loop()

    # ------------------------------------------------------------------
    def _consecutive_failures(
        self,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> int:
        router = GLOBAL_ROUTER
        if router is None:
            raise RuntimeError("Database router is not initialised")
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT stack_trace FROM telemetry",
            clause,
        ) + " ORDER BY id DESC LIMIT ?"
        conn = router.get_connection("errors")
        rows = conn.execute(
            query,
            [*params, self.thresholds.consecutive_failures],
        ).fetchall()
        return len(rows)

    def _roi_loss(self) -> float:
        df = self.roi_db.history(limit=2)
        if len(df) < 2:
            return 0.0
        prev = float(df.iloc[1]["revenue"] - df.iloc[1]["api_cost"])
        curr = float(df.iloc[0]["revenue"] - df.iloc[0]["api_cost"])
        if prev == 0:
            return 0.0
        return max(0.0, (prev - curr) / abs(prev) * 100.0)

    # ------------------------------------------------------------------
    def record_heartbeat(self, bot: str) -> None:
        """Record that *bot* sent a heartbeat."""
        ts = time.time()
        self.heartbeats[bot] = ts
        try:
            self.registry.record_heartbeat(bot)
        except Exception as exc:
            self.logger.error("failed recording heartbeat for %s: %s", bot, exc)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "watchdog:heartbeat_error",
                        {"bot": bot, "error": str(exc)},
                    )
                except Exception:
                    self.logger.exception("failed publishing heartbeat error")

    def _check_heartbeats(self, timeout: float = 60.0) -> None:
        """Restart bots with stale heartbeats."""
        now = time.time()
        hb_map = {**self.registry.heartbeats, **self.heartbeats}
        lost = [b for b, ts in hb_map.items() if now - ts > timeout]
        for bot in lost:
            self.logger.warning("Lost heartbeat for %s", bot)
            try:
                self.healer.heal(bot)
                self.metrics_db.log_eval(bot, "restart_attempt", 1.0)
                self._log_restart(f"restarted {bot}")
            except Exception:
                self.logger.error("Self-healing failed for %s", bot)
                self._log_restart(f"restart failed for {bot}")
                if self.failover_hosts:
                    host = self.failover_hosts.pop(0)
                    try:
                        subprocess.Popen(
                            ["ssh", host, "python", resolve_path(f"{bot}.py")],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self.logger.info("failover restart for %s on %s", bot, host)
                        self._log_restart(f"failover {bot} on {host}")
                    except Exception:
                        self.logger.error("Failover restart failed for %s on %s", bot, host)

    def _downtime_hours(self) -> float:
        df = self.metrics_db.fetch(1)
        if df.empty:
            return self.thresholds.downtime_hours + 1.0
        last = datetime.fromisoformat(str(df.iloc[0]["ts"]))
        delta = datetime.utcnow() - last
        return delta.total_seconds() / 3600.0

    def _error_rate_trend(self, limit: int = 20) -> float:
        """Return increase in average errors between early and late metrics."""
        df = self.metrics_db.fetch(limit)
        if hasattr(df, "empty"):
            if getattr(df, "empty", True) or "errors" not in df.columns:
                return 0.0
            half = len(df) // 2 or 1
            first = float(df.iloc[:half]["errors"].mean())
            last = float(df.iloc[half:]["errors"].mean())
        elif isinstance(df, list):
            if not df:
                return 0.0
            half = len(df) // 2 or 1
            first = sum(float(r.get("errors", 0.0)) for r in df[:half]) / half
            last = sum(float(r.get("errors", 0.0)) for r in df[half:]) / max(len(df) - half, 1)
        else:
            return 0.0
        return last - first

    def _thresholds_met(self) -> bool:
        fails = self._consecutive_failures()
        loss = self._roi_loss()
        down = self._downtime_hours()
        self.logger.debug("fails=%s loss=%s down=%s", fails, loss, down)
        return (
            fails >= self.thresholds.consecutive_failures
            and loss >= self.thresholds.roi_loss_percent
            and down >= self.thresholds.downtime_hours

        )

    def _run_debugger(self) -> None:
        """Run AutomatedDebugger when error trends breach the threshold."""
        trend = self._error_rate_trend()
        if trend < self.thresholds.error_trend:
            return

        class _Proxy:
            def __init__(self, db: ErrorDB) -> None:
                self.db = db

            def recent_errors(
                self,
                limit: int = 5,
                *,
                scope: Scope | str = "local",
                source_menace_id: str | None = None,
            ) -> list[str]:
                router = GLOBAL_ROUTER
                if router is None:
                    raise RuntimeError("Database router is not initialised")
                menace_id = self.db._menace_id(source_menace_id)
                clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
                query = apply_scope(
                    "SELECT stack_trace FROM telemetry",
                    clause,
                ) + " ORDER BY id DESC LIMIT ?"
                conn = router.get_connection("errors")
                cur = conn.execute(query, [*params, limit])
                return [str(r[0]) for r in cur.fetchall()]

        try:
            from .automated_debugger import AutomatedDebugger
            from .self_coding_engine import SelfCodingEngine
            from .code_database import CodeDB
            from .menace_memory_manager import MenaceMemoryManager

            self.context_builder.refresh_db_weights()
            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=self.context_builder
            )
            dbg = AutomatedDebugger(
                _Proxy(self.error_db), engine, context_builder=self.context_builder
            )
            dbg.analyse_and_fix()
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "debugger:run", {"trend": float(trend)}
                    )
                except Exception as exc:
                    self.logger.exception("debugger publish failed: %s", exc)
                    if RAISE_ERRORS:
                        raise
        except Exception:
            self.logger.exception("automated debugging failed")

    def record_fault(
        self,
        action: str,
        bot: object | None = None,
        verify: Callable[[], None] | None = None,
        *,
        workflow: str | None = None,
    ) -> bool:
        """Record a synthetic fault and test recovery if possible."""
        recovered = False
        if bot is not None and verify is not None:
            try:
                recovered = ChaosTester.validate_recovery(bot, verify)
            except Exception:
                recovered = False
        self.synthetic_faults.append(
            {
                "ts": datetime.utcnow().isoformat(),
                "action": action,
                "bot": getattr(bot, "name", None),
                "recovered": recovered,
            }
        )
        if workflow and not recovered:
            self.failed_workflows.append(workflow)
        return recovered

    def validate_workflows(
        self,
        validator: "ReplayValidator",
        orchestrator: "MenaceOrchestrator" | None = None,
    ) -> dict[str, bool]:
        """Replay failed workflows and optionally update the orchestrator."""
        results = validator.validate(self.failed_workflows)
        self.failed_workflows.clear()
        if orchestrator:
            orchestrator.update_confidence_metrics(results)
        return results

    def compile_dossier(self, limit: int = 5) -> tuple[str, list[str]]:
        """Gather recent traces and metrics and build a runbook."""
        bots: set[str] = set()
        router = GLOBAL_ROUTER
        if router is None:
            raise RuntimeError("Database router is not initialised")
        menace_id = self.error_db._menace_id(None)
        clause, params = build_scope_clause("telemetry", Scope.LOCAL, menace_id)
        query = apply_scope(
            "SELECT bot_id, stack_trace FROM telemetry",
            clause,
        ) + " ORDER BY id DESC LIMIT ?"
        conn = router.get_connection("errors")
        rows = conn.execute(query, [*params, limit]).fetchall()
        traces = []
        for bot, trace in rows:
            traces.append(trace)
            if bot:
                for b in str(bot).split(";"):
                    if b:
                        bots.add(b)
        roi = self.roi_db.history(limit=limit)
        metrics = self.metrics_db.fetch(limit=limit)
        parts = [
            "### Failure Dossier ###",
            "",
            "Stack traces:",
            "\n".join(traces) if traces else "none",
            "",
            "ROI history:",
            roi.to_csv(index=False),
            "",
            "Metrics history:",
            metrics.to_csv(index=False),
        ]
        dossier = "\n".join(parts)

        # --------------------------------------------------------------
        # build runbook summary
        if pd is not None:
            df = metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame(metrics)
            has_data = not df.empty
        else:
            # metrics is a list of dicts when pandas is unavailable
            if isinstance(metrics, list) and metrics:
                keys = metrics[0].keys()
                cols = {k: [m[k] for m in metrics] for k in keys}
                df = None
                has_data = True
            else:
                df = None
                has_data = False
        summary: dict[str, float | int] = {}
        if has_data:
            if df is not None:
                summary = {
                    "avg_cpu": float(df["cpu"].mean()),
                    "avg_memory": float(df["memory"].mean()),
                    "error_count": int(df["errors"].sum()),
                }
            else:
                cpu = sum(cols.get("cpu", [])) / len(metrics)
                mem = sum(cols.get("memory", [])) / len(metrics)
                errs = sum(cols.get("errors", []))
                summary = {
                    "avg_cpu": float(cpu),
                    "avg_memory": float(mem),
                    "error_count": int(errs),
                }
        suspected: list[str] = []
        remediation: list[str] = []
        if summary.get("error_count", 0) > 0:
            suspected.append("Increased error count")
            remediation.append(
                "Inspect logs for failures and resolve underlying issues."
            )
        if summary.get("avg_cpu", 0.0) > 80.0:
            suspected.append("High CPU usage")
            remediation.append("Optimize code or scale CPU resources.")
        if summary.get("avg_memory", 0.0) > 80.0:
            suspected.append("High memory usage")
            remediation.append("Check for memory leaks and restart services.")
        if self._roi_loss() > self.thresholds.roi_loss_percent:
            suspected.append("ROI decline")
            remediation.append("Investigate revenue drop or cost spike.")

        # gather dependency information from the knowledge graph
        deps: dict[str, list[str]] = {}
        try:
            kg = KnowledgeGraph()
            kg.ingest_error_db(self.error_db)
            for b in bots:
                deps[b] = kg.root_causes(b)
        except Exception:
            deps = {}

        runbook = {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics_summary": summary,
            "suspected_causes": suspected,
            "remediation_steps": remediation,
            "dependencies": deps,
        }
        runbook_path = f"/tmp/watchdog_runbook_{int(datetime.utcnow().timestamp())}.json"
        try:
            with open(runbook_path, "w") as f:
                json.dump(runbook, f, indent=2)
        except Exception:
            logging.error("Failed to write runbook")

        playbook_path: str | None = None
        if remediation:
            try:
                playbook_path = PlaybookGenerator().generate(remediation)
            except Exception:
                logging.error("Failed to generate playbook")

        attachments = [runbook_path]
        if playbook_path:
            attachments.append(playbook_path)

        templates_dir = resolve_path("docs/runbooks").as_posix()
        try:
            for name in os.listdir(templates_dir):
                path = os.path.join(templates_dir, name)
                if os.path.isfile(path):
                    attachments.append(path)
        except Exception as exc:
            logging.warning("runbook discovery failed: %s", exc, exc_info=True)

        return dossier, attachments

    def check(self) -> None:
        """Evaluate metrics and notify if thresholds are breached."""
        self._check_heartbeats()
        self._run_debugger()
        if self._thresholds_met():
            dossier, attachments = self.compile_dossier()
            runbook_id = self.protocol.escalate(dossier, attachments)
            self.logger.info("Escalation sent with runbook %s", runbook_id)


__all__ = ["Thresholds", "Notifier", "Watchdog"]

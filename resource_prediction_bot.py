"""Resource Prediction Bot for forecasting and optimising task resources."""

from __future__ import annotations

from .bot_registry import BotRegistry

from .coding_bot_interface import self_coding_managed, _DisabledSelfCodingManager
from .safe_repr import basic_repr
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, TYPE_CHECKING, Any
import logging
import threading

registry = BotRegistry()

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
import psutil
import networkx as nx
import pulp

try:  # pragma: no cover - support flat execution
    from .shared.self_coding_import_guard import is_self_coding_import_active
except Exception:  # pragma: no cover - fallback when package layout differs
    from shared.self_coding_import_guard import is_self_coding_import_active  # type: ignore

from .threshold_service import threshold_service

logger = logging.getLogger(__name__)


class _BootstrapDataBotProxy:
    """Lazily resolve the shared :class:`DataBot` without circular imports."""

    __slots__ = ("_instance", "_lock")

    def __init__(self) -> None:
        self._instance: DataBot | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> DataBot:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    from .data_bot import get_shared_data_bot

                    self._instance = get_shared_data_bot(start_server=False)
        return self._instance

    def reload_thresholds(self, bot: str | None = None) -> Any:
        """Reload thresholds using a lightweight path during bootstrap."""

        if self._instance is None and is_self_coding_import_active(__file__):
            try:
                return threshold_service.reload(bot)
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug(
                    "threshold reload via service failed for %s", bot, exc_info=True
                )
        return self._ensure().reload_thresholds(bot)

    def instance(self) -> DataBot:
        """Return the underlying shared :class:`DataBot` instance."""

        return self._ensure()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return True


_DATA_BOT_PROXY = _BootstrapDataBotProxy()

try:
    import risky  # type: ignore
except Exception:  # pragma: no cover - optional
    risky = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .capital_management_bot import CapitalManagementBot
    from .data_bot import DataBot
else:  # pragma: no cover - runtime fallback for type hints
    CapitalManagementBot = Any  # type: ignore[assignment]
    DataBot = Any  # type: ignore[assignment]


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu: float
    memory: float
    disk: float
    time: float


class TemplateDB:
    """Load and store historical resource usage."""

    def __init__(self, path: Path = Path("template_data.csv")) -> None:
        if pd is None:
            raise ImportError(
                "pandas is required for TemplateDB; please install the pandas package"
            )
        self.path = path
        if self.path.exists():
            try:
                self.df = pd.read_csv(self.path)
            except Exception:
                self.df = pd.DataFrame(
                    columns=["task", "cpu", "memory", "disk", "time"]
                )
        else:
            self.df = pd.DataFrame(columns=["task", "cpu", "memory", "disk", "time"])

    def query(self, task: str) -> pd.DataFrame:
        return self.df[self.df["task"] == task]

    def add(self, task: str, metrics: ResourceMetrics) -> None:
        row = {
            "task": task,
            "cpu": metrics.cpu,
            "memory": metrics.memory,
            "disk": metrics.disk,
            "time": metrics.time,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)


class ResourcePredictionBot:
    """Predict resources, detect redundancies and assess risk."""

    def __init__(
        self,
        db: TemplateDB | None = None,
        data_bot: Optional[DataBot] = None,
        capital_bot: Optional[CapitalManagementBot] = None,
    ) -> None:
        self.db = db or TemplateDB()
        if data_bot is not None:
            self.data_bot = data_bot
        else:
            try:
                self.data_bot = _DATA_BOT_PROXY.instance()
            except Exception:
                logger.debug("shared data bot unavailable during bootstrap", exc_info=True)
                self.data_bot = _DATA_BOT_PROXY
        self.capital_bot = capital_bot
        self.graph = nx.DiGraph()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return basic_repr(
            self,
            attrs={
                "db": self.db,
                "data_bot": self.data_bot,
                "capital_bot": self.capital_bot,
            },
        )

    def predict(self, task: str) -> ResourceMetrics:
        df = self.db.query(task)
        if df.empty:
            metrics = ResourceMetrics(cpu=1.0, memory=1.0, disk=10.0, time=1.0)
        else:
            metrics = ResourceMetrics(
                cpu=float(df["cpu"].mean()),
                memory=float(df["memory"].mean()),
                disk=float(df["disk"].mean()),
                time=float(df["time"].mean()),
            )

        if self.data_bot:
            data = self.data_bot.db.fetch(20)
            data = data[data["bot"] == task]
            if not data.empty:
                metrics = ResourceMetrics(
                    cpu=float(data["cpu"].mean()),
                    memory=float(data["memory"].mean()),
                    disk=float(data["disk_io"].mean()),
                    time=float(data["response_time"].mean()),
                )
            else:
                comp = self.data_bot.complexity_score(data)
                metrics = ResourceMetrics(
                    cpu=metrics.cpu * (1.0 + comp / 200.0),
                    memory=metrics.memory * (1.0 + comp / 200.0),
                    disk=metrics.disk,
                    time=metrics.time,
                )

        if self.capital_bot:
            energy = self.capital_bot.energy_score(
                load=0.5,
                success_rate=0.5,
                deploy_eff=0.5,
                failure_rate=0.1,
            )
            metrics = ResourceMetrics(
                cpu=metrics.cpu * (1.0 + energy * 0.1),
                memory=metrics.memory * (1.0 + energy * 0.1),
                disk=metrics.disk,
                time=metrics.time,
            )

        return metrics

    def monitor_live(self) -> ResourceMetrics:
        return ResourceMetrics(
            cpu=psutil.cpu_percent(),
            memory=psutil.virtual_memory().percent,
            disk=psutil.disk_usage("/").percent,
            time=0.0,
        )

    @staticmethod
    def detect_redundancies(tasks: Iterable[str]) -> List[str]:
        """Detect tasks that appear to be semantic duplicates."""
        from difflib import SequenceMatcher

        redundancies: List[str] = []
        task_list = list(tasks)
        lower_tasks = [t.lower() for t in task_list]

        for i, t1 in enumerate(lower_tasks):
            for j in range(i + 1, len(lower_tasks)):
                t2 = lower_tasks[j]
                ratio = SequenceMatcher(None, t1, t2).ratio()
                if ratio >= 0.8 or t1 in t2 or t2 in t1:
                    redundancies.append(task_list[j])
                    break

        return list(dict.fromkeys(redundancies))

    def optimise_schedule(self, tasks: Iterable[str], cpu_limit: float = 100.0) -> List[str]:
        preds: Dict[str, ResourceMetrics] = {t: self.predict(t) for t in tasks}
        prob = pulp.LpProblem("schedule", pulp.LpMinimize)
        order_vars = {t: pulp.LpVariable(f"o_{i}", lowBound=0) for i, t in enumerate(tasks)}
        prob += pulp.lpSum(order_vars.values())
        for t, var in order_vars.items():
            prob += var * preds[t].cpu <= cpu_limit
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        sorted_tasks = sorted(tasks, key=lambda t: order_vars[t].value())
        return list(sorted_tasks)

    @staticmethod
    def assess_risk(metrics: ResourceMetrics) -> float:
        if risky and hasattr(risky, "risk"):
            try:
                return float(risky.risk(metrics.time))
            except Exception:
                logger.warning("risk calculation failed", exc_info=True)
        base = metrics.cpu / 100 + metrics.memory / 100 + metrics.disk / 100
        return min(1.0, base / 3)


class _TruthyManagerWrapper:
    """Proxy object that keeps disabled managers truthy for registry checks."""

    __slots__ = ("_manager",)

    def __init__(self, manager: object) -> None:
        self._manager = manager

    def __getattr__(self, name: str) -> object:
        return getattr(self._manager, name)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return True


def _resolve_fallback_manager() -> object | None:
    """Return a lightweight manager so registration skips retry loops."""

    try:
        existing = registry.graph.nodes.get("ResourcePredictionBot", {})
        manager = existing.get("selfcoding_manager") or existing.get("manager")
        if manager is not None:
            return manager
    except Exception:  # pragma: no cover - best effort lookup
        logger.debug("ResourcePredictionBot manager lookup failed", exc_info=True)

    try:
        disabled = _DisabledSelfCodingManager(
            bot_registry=registry, data_bot=_DATA_BOT_PROXY
        )
        return _TruthyManagerWrapper(disabled)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug(
            "ResourcePredictionBot fallback manager initialisation failed",
            exc_info=True,
        )
        return None


_manager = _resolve_fallback_manager()
ResourcePredictionBot = self_coding_managed(
    bot_registry=registry,
    data_bot=_DATA_BOT_PROXY,
    manager=_manager,
)(ResourcePredictionBot)


__all__ = [
    "ResourceMetrics",
    "TemplateDB",
    "ResourcePredictionBot",
]
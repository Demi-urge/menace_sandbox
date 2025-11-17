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
import time

registry = BotRegistry()

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    psutil = None  # type: ignore[assignment]

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - rely on registry shim
    from .bot_registry import nx as nx  # type: ignore

try:
    import pulp
except ModuleNotFoundError:  # pragma: no cover - optional optimisation
    pulp = None  # type: ignore[assignment]

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
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback for type hints
    CapitalManagementBot = Any  # type: ignore[assignment]
    DataBot = Any  # type: ignore[assignment]
    SelfCodingManager = Any  # type: ignore[assignment]


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu: float
    memory: float
    disk: float
    time: float


class _InMemoryTemplateDB:
    """Simple CSV-backed fallback when pandas is unavailable."""

    __slots__ = ("path", "records")

    def __init__(self, path: Path) -> None:
        import csv

        self.path = path
        self.records: list[dict[str, float | str]] = []
        if self.path.exists():
            try:
                with self.path.open(newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if not row:
                            continue
                        self.records.append(
                            {
                                "task": row.get("task", ""),
                                "cpu": float(row.get("cpu", 0.0) or 0.0),
                                "memory": float(row.get("memory", 0.0) or 0.0),
                                "disk": float(row.get("disk", 0.0) or 0.0),
                                "time": float(row.get("time", 0.0) or 0.0),
                            }
                        )
            except Exception:
                logger.debug("failed to load fallback TemplateDB", exc_info=True)

    def query(self, task: str) -> list[dict[str, float | str]]:
        return [row for row in self.records if row.get("task") == task]

    def add(self, row: dict[str, float | str]) -> None:
        self.records.append(row)

    def save(self) -> None:
        import csv

        fieldnames = ["task", "cpu", "memory", "disk", "time"]
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.records:
                writer.writerow(row)


class TemplateDB:
    """Load and store historical resource usage."""

    _columns = ["task", "cpu", "memory", "disk", "time"]

    def __init__(self, path: Path = Path("template_data.csv")) -> None:
        self.path = path
        self._use_pandas = pd is not None
        self._fallback: _InMemoryTemplateDB | None = None
        if self._use_pandas:
            if self.path.exists():
                try:
                    self.df = pd.read_csv(self.path)
                except Exception:
                    self.df = pd.DataFrame(columns=self._columns)
            else:
                self.df = pd.DataFrame(columns=self._columns)
        else:
            self._fallback = _InMemoryTemplateDB(self.path)

    def query(self, task: str):
        if self._use_pandas:
            return self.df[self.df["task"] == task]
        assert self._fallback is not None
        return self._fallback.query(task)

    def add(self, task: str, metrics: ResourceMetrics) -> None:
        row = {
            "task": task,
            "cpu": float(metrics.cpu),
            "memory": float(metrics.memory),
            "disk": float(metrics.disk),
            "time": float(metrics.time),
        }
        if self._use_pandas:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        else:
            assert self._fallback is not None
            self._fallback.add(row)

    def save(self) -> None:
        if self._use_pandas:
            self.df.to_csv(self.path, index=False)
        else:
            assert self._fallback is not None
            self._fallback.save()


class ResourcePredictionBot:
    """Predict resources, detect redundancies and assess risk."""

    def __init__(
        self,
        db: TemplateDB | None = None,
        data_bot: Optional[DataBot] = None,
        capital_bot: Optional[CapitalManagementBot] = None,
        *,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.manager = manager
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

    @staticmethod
    def _aggregate_metrics(records: Any) -> ResourceMetrics:
        def _default() -> ResourceMetrics:
            return ResourceMetrics(cpu=1.0, memory=1.0, disk=10.0, time=1.0)

        if pd is not None and isinstance(records, pd.DataFrame):
            if records.empty:
                return _default()
            return ResourceMetrics(
                cpu=float(records["cpu"].mean()),
                memory=float(records["memory"].mean()),
                disk=float(records["disk"].mean()),
                time=float(records["time"].mean()),
            )

        try:
            iterable = list(records or [])
        except TypeError:
            return _default()

        if not iterable:
            return _default()

        def _mean(key: str) -> tuple[float, bool]:
            values = [float(row.get(key, 0.0)) for row in iterable if isinstance(row, dict)]
            if not values:
                return 0.0, False
            return sum(values) / len(values), True

        cpu_mean, has_cpu = _mean("cpu")
        mem_mean, has_mem = _mean("memory")
        disk_mean, has_disk = _mean("disk")
        time_mean, has_time = _mean("time")

        return ResourceMetrics(
            cpu=cpu_mean if has_cpu else 1.0,
            memory=mem_mean if has_mem else 1.0,
            disk=disk_mean if has_disk else 10.0,
            time=time_mean if has_time else 1.0,
        )

    def predict(self, task: str) -> ResourceMetrics:
        records = self.db.query(task)
        metrics = self._aggregate_metrics(records)

        if self.data_bot:
            try:
                data = self.data_bot.db.fetch(20)
                if hasattr(data, "__getitem__") and hasattr(data, "__class__"):
                    try:
                        data = data[data["bot"] == task]
                    except Exception:
                        logger.debug(
                            "data bot filtering failed; falling back to base metrics",
                            exc_info=True,
                        )
                        data = None
                if data is not None and hasattr(data, "empty") and not data.empty:
                    try:
                        metrics = ResourceMetrics(
                            cpu=float(data["cpu"].mean()),
                            memory=float(data["memory"].mean()),
                            disk=float(data["disk_io"].mean()),
                            time=float(data["response_time"].mean()),
                        )
                    except Exception:
                        logger.debug(
                            "data bot aggregation failed; retaining fallback metrics",
                            exc_info=True,
                        )
                elif data is not None:
                    try:
                        comp = self.data_bot.complexity_score(data)
                    except Exception:
                        logger.debug(
                            "complexity score unavailable; using existing metrics",
                            exc_info=True,
                        )
                    else:
                        metrics = ResourceMetrics(
                            cpu=metrics.cpu * (1.0 + comp / 200.0),
                            memory=metrics.memory * (1.0 + comp / 200.0),
                            disk=metrics.disk,
                            time=metrics.time,
                        )
            except Exception:
                logger.debug(
                    "data bot interaction failed; returning aggregated metrics",
                    exc_info=True,
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
        if psutil is None:
            logger.warning(
                "psutil unavailable; live resource monitoring returning baseline metrics"
            )
            try:
                import shutil
                import os

                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
                cpu_pct = min(100.0, max(0.0, load_avg * 100))
                disk = shutil.disk_usage("/")
                disk_pct = (disk.used / disk.total) * 100 if disk.total else 0.0
            except Exception:
                cpu_pct = 0.0
                disk_pct = 0.0
            return ResourceMetrics(cpu=cpu_pct, memory=0.0, disk=disk_pct, time=0.0)

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
        if pulp is None:
            logger.warning(
                "pulp not installed; returning tasks sorted by predicted CPU usage"
            )
            return sorted(tasks, key=lambda t: preds[t].cpu)
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

    @property
    def __wrapped_manager__(self) -> object:
        return self._manager


def _unwrap_manager(candidate: object | None) -> object | None:
    """Return the underlying manager instance when wrapped."""

    if candidate is None:
        return None
    wrapped = getattr(candidate, "__wrapped_manager__", None)
    if wrapped is not None:
        return wrapped
    return getattr(candidate, "_manager", candidate)


def _is_disabled_manager(candidate: object | None) -> bool:
    """Return ``True`` when *candidate* represents a disabled manager."""

    inner = _unwrap_manager(candidate)
    return isinstance(inner, _DisabledSelfCodingManager)


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


def force_manager_hotpatch(
    *, retries: int = 3, delay: float = 2.0
) -> object | None:
    """Attempt to attach a real ``SelfCodingManager`` after bootstrap.

    When the module imports before self-coding dependencies become available the
    decorator receives a disabled manager stub.  Calling this helper after the
    environment has stabilised retries ``BotRegistry``'s internal bootstrap so a
    genuine manager is captured and wired into the class.
    """

    global _manager

    internalize = getattr(registry, "_internalize_missing_coding_bot", None)
    if not callable(internalize):
        logger.debug(
            "ResourcePredictionBot hotpatch unavailable; registry missing internalize helper"
        )
        return _unwrap_manager(_manager)

    attempt = 0
    last_error: BaseException | None = None
    total_retries = max(1, retries)
    while attempt < total_retries:
        attempt += 1
        if not _is_disabled_manager(_manager):
            return _unwrap_manager(_manager)

        try:
            data_bot_obj = _DATA_BOT_PROXY.instance()
        except Exception:
            data_bot_obj = _DATA_BOT_PROXY

        try:
            internalize(
                "ResourcePredictionBot",
                manager=None,
                data_bot=data_bot_obj,
            )
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            last_error = exc
            logger.debug(
                "ResourcePredictionBot hotpatch attempt %s failed", attempt, exc_info=True
            )
            if attempt < total_retries and delay > 0:
                time.sleep(delay)
            continue

        node = registry.graph.nodes.get("ResourcePredictionBot", {})
        manager_obj = _unwrap_manager(
            node.get("selfcoding_manager") or node.get("manager")
        )
        if manager_obj is None or _is_disabled_manager(manager_obj):
            if attempt < total_retries and delay > 0:
                time.sleep(delay)
            continue

        _manager = manager_obj

        bot_cls = globals().get("ResourcePredictionBot")
        if isinstance(bot_cls, type):
            try:
                bot_cls.manager = manager_obj  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "ResourcePredictionBot hotpatch failed to update class manager",
                    exc_info=True,
                )
            try:
                bot_cls._self_coding_manual_mode = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                bot_cls(manager=manager_obj, data_bot=data_bot_obj)
            except Exception:  # pragma: no cover - best effort bootstrap
                logger.debug(
                    "ResourcePredictionBot hotpatch bootstrap instantiation failed",
                    exc_info=True,
                )
        return manager_obj

    if last_error is not None:
        logger.debug(
            "ResourcePredictionBot hotpatch exhausted retries", exc_info=last_error
        )
    return _unwrap_manager(_manager)


def schedule_manager_hotpatch(*, delay: float = 2.0, retries: int = 3) -> None:
    """Schedule :func:`force_manager_hotpatch` to run after a short delay."""

    def _runner() -> None:
        force_manager_hotpatch(retries=retries, delay=delay)

    try:
        threading.Timer(delay, _runner).start()
    except Exception:  # pragma: no cover - best effort scheduling
        logger.debug(
            "ResourcePredictionBot hotpatch scheduling failed", exc_info=True
        )


if _is_disabled_manager(_manager):
    schedule_manager_hotpatch()


__all__ = [
    "ResourceMetrics",
    "TemplateDB",
    "ResourcePredictionBot",
    "force_manager_hotpatch",
    "schedule_manager_hotpatch",
]

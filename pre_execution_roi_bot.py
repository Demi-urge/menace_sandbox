"""Pre-Execution ROI Bot for forecasting deployment costs and returns."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot
from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Dict, List, Callable, Optional, Any, Type
import logging

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
import sqlite3
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from .task_handoff_bot import TaskHandoffBot, TaskInfo, TaskPackage
from .implementation_optimiser_bot import ImplementationOptimiserBot
from .chatgpt_enhancement_bot import EnhancementDB
from .task_handoff_bot import WorkflowDB
from .database_manager import DB_PATH, update_model, init_db
from .unified_event_bus import UnifiedEventBus
from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
try:  # pragma: no cover - optional dependency
    from .adaptive_roi_predictor import AdaptiveROIPredictor
except Exception:  # pragma: no cover - predictor missing
    AdaptiveROIPredictor = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .prediction_manager_bot import PredictionManager
else:  # pragma: no cover - runtime fallback when dependency missing
    PredictionManager = Any  # type: ignore[assignment]


@lru_cache(maxsize=1)
def _prediction_manager_cls() -> Type["PredictionManager"]:
    """Resolve :class:`PredictionManager` lazily via :mod:`importlib`."""

    module = import_module(".prediction_manager_bot", __package__)
    return module.PredictionManager  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)


@dataclass
class BuildTask:
    """Minimal representation of a planned task."""

    name: str
    complexity: float
    frequency: float
    expected_income: float
    resources: Dict[str, float] = field(default_factory=dict)


@dataclass
class ROIResult:
    """Combined cost, time and ROI projection."""

    income: float
    cost: float
    time: float
    roi: float
    margin: float
    predicted_class: str = ""
    roi_pct: float = 0.0
    npv: float = 0.0


class ROIHistoryDB:
    """Store and fetch historical cost and time data."""

    def __init__(self, path: Path | str = Path("roi_history.csv")) -> None:
        self.path = Path(path)
        if pd is None:
            self.df = None
            return
        if self.path.exists():
            try:
                self.df = pd.read_csv(self.path)
            except Exception:
                self.df = pd.DataFrame(
                    columns=["compute", "storage", "api", "supervision", "income", "time"]
                )
        else:
            self.df = pd.DataFrame(
                columns=["compute", "storage", "api", "supervision", "income", "time"]
            )

    def add(
        self,
        compute: float,
        storage: float,
        api: float,
        supervision: float,
        income: float,
        time: float,
    ) -> None:
        row = {
            "compute": compute,
            "storage": storage,
            "api": api,
            "supervision": supervision,
            "income": income,
            "time": time,
        }
        if self.df is not None and pd is not None:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def save(self) -> None:
        if self.df is not None and pd is not None:
            self.df.to_csv(self.path, index=False)

    def averages(self) -> Dict[str, float]:
        if self.df is None or self.df.empty:
            return {k: 1.0 for k in ["compute", "storage", "api", "supervision", "time"]}
        return {
            "compute": float(self.df["compute"].mean()),
            "storage": float(self.df["storage"].mean()),
            "api": float(self.df["api"].mean()),
            "supervision": float(self.df["supervision"].mean()),
            "time": float(self.df["time"].mean()),
        }


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class PreExecutionROIBot:
    """Predict costs, timelines and return on investment."""

    prediction_profile = {"scope": ["roi"], "risk": ["medium"]}

    def __init__(
        self,
        history: ROIHistoryDB | None = None,
        *,
        router: DBRouter | None = None,
        models_db: Path | str = DB_PATH,
        workflows_db: Path | str = "workflows.db",
        enhancements_db: Path | str = "enhancements.db",
        data_bot: Optional[DataBot] = None,
        handoff: Optional[TaskHandoffBot] = None,
        prediction_manager: "PredictionManager" | None = None,
        event_bus: UnifiedEventBus | None = None,
    ) -> None:
        self.history = history or ROIHistoryDB()
        self.router = router or GLOBAL_ROUTER or init_db_router("pre_execution_roi")
        self.models_db = Path(models_db)
        self.workflows_db = Path(workflows_db)
        self.enhancements_db = Path(enhancements_db)
        self.data_bot = data_bot or DataBot()
        self.handoff = handoff or TaskHandoffBot(event_bus=event_bus)
        self.event_bus = event_bus
        if prediction_manager is None:
            try:
                prediction_cls = _prediction_manager_cls()
                prediction_manager = prediction_cls(
                    data_bot=self.data_bot,
                )
            except Exception:
                prediction_manager = None
        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.predictor = None
        if AdaptiveROIPredictor is not None:
            try:
                self.predictor = AdaptiveROIPredictor()
            except Exception:
                self.predictor = None

    def _apply_prediction_bots(self, base: float, feats: Iterable[float]) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        for bid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bid)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(list(feats))
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue
        return float(score / count)

    def predict_metric(self, metric: str, feats: Iterable[float]) -> float:
        """Return averaged ``metric`` prediction from assigned bots."""
        if not self.prediction_manager:
            return 0.0
        score = 0.0
        count = 0
        vec = list(feats)
        for bid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bid)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict_metric", None)
            if not callable(pred):
                continue
            try:
                val = pred(metric, vec)
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                logger.exception(
                    "prediction bot %s metric %s failed",
                    entry.bot.__class__.__name__ if entry else "unknown",
                    metric,
                )
                continue
        return float(score / count) if count else 0.0

    # ------------------------------------------------------------------
    def _avg_model_roi(self, name: str) -> float:
        if not self.router:
            return 0.0
        try:
            with self.router.get_connection("models") as conn:
                try:
                    init_db(conn)
                except Exception:
                    pass
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT current_roi FROM models WHERE name LIKE ?",
                    (f"%{name}%",),
                ).fetchall()
            if rows:
                return float(sum(r["current_roi"] or 0.0 for r in rows) / len(rows))
        except Exception as exc:
            logger.warning("_avg_model_roi failure: %s", exc)
        return 0.0

    def _avg_workflow_profit(self) -> float:
        try:
            wf_db = WorkflowDB(Path(self.workflows_db), event_bus=self.event_bus)
            recs = wf_db.fetch()
            if recs:
                return float(
                    sum(r.estimated_profit_per_bot for r in recs) / len(recs)
                )
        except Exception as exc:
            logger.warning("_avg_workflow_profit failure: %s", exc)
        return 0.0

    def _avg_enhancement_score(self) -> float:
        try:
            enh_db = EnhancementDB(Path(self.enhancements_db))
            enhs = enh_db.fetch()
            if enhs:
                return float(sum(e.score for e in enhs) / len(enhs))
        except Exception as exc:
            logger.warning("_avg_enhancement_score failure: %s", exc)
        return 0.0

    def _data_complexity(self) -> float:
        try:
            df = self.data_bot.db.fetch(20)
            return self.data_bot.complexity_score(df)
        except Exception:
            return 0.0

    def _scrape_bonus(self) -> float:
        """Return a small bonus value based on recent Python community sentiment."""

        if requests is None:
            return 0.0

        import asyncio
        import importlib

        try:
            aiohttp = importlib.import_module("aiohttp")
        except Exception:  # pragma: no cover - optional dependency
            aiohttp = None  # type: ignore

        try:
            from textblob import TextBlob  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            TextBlob = None  # type: ignore

        urls = [
            "https://www.python.org",
            "https://pypi.org",
            "https://planetpython.org",
        ]

        async def _fetch(session: Any, url: str) -> str:
            try:
                if aiohttp is not None and session is not None:
                    async with session.get(url, timeout=3) as resp:
                        resp.raise_for_status()
                        return await resp.text()
                loop = asyncio.get_running_loop()

                def _get() -> str:
                    resp = requests.get(url, timeout=3)
                    resp.raise_for_status()
                    return resp.text

                return await loop.run_in_executor(None, _get)
            except Exception:  # pragma: no cover - network issues
                return ""

        async def _gather() -> List[str]:
            if aiohttp is not None:
                async with aiohttp.ClientSession() as sess:
                    return await asyncio.gather(*[_fetch(sess, u) for u in urls])
            return await asyncio.gather(*[_fetch(None, u) for u in urls])

        try:
            texts = asyncio.run(_gather())
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("_scrape_bonus network failure: %s", exc)
            return 0.0

        text = " ".join(t for t in texts if t)
        if not text:
            return 0.0

        sentiment = 0.0
        if TextBlob is not None:
            try:
                sentiment = TextBlob(text).sentiment.polarity
            except Exception as exc:  # pragma: no cover - sentiment failure
                logger.warning("_scrape_bonus sentiment failure: %s", exc)
                sentiment = 0.0
        else:
            pos_words = [
                "release",
                "success",
                "growth",
                "improvement",
                "stable",
                "up",
            ]
            neg_words = [
                "error",
                "deprecated",
                "warning",
                "downtime",
                "failure",
                "bug",
            ]
            low = text.lower()
            pos = sum(low.count(w) for w in pos_words)
            neg = sum(low.count(w) for w in neg_words)
            total = pos + neg
            if total:
                sentiment = (pos - neg) / total

        volume = min(len(text.split()) / 10000.0, 1.0)
        return float(sentiment * volume)

    def estimate_cost(self, tasks: Iterable[BuildTask]) -> float:
        avgs = self.history.averages()
        cost = 0.0
        for t in tasks:
            compute = t.resources.get("compute", 1.0) * avgs["compute"]
            storage = t.resources.get("storage", 1.0) * avgs["storage"]
            api = t.resources.get("api", 0.0) * avgs["api"]
            supervision = t.resources.get("supervision", 0.5) * avgs["supervision"]
            cost += compute + storage + api + supervision
        return cost

    def estimate_time(self, tasks: Iterable[BuildTask]) -> float:
        avgs = self.history.averages()
        base = sum(t.complexity * t.frequency for t in tasks)
        return base + avgs["time"]

    def forecast_roi(
        self,
        tasks: Iterable[BuildTask],
        projected_income: float,
        *,
        discount_rate: float = 0.0,
    ) -> ROIResult:
        data = list(tasks)
        cost = self.estimate_cost(data)
        time = self.estimate_time(data)
        discounted_income = projected_income / ((1 + discount_rate) ** max(time, 1.0))
        income = discounted_income / (1 + len(data) / 10)
        roi = income - cost
        margin = abs(roi) * 0.1
        roi_pct = (roi / cost * 100) if cost else 0.0
        npv = discounted_income - cost
        predicted_class = ""
        if self.predictor is not None:
            try:
                feats = [
                    [t.complexity, t.frequency, t.expected_income]
                    for t in data
                ] or [[0.0, 0.0, 0.0]]
                try:
                    _, predicted_class, _, _ = self.predictor.predict(
                        feats, horizon=len(feats)
                    )
                except TypeError:
                    _, predicted_class, _, _ = self.predictor.predict(feats)
            except Exception:
                predicted_class = ""
        logger.info(
            "forecast_roi",
            extra={"predicted_class": predicted_class, "roi": roi},
        )
        return ROIResult(
            income=income,
            cost=cost,
            time=time,
            roi=roi,
            margin=margin,
            predicted_class=predicted_class,
            roi_pct=roi_pct,
            npv=npv,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _diminishing_factor(amount: float, k: float = 0.0001) -> float:
        """Return a diminishing factor for a given spend."""
        import math

        return 1.0 - math.exp(-k * amount)

    def forecast_roi_diminishing(
        self,
        tasks: Iterable[BuildTask],
        projected_income: float,
        *,
        discount_rate: float = 0.0,
    ) -> ROIResult:
        """Forecast ROI with diminishing returns as funding increases."""
        base = self.forecast_roi(tasks, projected_income, discount_rate=discount_rate)
        factor = self._diminishing_factor(projected_income)
        income = base.income * factor
        roi = income - base.cost
        roi_pct = (roi / base.cost * 100) if base.cost else 0.0
        npv = (base.npv + base.cost) * factor - base.cost
        return ROIResult(
            income=income,
            cost=base.cost,
            time=base.time,
            roi=roi,
            margin=abs(roi) * 0.1,
            predicted_class=base.predicted_class,
            roi_pct=roi_pct,
            npv=npv,
        )

    def predict_model_roi(
        self,
        model: str,
        tasks: Iterable[BuildTask],
        *,
        discount_rate: float = 0.0,
    ) -> ROIResult:
        result = self.forecast_roi(
            tasks, sum(t.expected_income for t in tasks), discount_rate=discount_rate
        )
        profit = (
            self._avg_model_roi(model)
            + self._avg_workflow_profit()
            + self._avg_enhancement_score()
            + self._scrape_bonus()
        )
        complexity = self._data_complexity()
        final_income = result.income + profit - complexity
        final_roi = final_income - result.cost
        final_roi = self._apply_prediction_bots(final_roi, [profit, complexity])
        roi_pct = (final_roi / result.cost * 100) if result.cost else 0.0
        npv = final_income / ((1 + discount_rate) ** max(result.time, 1.0)) - result.cost
        final = ROIResult(
            income=final_income,
            cost=result.cost,
            time=result.time,
            roi=final_roi,
            margin=abs(final_roi) * 0.1,
            predicted_class=result.predicted_class,
            roi_pct=roi_pct,
            npv=npv,
        )
        try:
            if self.router:
                with self.router.get_connection("models") as conn:
                    try:
                        init_db(conn)
                    except Exception:
                        pass
                    row = conn.execute(
                        "SELECT id FROM models WHERE name LIKE ? ORDER BY id DESC LIMIT 1",
                        (f"%{model}%",),
                    ).fetchone()
                    if row:
                        update_model(
                            row[0],
                            db_path=Path(self.models_db),
                            final_roi_prediction=final.roi,
                        )
        except Exception as exc:
            logger.warning("Failed to record ROI to database: %s", exc)
        return final

    def run_scenario(
        self,
        tasks: Iterable[BuildTask],
        modifier: Callable[[List[BuildTask]], List[BuildTask]] | None = None,
    ) -> ROIResult:
        data = [BuildTask(**vars(t)) for t in tasks]
        if modifier:
            data = modifier(data)
        income = sum(t.expected_income * t.frequency for t in data)
        return self.forecast_roi(data, income)

    # ------------------------------------------------------------------
    def handoff_to_implementation(
        self,
        tasks: Iterable[BuildTask],
        optimiser: ImplementationOptimiserBot,
        *,
        title: str = "",
        description: str = "",
    ) -> TaskPackage:
        """Compile tasks and send them to the implementation optimiser."""
        infos = [
            TaskInfo(
                name=t.name,
                dependencies=[],
                resources=t.resources,
                schedule="once",
                code="# plan",
                metadata={},
            )
            for t in tasks
        ]
        package = self.handoff.compile(infos)
        try:
            self.handoff.store_plan(infos, title=title, description=description)
            self.handoff.send_package(package)
        except Exception as exc:
            logger.warning("handoff_to_implementation failure: %s", exc)
        optimiser.process(package)
        return package


class PreExecutionROIBotStub:
    """Fallback no-op implementation used when dependencies are missing."""

    prediction_profile = {"scope": ["roi"], "risk": ["medium"]}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.info("PreExecutionROIBotStub active")

    def forecast_roi(self, *args: Any, **kwargs: Any) -> ROIResult:
        return ROIResult(
            income=0.0,
            cost=0.0,
            time=0.0,
            roi=0.0,
            margin=0.0,
            roi_pct=0.0,
            npv=0.0,
        )

    def forecast_roi_diminishing(self, *a: Any, **kw: Any) -> ROIResult:
        return self.forecast_roi()

    def predict_model_roi(self, *a: Any, **kw: Any) -> ROIResult:
        return self.forecast_roi()

    def run_scenario(self, *a: Any, **kw: Any) -> ROIResult:
        return self.forecast_roi()

    def handoff_to_implementation(self, *a: Any, **kw: Any) -> TaskPackage:
        return TaskPackage(tasks=[])


__all__ = [
    "BuildTask",
    "ROIResult",
    "ROIHistoryDB",
    "PreExecutionROIBot",
    "PreExecutionROIBotStub",
]
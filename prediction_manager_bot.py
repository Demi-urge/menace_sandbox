"""Prediction Manager Bot orchestrates prediction bots."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .coding_bot_interface import self_coding_managed
from .data_bot import DataBot, MetricsDB

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
from uuid import uuid4
import os
import logging
from types import ModuleType, SimpleNamespace
from functools import lru_cache


class _LazyBotRegistry:
    """Lightweight proxy deferring :class:`BotRegistry` construction."""

    def __init__(self) -> None:
        self._real: BotRegistry | None = None
        self._pending: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.graph = SimpleNamespace(nodes={})
        self.modules: dict[str, Any] = {}
        self._bootstrap = False

    def register_bot(
        self,
        name: str,
        module_path: str | os.PathLike[str] | None = None,
        *,
        roi_threshold: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        manager: Any | None = None,
        data_bot: Any | None = None,
        patch_id: int | str | None = None,
        commit: str | None = None,
        provenance: Any | None = None,
        is_coding_bot: bool = False,
        bootstrap_mode: bool | None = None,
        lock_timeout: float | None = None,
    ) -> str:
        if self._real is None:
            self.graph.nodes.setdefault(name, {})
            self.modules.setdefault(name, module_path)
            self._pending.append(
                (
                    "register_bot",
                    (name,),
                    {
                        "module_path": module_path,
                        "roi_threshold": roi_threshold,
                        "error_threshold": error_threshold,
                        "test_failure_threshold": test_failure_threshold,
                        "manager": manager,
                        "data_bot": data_bot,
                        "patch_id": patch_id,
                        "commit": commit,
                        "provenance": provenance,
                        "is_coding_bot": is_coding_bot,
                        "bootstrap_mode": bootstrap_mode,
                        "lock_timeout": lock_timeout,
                    },
                )
            )
            return str(uuid4())
        return str(
            self._real.register_bot(  # type: ignore[return-value]
                name,
                module_path,
                roi_threshold=roi_threshold,
                error_threshold=error_threshold,
                test_failure_threshold=test_failure_threshold,
                manager=manager,
                data_bot=data_bot,
                patch_id=patch_id,
                commit=commit,
                provenance=provenance,
                is_coding_bot=is_coding_bot,
                bootstrap_mode=bootstrap_mode,
                lock_timeout=lock_timeout,
            )
        )

    def update_bot(
        self,
        name: str,
        module_path: str | os.PathLike[str],
        *,
        patch_id: int | None = None,
        commit: str | None = None,
    ) -> None:
        if self._real is None:
            self._pending.append(
                (
                    "update_bot",
                    (name, module_path),
                    {"patch_id": patch_id, "commit": commit},
                )
            )
            return
        self._real.update_bot(
            name,
            module_path,
            patch_id=patch_id,
            commit=commit,
        )

    def hot_swap_active(self) -> bool:
        if self._real is None:
            return False
        return bool(self._real.hot_swap_active())

    def _hydrate(self) -> BotRegistry:
        if self._real is None:
            real = BotRegistry(bootstrap=self._bootstrap)
            for method, args, kwargs in self._pending:
                getattr(real, method)(*args, **kwargs)
            self._pending.clear()
            self.graph = real.graph
            self.modules = real.modules
            self._real = real
        return self._real

    def set_bootstrap_mode(self, bootstrap: bool) -> None:
        self._bootstrap = bootstrap
        if self._real is not None:
            try:
                self._real.set_bootstrap_mode(bootstrap)  # type: ignore[attr-defined]
            except Exception:
                logging.getLogger(__name__).debug(
                    "failed to propagate bootstrap flag to real registry",
                    exc_info=True,
                )

    def __getattr__(self, name: str) -> Any:
        if name in {"register_bot", "update_bot", "hot_swap_active"}:
            return object.__getattribute__(self, name)
        return getattr(self._hydrate(), name)


class _LazyDataBot:
    """Proxy delaying :class:`DataBot` instantiation."""

    def __init__(self) -> None:
        self._real: DataBot | None = None
        self._pending_thresholds: set[str] = set()

    def reload_thresholds(self, name: str) -> Any:
        if self._real is None:
            self._pending_thresholds.add(name)
            return SimpleNamespace(
                roi_drop=None,
                error_threshold=None,
                test_failure_threshold=None,
            )
        return self._real.reload_thresholds(name)

    def _hydrate(self) -> DataBot:
        if self._real is None:
            real = DataBot(start_server=False)
            self._real = real
            pending = list(self._pending_thresholds)
            self._pending_thresholds.clear()
            for metric in pending:
                try:
                    real.reload_thresholds(metric)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "threshold reload failed during hydration for %s",
                        metric,
                        exc_info=True,
                    )
        return self._real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hydrate(), name)


_REGISTRY_PROXY = _LazyBotRegistry()
_DATA_BOT_PROXY = _LazyDataBot()


def _get_registry_proxy() -> _LazyBotRegistry:
    """Return the lazy registry proxy used for self-coding decorators."""

    return _REGISTRY_PROXY


def _get_data_bot_proxy() -> _LazyDataBot:
    """Return the lazy data bot proxy used for self-coding decorators."""

    return _DATA_BOT_PROXY


@lru_cache(maxsize=1)
def _get_registry(*, bootstrap: bool = False) -> BotRegistry:
    """Instantiate and return the shared :class:`BotRegistry`."""

    _REGISTRY_PROXY.set_bootstrap_mode(bootstrap)
    return _REGISTRY_PROXY._hydrate()


@lru_cache(maxsize=1)
def _get_data_bot() -> DataBot:
    """Instantiate and return the shared :class:`DataBot`."""

    return _DATA_BOT_PROXY._hydrate()

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .future_prediction_bots import (
        FutureLucrativityBot,
        FutureProfitabilityBot,
        FutureAntifragilityBot,
        FutureShannonEntropyBot,
        FutureSynergyProfitBot,
        FutureSynergyMaintainabilityBot,
        FutureSynergyCodeQualityBot,
        FutureSynergyNetworkLatencyBot,
        FutureSynergyThroughputBot,
    )
else:  # pragma: no cover - runtime fallback without importing lazily
    FutureLucrativityBot = FutureProfitabilityBot = None  # type: ignore
    FutureAntifragilityBot = FutureShannonEntropyBot = None  # type: ignore
    FutureSynergyProfitBot = FutureSynergyMaintainabilityBot = None  # type: ignore
    FutureSynergyCodeQualityBot = FutureSynergyNetworkLatencyBot = None  # type: ignore
    FutureSynergyThroughputBot = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - avoid circular imports at runtime
    from .capital_management_bot import CapitalManagementBot
if TYPE_CHECKING:
    from .ga_prediction_bot import GAPredictionBot
    from .genetic_algorithm_bot import GeneticAlgorithmBot


def _module_prefix() -> str:
    """Return the module prefix used for dynamic imports."""

    if __package__:
        return __package__.split(".")[0]
    return "menace_sandbox"


def _import_future_prediction_bots() -> ModuleType | None:
    """Import the future prediction bots module without creating cycles."""

    candidates = []
    if __package__:
        candidates.append(f"{__package__}.future_prediction_bots")
    prefix = _module_prefix()
    candidates.extend(
        [
            f"{prefix}.future_prediction_bots",
            "menace.future_prediction_bots",
            "future_prediction_bots",
        ]
    )

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        except Exception:
            logging.getLogger(__name__).debug(
                "future prediction bots import failed for %s", name, exc_info=True
            )
            return None
    return None


FUTURE_PREDICTION_EXPORTS = (
    "FutureLucrativityBot",
    "FutureProfitabilityBot",
    "FutureAntifragilityBot",
    "FutureShannonEntropyBot",
    "FutureSynergyProfitBot",
    "FutureSynergyMaintainabilityBot",
    "FutureSynergyCodeQualityBot",
    "FutureSynergyNetworkLatencyBot",
    "FutureSynergyThroughputBot",
)


@lru_cache(maxsize=1)
def _future_prediction_classes() -> Dict[str, Any]:
    """Return available future prediction bot classes, importing lazily."""

    module = _import_future_prediction_bots()
    if module is None:
        return {}
    return {name: getattr(module, name, None) for name in FUTURE_PREDICTION_EXPORTS}


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class AverageMetricBot:
    """Predict metrics by averaging recent values."""

    prediction_profile = {"scope": ["simple_avg"], "risk": ["low"]}

    def __init__(self, metric: str, data_bot: DataBot) -> None:
        self.metric = metric
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != self.metric:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or self.metric not in df.columns:
                    return 0.0
                return float(df[self.metric].mean())
            vals = [float(r.get(self.metric, 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            return 0.0


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FlexibilityPredictionBot:
    """Predict upcoming flexibility by averaging recent data."""

    prediction_profile = {"scope": ["flexibility"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != "flexibility":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "flexibility" not in df.columns:
                    return 0.0
                return float(df["flexibility"].mean())
            vals = [float(r.get("flexibility", 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logging.getLogger(__name__).exception("flexibility prediction failed")
            return 0.0


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class AntifragilityPredictionBot:
    """Predict upcoming antifragility by averaging recent data."""

    prediction_profile = {"scope": ["antifragility"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != "antifragility":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "antifragility" not in df.columns:
                    return 0.0
                return float(df["antifragility"].mean())
            vals = [float(r.get("antifragility", 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logging.getLogger(__name__).exception("antifragility prediction failed")
            return 0.0


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class ShannonEntropyPredictionBot:
    """Predict upcoming Shannon entropy by averaging recent data."""

    prediction_profile = {"scope": ["entropy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != "shannon_entropy":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "shannon_entropy" not in df.columns:
                    return 0.0
                return float(df["shannon_entropy"].mean())
            vals = [float(r.get("shannon_entropy", 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logging.getLogger(__name__).exception("shannon entropy prediction failed")
            return 0.0


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class AverageSynergyMetricBot:
    """Predict synergy metrics by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, metric: str, data_bot: DataBot | None = None) -> None:
        self.metric = metric
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != self.metric:
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or self.metric not in df.columns:
                    return 0.0
                vals = [float(v) for v in df[self.metric].tolist()]
            else:
                vals = [float(r.get(self.metric, 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logging.getLogger(__name__).exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logging.getLogger(__name__).exception("synergy metric prediction failed")
            return 0.0


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class AverageSynergyROIBot(AverageSynergyMetricBot):
    """Predict synergy ROI by averaging recent data."""

    prediction_profile = {
        "metric": ["synergy_roi"],
        "scope": ["synergy"],
        "risk": ["low"],
    }

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_roi", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergySecurityScoreBot(AverageSynergyMetricBot):
    """Predict upcoming synergy security score."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_security_score", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyEfficiencyBot(AverageSynergyMetricBot):
    """Predict upcoming synergy efficiency."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_efficiency", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyAntifragilityBot(AverageSynergyMetricBot):
    """Predict upcoming synergy antifragility."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_antifragility", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyResilienceBot(AverageSynergyMetricBot):
    """Predict upcoming synergy resilience."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_resilience", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyShannonEntropyBot(AverageSynergyMetricBot):
    """Predict upcoming synergy Shannon entropy."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_shannon_entropy", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyFlexibilityBot(AverageSynergyMetricBot):
    """Predict upcoming synergy flexibility."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_flexibility", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyEnergyConsumptionBot(AverageSynergyMetricBot):
    """Predict upcoming synergy energy consumption."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_energy_consumption", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyAdaptabilityBot(AverageSynergyMetricBot):
    """Predict upcoming synergy adaptability."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_adaptability", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergySafetyRatingBot(AverageSynergyMetricBot):
    """Predict upcoming synergy safety rating."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_safety_rating", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyRiskIndexBot(AverageSynergyMetricBot):
    """Predict upcoming synergy risk index."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_risk_index", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyRecoveryTimeBot(AverageSynergyMetricBot):
    """Predict upcoming synergy recovery time."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_recovery_time", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyDiscrepancyCountBot(AverageSynergyMetricBot):
    """Predict upcoming synergy discrepancy count."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_discrepancy_count", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyGPUUsageBot(AverageSynergyMetricBot):
    """Predict upcoming synergy GPU usage."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_gpu_usage", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyCPUUsageBot(AverageSynergyMetricBot):
    """Predict upcoming synergy CPU usage."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_cpu_usage", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyMemoryUsageBot(AverageSynergyMetricBot):
    """Predict upcoming synergy memory usage."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_memory_usage", data_bot)


@self_coding_managed(
    bot_registry=_get_registry_proxy(), data_bot=_get_data_bot_proxy()
)
class FutureSynergyLongTermLucrativityBot(AverageSynergyMetricBot):
    """Predict upcoming synergy long term lucrativity."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        super().__init__("synergy_long_term_lucrativity", data_bot)


@dataclass
class PredictionBotEntry:
    """Registered prediction bot and its profile."""

    id: str
    bot: Any
    profile: Dict[str, Iterable[str]]


class PredictionModelMatcher:
    """Simple matcher selecting bots compatible with a profile."""

    @staticmethod
    def match(
        registry: Iterable[PredictionBotEntry], profile: Dict[str, Iterable[str]]
    ) -> List[PredictionBotEntry]:
        matches: List[PredictionBotEntry] = []
        for entry in registry:
            ok = True
            for key, vals in profile.items():
                if not isinstance(vals, Iterable):
                    continue
                pvals = set(str(v).lower() for v in entry.profile.get(key, []))
                if pvals.isdisjoint({str(v).lower() for v in vals}):
                    ok = False
                    break
            if ok:
                matches.append(entry)
        return matches


class PredictionManager:
    """Central controller for prediction bots."""

    DEFAULT_METRIC_BOTS = (
        "security_score",
        "safety_rating",
        "adaptability",
        "antifragility",
        "shannon_entropy",
        "efficiency",
        "flexibility",
        "projected_lucrativity",
        "profitability",
        "patch_complexity",
        "energy_consumption",
        "resilience",
        "network_latency",
        "throughput",
        "risk_index",
        "long_term_lucrativity",
        "synergy_roi",
        "synergy_security_score",
        "synergy_efficiency",
        "synergy_antifragility",
        "synergy_resilience",
        "synergy_shannon_entropy",
        "synergy_flexibility",
        "synergy_energy_consumption",
        "synergy_adaptability",
        "synergy_safety_rating",
        "synergy_profitability",
        "synergy_projected_lucrativity",
        "synergy_maintainability",
        "synergy_code_quality",
        "synergy_network_latency",
        "synergy_throughput",
        "synergy_risk_index",
        "synergy_recovery_time",
        "synergy_discrepancy_count",
        "synergy_gpu_usage",
        "synergy_cpu_usage",
        "synergy_memory_usage",
        "synergy_long_term_lucrativity",
    )

    def __init__(
        self,
        registry_file: Path | str = "prediction_registry.json",
        *,
        data_bot: DataBot | None = None,
        capital_bot: "CapitalManagementBot | None" = None,
        default_metric_bots: Iterable[str] | None = DEFAULT_METRIC_BOTS,
    ) -> None:
        self.registry_path = Path(registry_file)
        self.registry: Dict[str, PredictionBotEntry] = {}
        self.assignments: Dict[str, List[str]] = {}
        self.matcher = PredictionModelMatcher()
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self._auto_evolution_available = True
        self._load_registry()
        if default_metric_bots and self.data_bot:
            synergy_map = {
                "synergy_security_score": FutureSynergySecurityScoreBot,
                "synergy_efficiency": FutureSynergyEfficiencyBot,
                "synergy_antifragility": FutureSynergyAntifragilityBot,
                "synergy_resilience": FutureSynergyResilienceBot,
                "synergy_shannon_entropy": FutureSynergyShannonEntropyBot,
                "synergy_flexibility": FutureSynergyFlexibilityBot,
                "synergy_energy_consumption": FutureSynergyEnergyConsumptionBot,
                "synergy_adaptability": FutureSynergyAdaptabilityBot,
                "synergy_safety_rating": FutureSynergySafetyRatingBot,
                "synergy_risk_index": FutureSynergyRiskIndexBot,
                "synergy_recovery_time": FutureSynergyRecoveryTimeBot,
                "synergy_discrepancy_count": FutureSynergyDiscrepancyCountBot,
                "synergy_gpu_usage": FutureSynergyGPUUsageBot,
                "synergy_cpu_usage": FutureSynergyCPUUsageBot,
                "synergy_memory_usage": FutureSynergyMemoryUsageBot,
                "synergy_long_term_lucrativity": FutureSynergyLongTermLucrativityBot,
            }
            for metric in default_metric_bots:
                try:
                    if str(metric).startswith("synergy_"):
                        if metric == "synergy_roi":
                            bot = AverageSynergyROIBot(self.data_bot)
                        else:
                            cls = synergy_map.get(str(metric))
                            if cls:
                                bot = cls(self.data_bot)
                            else:
                                bot = AverageSynergyMetricBot(metric, self.data_bot)
                    else:
                        bot = AverageMetricBot(metric, self.data_bot)
                    self.register_bot(bot, {"metric": [metric]})
                except Exception:
                    logging.getLogger(__name__).exception("default bot failed")
        future_classes = _future_prediction_classes()
        if self.data_bot and future_classes:
            logger = logging.getLogger(__name__)

            def _register_future_bot(
                cls_name: str, profile: Dict[str, Iterable[str]],
            ) -> None:
                cls = future_classes.get(cls_name)
                if not cls:
                    return
                try:
                    bot = cls(self.data_bot)
                    self.register_bot(bot, profile)
                except Exception:
                    logger.exception("init %s failed", cls_name)

            _register_future_bot(
                "FutureLucrativityBot",
                {
                    "metric": ["projected_lucrativity", "lucrativity"],
                    "scope": ["lucrativity"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureProfitabilityBot",
                {
                    "metric": ["profitability"],
                    "scope": ["profitability"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureAntifragilityBot",
                {
                    "metric": ["antifragility"],
                    "scope": ["antifragility"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureShannonEntropyBot",
                {
                    "metric": ["shannon_entropy"],
                    "scope": ["entropy"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureSynergyProfitBot",
                {
                    "metric": [
                        "synergy_profitability",
                        "synergy_projected_lucrativity",
                    ],
                    "scope": ["synergy"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureSynergyMaintainabilityBot",
                {
                    "metric": ["synergy_maintainability"],
                    "scope": ["synergy"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureSynergyCodeQualityBot",
                {
                    "metric": ["synergy_code_quality"],
                    "scope": ["synergy"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureSynergyNetworkLatencyBot",
                {
                    "metric": ["synergy_network_latency"],
                    "scope": ["synergy"],
                    "risk": ["low"],
                },
            )
            _register_future_bot(
                "FutureSynergyThroughputBot",
                {
                    "metric": ["synergy_throughput"],
                    "scope": ["synergy"],
                    "risk": ["low"],
                },
            )
        if self.data_bot:
            try:
                flxb = FlexibilityPredictionBot(self.data_bot)
                self.register_bot(
                    flxb,
                    {
                        "metric": ["flexibility"],
                        "scope": ["flexibility"],
                        "risk": ["low"],
                    },
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "init FlexibilityPredictionBot failed"
                )
            try:
                apb = AntifragilityPredictionBot(self.data_bot)
                self.register_bot(
                    apb,
                    {
                        "metric": ["antifragility"],
                        "scope": ["antifragility"],
                        "risk": ["low"],
                    },
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "init AntifragilityPredictionBot failed"
                )
            try:
                spb = ShannonEntropyPredictionBot(self.data_bot)
                self.register_bot(
                    spb,
                    {
                        "metric": ["shannon_entropy"],
                        "scope": ["entropy"],
                        "risk": ["low"],
                    },
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "init ShannonEntropyPredictionBot failed"
                )

    # registry persistence
    def _load_registry(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            for bot_id, info in data.items():
                self.registry[bot_id] = PredictionBotEntry(
                    id=bot_id, bot=None, profile=info["profile"]
                )

    def _save_registry(self) -> None:
        data = {bid: {"profile": entry.profile} for bid, entry in self.registry.items()}
        self.registry_path.write_text(json.dumps(data))

    # lifecycle management
    def register_bot(self, bot: Any, profile: Dict[str, Iterable[str]]) -> str:
        bot_id = str(uuid4())
        self.registry[bot_id] = PredictionBotEntry(id=bot_id, bot=bot, profile=profile)
        self._save_registry()
        return bot_id

    def retire_bot(self, bot_id: str) -> None:
        self.registry.pop(bot_id, None)
        for bots in self.assignments.values():
            if bot_id in bots:
                bots.remove(bot_id)
        self._save_registry()

    def get_prediction_bots_for(self, bot_name: str) -> List[str]:
        return self.assignments.get(bot_name, [])

    def reassign_bot(self, bot_name: str, new_bots: List[str]) -> None:
        self.assignments[bot_name] = new_bots

    # core features
    def assign_prediction_bots(self, bot: Any) -> List[str]:
        profile = getattr(bot, "prediction_profile", {})
        matches = self.matcher.match(self.registry.values(), profile)
        if not matches:
            if os.getenv("MENACE_LIGHT_IMPORTS"):
                matches = []
            else:
                try:
                    matches = self.trigger_evolution(profile)
                except Exception as exc:  # pragma: no cover - best effort
                    logging.exception("auto evolution failed: %s", exc)
                    matches = []
        bot_ids = [m.id for m in matches]
        self.assignments[getattr(bot, "name", str(bot))] = bot_ids
        return bot_ids

    def trigger_evolution(
        self, profile_space: Dict[str, Iterable[str]]
    ) -> List[PredictionBotEntry]:
        if not getattr(self, "_auto_evolution_available", True):
            return []

        try:
            from .genetic_algorithm_bot import GeneticAlgorithmBot
            from .ga_prediction_bot import GAPredictionBot
        except ImportError as exc:
            logging.getLogger(__name__).warning(
                "auto evolution unavailable: %s", exc
            )
            self._auto_evolution_available = False
            return []

        ga = GeneticAlgorithmBot(
            pop_size=4, data_bot=self.data_bot, capital_bot=self.capital_bot
        )
        ga.evolve(generations=1)

        X: List[List[float]] = []
        y: List[int] = []
        if self.data_bot:
            try:
                df = self.data_bot.db.fetch(50)
                if pd is None:
                    rows = [r for r in df if r.get("bot")]
                    for row in rows:
                        X.append(
                            [
                                float(row.get("cpu", 0.0)),
                                float(row.get("memory", 0.0)),
                                float(row.get("response_time", 0.0)),
                                float(row.get("errors", 0.0)),
                            ]
                        )
                        y.append(0 if row.get("errors", 0) else 1)
                else:
                    df = df[["cpu", "memory", "response_time", "errors"]]
                    X = df.values.tolist()
                    y = [0 if e else 1 for e in df["errors"].tolist()]
            except Exception:
                X = []
                y = []

        if not X or not y:
            X = [[0.0], [1.0], [0.2], [0.8], [0.1], [0.9], [0.3], [0.7]]
            y = [0, 1, 0, 1, 0, 1, 0, 1]

        ga_pred = GAPredictionBot(
            X,
            y,
            pop_size=2,
            data_bot=self.data_bot,
            capital_bot=self.capital_bot,
        )
        ga_pred.evolve(generations=1)

        new_bot = PredictionBotEntry(
            id=str(uuid4()),
            bot=ga_pred,
            profile=profile_space,
        )
        self.registry[new_bot.id] = new_bot
        self._save_registry()
        return [new_bot]

    def monitor_bot_performance(
        self, metrics_db: MetricsDB, threshold: float = 0.1
    ) -> None:
        df = metrics_db.fetch(limit=100)
        for bot_id, entry in list(self.registry.items()):
            if pd is None:
                rows = [r for r in df if r.get("bot") == bot_id]
                if not rows:
                    continue
                err = sum(float(r.get("errors", 0.0)) for r in rows) / len(rows)
                resp = sum(float(r.get("response_time", 0.0)) for r in rows) / len(rows)
            else:
                dfb = df[df["bot"] == bot_id]
                if dfb.empty:
                    continue
                err = float(dfb["errors"].mean() or 0.0)
                resp = float(dfb["response_time"].mean() or 0.0)
            accuracy = 1.0 - err / (err + 1.0)
            score = 1.0 / (resp + err + 1.0)
            if self.capital_bot:
                try:
                    accuracy = (
                        accuracy + min(1.0, self.capital_bot.bot_roi(bot_id) / 100.0)
                    ) / 2.0
                except Exception:
                    logging.getLogger(__name__).exception("capital bot ROI failed")
            if accuracy < threshold or score < threshold:
                self.retire_bot(bot_id)
                self.trigger_evolution(entry.profile)

    def ensure_diversity(self) -> None:
        profiles = [e.profile for e in self.registry.values()]
        if pd is None:
            counter: Dict[str, Dict[str, int]] = {}
            for prof in profiles:
                for key, vals in prof.items():
                    if not isinstance(vals, Iterable):
                        continue
                    cmap = counter.setdefault(key, {})
                    for val in vals:
                        sval = str(val)
                        cmap[sval] = cmap.get(sval, 0) + 1
            for key, cmap in counter.items():
                if not cmap:
                    continue
                counts = list(cmap.values())
                mean = sum(counts) / len(counts)
                if mean and max(counts) > 3 * mean:
                    self.trigger_evolution({key: list(cmap.keys())})
            return
        df = pd.DataFrame(profiles)
        if df.empty:
            return
        for col in df.columns:
            counts = df[col].explode().value_counts()
            if counts.empty:
                continue
            if counts.max() > 3 * counts.mean():
                self.trigger_evolution({col: [v for v in counts.index]})

    # ------------------------------------------------------------------
    def train_bots(
        self,
        bot_ids: Iterable[str],
        *,
        threshold: float = 0.8,
        max_generations: int = 3,
    ) -> List["TrainingOutcome"]:
        """Run the training pipeline for the given prediction bots."""
        from .prediction_training_pipeline import PredictionTrainingPipeline

        pipeline = PredictionTrainingPipeline(
            manager=self,
            data_bot=self.data_bot,
            capital_bot=self.capital_bot,
            threshold=threshold,
            max_generations=max_generations,
        )
        return pipeline.train(bot_ids)


__all__ = [
    "PredictionBotEntry",
    "PredictionModelMatcher",
    "PredictionManager",
    "FlexibilityPredictionBot",
    "AntifragilityPredictionBot",
    "ShannonEntropyPredictionBot",
    "AverageSynergyMetricBot",
    "AverageSynergyROIBot",
    "FutureSynergyProfitBot",
    "FutureSynergyMaintainabilityBot",
    "FutureSynergyCodeQualityBot",
    "FutureSynergyNetworkLatencyBot",
    "FutureSynergyThroughputBot",
    "FutureSynergySecurityScoreBot",
    "FutureSynergyEfficiencyBot",
    "FutureSynergyAntifragilityBot",
    "FutureSynergyResilienceBot",
    "FutureSynergyShannonEntropyBot",
    "FutureSynergyFlexibilityBot",
    "FutureSynergyEnergyConsumptionBot",
    "FutureSynergyAdaptabilityBot",
    "FutureSynergySafetyRatingBot",
    "FutureSynergyRiskIndexBot",
    "FutureSynergyRecoveryTimeBot",
    "FutureSynergyDiscrepancyCountBot",
    "FutureSynergyGPUUsageBot",
    "FutureSynergyCPUUsageBot",
    "FutureSynergyMemoryUsageBot",
    "FutureSynergyLongTermLucrativityBot",
]

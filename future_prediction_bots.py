from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Iterable, TypeVar, TYPE_CHECKING

try:  # pragma: no cover - prefer package import when available
    from .self_coding_dependency_probe import ensure_self_coding_ready
except Exception:  # pragma: no cover - fallback when executed from flat layout
    from self_coding_dependency_probe import ensure_self_coding_ready  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type checking only imports
    from .bot_registry import BotRegistry
    from .data_bot import DataBot
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime aliases avoid importing heavy dependencies early
    BotRegistry = Any  # type: ignore[assignment]
    DataBot = Any  # type: ignore[assignment]
    SelfCodingManager = Any  # type: ignore[assignment]


logger = logging.getLogger(__name__)

F = TypeVar("F")
DecoratorFactory = Callable[..., Callable[[F], F]]


def _exc_info(exc: BaseException) -> tuple[type[BaseException], BaseException, Any] | None:
    """Return ``exc_info`` tuple when debug logging is enabled."""

    if not logger.isEnabledFor(logging.DEBUG):
        return None
    return (exc.__class__, exc, exc.__traceback__)


def _noop_self_coding(
    *,
    bot_registry: BotRegistry | None = None,
    data_bot: DataBot | None = None,
    manager: SelfCodingManager | None = None,
) -> Callable[[F], F]:
    """Fallback decorator used when self-coding infrastructure is unavailable."""

    def decorator(cls: F) -> F:
        if bot_registry is not None:  # pragma: no cover - defensive attribute wiring
            setattr(cls, "bot_registry", bot_registry)
        if data_bot is not None:
            setattr(cls, "data_bot", data_bot)
        if manager is not None:
            setattr(cls, "manager", manager)
        return cls

    return decorator


def _module_prefix() -> str:
    """Return the package prefix used for runtime imports."""

    if __package__:
        return __package__.split(".")[0]
    # When executed from a flat layout fall back to the canonical package name.
    return "menace_sandbox"


def _import_optional(module: str) -> Any:
    """Import *module* relative to the sandbox, tolerating flat layouts."""

    prefix = _module_prefix()
    dotted = f"{prefix}.{module}" if not module.startswith(prefix) else module
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError:
        if dotted != module:
            return importlib.import_module(module)
        raise


def _bootstrap_self_coding() -> tuple[
    DecoratorFactory,
    BotRegistry | None,
    DataBot | None,
    SelfCodingManager | None,
]:
    """Initialise shared self-coding helpers with strong fault tolerance."""

    ready, missing = ensure_self_coding_ready()
    if not ready:
        logger.warning(
            "Self-coding decorator unavailable; future prediction bots will run without autonomous updates (missing: %s)",
            ", ".join(missing),
        )
        return _noop_self_coding, None, None, None

    try:
        registry_mod = _import_optional("bot_registry")
        data_bot_mod = _import_optional("data_bot")
        interface_mod = _import_optional("coding_bot_interface")
        manager_mod = _import_optional("self_coding_manager")
        engine_mod = _import_optional("self_coding_engine")
        pipeline_mod = _import_optional("model_automation_pipeline")
        code_db_mod = _import_optional("code_database")
        memory_mod = _import_optional("gpt_memory")
        ctx_util_mod = _import_optional("context_builder_util")
    except ModuleNotFoundError as exc:
        logger.warning(
            "Self-coding decorator unavailable; future prediction bots will run without autonomous updates: %s",
            exc,
        )
        return _noop_self_coding, None, None, None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Self-coding bootstrap failed for future prediction bots: %s",
            exc,
            exc_info=_exc_info(exc),
        )
        return _noop_self_coding, None, None, None

    try:
        registry_local = registry_mod.BotRegistry()
    except Exception as exc:  # pragma: no cover - registry bootstrap failure
        logger.warning(
            "BotRegistry bootstrap failed; disabling self-coding for future prediction bots",
            exc_info=_exc_info(exc),
        )
        return _noop_self_coding, None, None, None

    try:
        data_bot_local = data_bot_mod.DataBot(start_server=False)
    except Exception as exc:  # pragma: no cover - data bot bootstrap failure
        logger.warning(
            "DataBot bootstrap failed; disabling self-coding for future prediction bots",
            exc_info=_exc_info(exc),
        )
        return _noop_self_coding, None, None, None

    try:
        context_builder = ctx_util_mod.create_context_builder()
        engine = engine_mod.SelfCodingEngine(
            code_db_mod.CodeDB(),
            memory_mod.GPTMemoryManager(),
            context_builder=context_builder,
        )
        pipeline = pipeline_mod.ModelAutomationPipeline(
            context_builder=context_builder,
            bot_registry=registry_local,
        )
        manager_local = manager_mod.SelfCodingManager(
            engine,
            pipeline,
            data_bot=data_bot_local,
            bot_registry=registry_local,
        )
    except Exception as exc:  # pragma: no cover - bootstrap degraded
        logger.warning(
            "Self-coding services unavailable for future prediction bots: %s",
            exc,
            exc_info=_exc_info(exc),
        )
        return _noop_self_coding, None, None, None

    decorator_factory = interface_mod.self_coding_managed

    return decorator_factory, registry_local, data_bot_local, manager_local


decorator_factory, registry, data_bot, manager = _bootstrap_self_coding()


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureLucrativityBot:
    """Predict upcoming lucrativity based on recent metrics."""

    prediction_profile = {"scope": ["lucrativity"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name not in {"projected_lucrativity", "lucrativity"}:
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "projected_lucrativity" not in df.columns:
                    return 0.0
                return float(df["projected_lucrativity"].mean())
            vals = [float(r.get("projected_lucrativity", 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("lucrativity prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureProfitabilityBot:
    """Predict upcoming profitability by averaging recent data."""

    prediction_profile = {"scope": ["profitability"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name != "profitability":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "profitability" not in df.columns:
                    return 0.0
                return float(df["profitability"].mean())
            vals = [float(r.get("profitability", 0.0)) for r in df]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("profitability prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureAntifragilityBot:
    """Predict upcoming antifragility by averaging recent data."""

    prediction_profile = {"scope": ["antifragility"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
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
            logger.exception("antifragility prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureShannonEntropyBot:
    """Predict upcoming Shannon entropy by averaging recent data."""

    prediction_profile = {"scope": ["entropy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
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
            logger.exception("shannon entropy prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureSynergyProfitBot:
    """Predict upcoming synergy profitability metrics by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name not in {"synergy_profitability", "synergy_projected_lucrativity"}:
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or name not in df.columns:
                    return 0.0
                vals = [float(v) for v in df[name].tolist()]
            else:
                vals = [float(r.get(name, 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logger.exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("synergy profit prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureSynergyMaintainabilityBot:
    """Predict upcoming synergy maintainability by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name != "synergy_maintainability":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "synergy_maintainability" not in df.columns:
                    return 0.0
                vals = [float(v) for v in df["synergy_maintainability"].tolist()]
            else:
                vals = [float(r.get("synergy_maintainability", 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logger.exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("synergy maintainability prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureSynergyCodeQualityBot:
    """Predict upcoming synergy code quality by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name != "synergy_code_quality":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "synergy_code_quality" not in df.columns:
                    return 0.0
                vals = [float(v) for v in df["synergy_code_quality"].tolist()]
            else:
                vals = [float(r.get("synergy_code_quality", 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logger.exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("synergy code quality prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureSynergyNetworkLatencyBot:
    """Predict upcoming synergy network latency by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name != "synergy_network_latency":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "synergy_network_latency" not in df.columns:
                    return 0.0
                vals = [float(v) for v in df["synergy_network_latency"].tolist()]
            else:
                vals = [float(r.get("synergy_network_latency", 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logger.exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("synergy network latency prediction failed")
            return 0.0


@decorator_factory(bot_registry=registry, data_bot=data_bot, manager=manager)
class FutureSynergyThroughputBot:
    """Predict upcoming synergy throughput by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(self, name: str, _features: Iterable[float] | None = None) -> float:
        if name != "synergy_throughput":
            return 0.0
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(20)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or "synergy_throughput" not in df.columns:
                    return 0.0
                vals = [float(v) for v in df["synergy_throughput"].tolist()]
            else:
                vals = [float(r.get("synergy_throughput", 0.0)) for r in df]
            if len(vals) > 10 and os.getenv("SANDBOX_SYNERGY_MODEL"):
                try:
                    from .synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor

                    model = os.getenv("SANDBOX_SYNERGY_MODEL", "").lower()
                    if model == "arima":
                        return float(ARIMASynergyPredictor().predict(vals))
                    if model == "lstm":
                        return float(LSTMSynergyPredictor().predict(vals))
                except Exception:
                    logger.exception("synergy predictor failed")
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            logger.exception("synergy throughput prediction failed")
            return 0.0


__all__ = [
    "FutureLucrativityBot",
    "FutureProfitabilityBot",
    "FutureAntifragilityBot",
    "FutureShannonEntropyBot",
    "FutureSynergyProfitBot",
    "FutureSynergyMaintainabilityBot",
    "FutureSynergyCodeQualityBot",
    "FutureSynergyNetworkLatencyBot",
    "FutureSynergyThroughputBot",
]

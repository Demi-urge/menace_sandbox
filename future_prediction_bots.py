from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, TypeVar

from .data_bot import DataBot
from .bot_registry import BotRegistry

try:  # pragma: no cover - allow flat execution fallback
    from .self_coding_dependency_probe import ensure_self_coding_ready
except Exception:  # pragma: no cover - fallback when package import unavailable
    from self_coding_dependency_probe import ensure_self_coding_ready  # type: ignore

try:  # pragma: no cover - optional self-coding dependency
    from .coding_bot_interface import self_coding_managed as _self_coding_managed
except Exception as exc:  # pragma: no cover - degrade gracefully when unavailable
    _self_coding_managed = None  # type: ignore[assignment]
    _SELF_CODING_IMPORT_ERROR = exc
else:
    _SELF_CODING_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

F = TypeVar("F")
DecoratorFactory = Callable[..., Callable[[F], F]]


def _noop_self_coding(
    *, bot_registry: BotRegistry | None = None, data_bot: DataBot | None = None
) -> Callable[[F], F]:
    """Fallback decorator used when self-coding infrastructure is unavailable."""

    def decorator(cls: F) -> F:
        if bot_registry is not None:  # pragma: no cover - defensive attribute wiring
            setattr(cls, "bot_registry", bot_registry)
        if data_bot is not None:
            setattr(cls, "data_bot", data_bot)
        return cls

    return decorator


def _bootstrap_self_coding() -> tuple[DecoratorFactory, BotRegistry | None, DataBot | None]:
    """Initialise shared self-coding helpers with strong fault tolerance.

    The Windows sandbox frequently executes in minimal environments where
    optional dependencies may be missing.  Importing ``BotRegistry`` and
    ``DataBot`` at module load time must therefore avoid raising and instead
    fall back to a no-op decorator so the broader sandbox can continue to load
    without repeatedly retrying internalisation.
    """

    ready, missing = ensure_self_coding_ready()
    if not ready:
        logger.warning(
            "Self-coding decorator unavailable; future prediction bots will run without autonomous updates (missing: %s)",
            ", ".join(missing),
        )
        return _noop_self_coding, None, None

    if _self_coding_managed is None:
        if _SELF_CODING_IMPORT_ERROR is not None:
            logger.warning(
                "Self-coding decorator unavailable; future prediction bots will run without autonomous updates",
                exc_info=_SELF_CODING_IMPORT_ERROR if logger.isEnabledFor(logging.DEBUG) else None,
            )
        return _noop_self_coding, None, None

    try:
        registry_local = BotRegistry()
    except Exception as registry_exc:  # pragma: no cover - registry bootstrap failure
        logger.warning(
            "BotRegistry bootstrap failed; disabling self-coding for future prediction bots",
            exc_info=registry_exc if logger.isEnabledFor(logging.DEBUG) else None,
        )
        return _noop_self_coding, None, None

    try:
        data_bot_local = DataBot(start_server=False)
    except Exception as data_exc:  # pragma: no cover - data bot bootstrap failure
        logger.warning(
            "DataBot bootstrap failed; disabling self-coding for future prediction bots",
            exc_info=data_exc if logger.isEnabledFor(logging.DEBUG) else None,
        )
        return _noop_self_coding, None, None

    return _self_coding_managed, registry_local, data_bot_local


decorator_factory, registry, data_bot = _bootstrap_self_coding()


@decorator_factory(bot_registry=registry, data_bot=data_bot)
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
class FutureSynergyMaintainabilityBot:
    """Predict upcoming synergy maintainability by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
class FutureSynergyCodeQualityBot:
    """Predict upcoming synergy code quality by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
class FutureSynergyNetworkLatencyBot:
    """Predict upcoming synergy network latency by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
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


@decorator_factory(bot_registry=registry, data_bot=data_bot)
class FutureSynergyThroughputBot:
    """Predict upcoming synergy throughput by averaging recent data."""

    prediction_profile = {"scope": ["synergy"], "risk": ["low"]}

    def __init__(self, data_bot: DataBot | None = None) -> None:
        self.data_bot = data_bot

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
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

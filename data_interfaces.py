"""Shared interfaces and lightweight data containers for metrics bots.

This module centralises the minimal protocol definitions and value objects
shared between the data-oriented bots.  Keeping these definitions in a single
place avoids circular imports between large runtime modules such as
``data_bot`` and ``capital_management_bot`` while still providing rich typing
information for callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, Iterator, Mapping, Protocol, runtime_checkable


@dataclass(slots=True)
class RawMetrics:
    """Point-in-time metrics captured for a bot.

    The fields mirror the columns persisted by :class:`data_bot.MetricsDB` so
    callers can rely on a consistent structure without importing the heavy
    ``data_bot`` module.  Additional metrics can be added over time without
    breaking consumers thanks to the permissive default values.
    """

    bot: str
    cpu: float
    memory: float
    response_time: float
    disk_io: float
    net_io: float
    errors: int
    tests_failed: int = 0
    tests_run: int = 0
    revenue: float = 0.0
    expense: float = 0.0
    security_score: float = 0.0
    safety_rating: float = 0.0
    adaptability: float = 0.0
    antifragility: float = 0.0
    shannon_entropy: float = 0.0
    efficiency: float = 0.0
    flexibility: float = 0.0
    gpu_usage: float = 0.0
    projected_lucrativity: float = 0.0
    profitability: float = 0.0
    patch_complexity: float = 0.0
    patch_entropy: float = 0.0
    energy_consumption: float = 0.0
    resilience: float = 0.0
    network_latency: float = 0.0
    throughput: float = 0.0
    risk_index: float = 0.0
    maintainability: float = 0.0
    code_quality: float = 0.0
    patch_success: float = 0.0
    patch_failure_reason: str | None = None
    bottleneck: float = 0.0
    ts: str = ""


@dataclass(frozen=True)
class CapitalMetrics(Mapping[str, float]):
    """Aggregated capital signals used by :class:`CapitalManagementBot`.

    The standard fields are exposed both as dataclass attributes and via the
    mapping protocol so existing call sites that expect a dictionary interface
    continue to function.  Any metrics outside the standard set are preserved
    in :attr:`extras`.
    """

    capital: float = 0.0
    profit_trend: float = 0.0
    load: float = 0.0
    success_rate: float = 0.0
    deploy_efficiency: float = 0.0
    failure_rate: float = 0.0
    extras: Mapping[str, float] = field(default_factory=dict)

    _STANDARD_FIELDS: ClassVar[tuple[str, ...]] = (
        "capital",
        "profit_trend",
        "load",
        "success_rate",
        "deploy_efficiency",
        "failure_rate",
    )

    def __getitem__(self, key: str) -> float:
        if key in self._STANDARD_FIELDS:
            return float(getattr(self, key))
        try:
            return float(self.extras[key])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        yield from self._STANDARD_FIELDS
        for key in self.extras:
            if key not in self._STANDARD_FIELDS:
                yield key

    def __len__(self) -> int:
        return len(self._STANDARD_FIELDS) + sum(
            1 for key in self.extras if key not in self._STANDARD_FIELDS
        )

    def to_dict(self) -> Dict[str, float]:
        """Return a ``dict`` representation including extra metrics."""

        data = {field: float(getattr(self, field)) for field in self._STANDARD_FIELDS}
        for key, value in self.extras.items():
            if key not in data:
                data[key] = float(value)
        return data


class MetricsStoreProtocol(Protocol):
    """Minimal protocol required from the metrics persistence layer."""

    def fetch(self, limit: int | None = 100, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


@runtime_checkable
class DataBotInterface(Protocol):
    """Shared protocol describing the interactions used by peer bots."""

    db: MetricsStoreProtocol

    def collect(
        self,
        bot: str,
        response_time: float = 0.0,
        errors: int = 0,
        **metrics: float,
    ) -> RawMetrics:  # pragma: no cover - protocol definition
        ...

    def detect_anomalies(
        self, data: Any, metric: str, *, threshold: float = 3.0, metrics_db: Any | None = None
    ) -> Iterable[int]:  # pragma: no cover - protocol definition
        ...

    def roi(self, bot: str) -> float:  # pragma: no cover - protocol definition
        ...


__all__ = ["CapitalMetrics", "DataBotInterface", "RawMetrics"]

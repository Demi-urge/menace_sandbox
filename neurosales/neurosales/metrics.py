from __future__ import annotations

try:  # optional dependency
    from prometheus_client import (
        Counter,
        Histogram,
        CollectorRegistry,
        generate_latest,
    )
except Exception:  # pragma: no cover - optional dep missing
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    CollectorRegistry = None  # type: ignore
    generate_latest = None  # type: ignore


class Metrics:
    """Prometheus metrics tracker for token spend and conversions."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if CollectorRegistry is None or Counter is None or Histogram is None:
            self.registry = None
            self.token_spend = None
            self.conversion_count = None
            self.request_count = None
            self.failure_count = None
            self.request_latency = None
            self.auth_failures = None
            self.rate_limited = None
            return
        self.registry = registry or CollectorRegistry()
        self.token_spend = Counter(
            "token_spend_total",
            "Total tokens processed",
            registry=self.registry,
        )
        self.conversion_count = Counter(
            "conversion_total",
            "Total conversion events",
            registry=self.registry,
        )
        self.request_count = Counter(
            "requests_total",
            "Total HTTP requests",
            ["path"],
            registry=self.registry,
        )
        self.failure_count = Counter(
            "request_failures_total",
            "Total failed HTTP requests",
            ["path"],
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "auth_failures_total",
            "Total authentication failures",
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "rate_limited_total",
            "Total requests rejected due to rate limiting",
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "request_duration_seconds",
            "HTTP request duration in seconds",
            ["path"],
            registry=self.registry,
        )

    def record_tokens(self, n: int) -> None:
        if self.token_spend is not None:
            self.token_spend.inc(n)

    def record_conversion(self, n: int = 1) -> None:
        if self.conversion_count is not None:
            self.conversion_count.inc(n)

    def record_request(self, path: str, status_code: int, duration: float) -> None:
        """Record request statistics."""
        if self.request_count is not None:
            self.request_count.labels(path=path).inc()
        if self.request_latency is not None:
            self.request_latency.labels(path=path).observe(duration)
        if status_code >= 400 and self.failure_count is not None:
            self.failure_count.labels(path=path).inc()

    def exposition(self) -> bytes:
        """Return Prometheus exposition text."""
        if generate_latest is None or self.registry is None:
            return b""
        return generate_latest(self.registry)


metrics = Metrics()

__all__ = ["metrics", "Metrics"]

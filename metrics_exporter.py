from __future__ import annotations

"""Helper to expose Prometheus metrics."""

from typing import Optional, List, Iterable, Dict, Sequence, Tuple

import sys

import logging

# Ensure this module is a single instance regardless of import path
_ALIASES = (
    "metrics_exporter",
    "menace.metrics_exporter",
    "menace_sandbox.metrics_exporter",
)
for _alias in _ALIASES:
    _existing = sys.modules.get(_alias)
    if _existing is not None and _existing is not sys.modules.get(__name__):
        globals().update(_existing.__dict__)
        sys.modules[__name__] = _existing
        break
else:  # first import - register under all aliases
    _current = sys.modules[__name__]
    for _alias in _ALIASES:
        sys.modules.setdefault(_alias, _current)

try:
    from prometheus_client import (
        Gauge as _PromGauge,
        start_http_server as _start_http_server,  # type: ignore
        CollectorRegistry as _PromCollectorRegistry,  # type: ignore
        REGISTRY as _PROM_REGISTRY,  # type: ignore
    )

    def Gauge(name: str, documentation: str, labelnames: Sequence[str] | None = None, **kw: object) -> _PromGauge:  # type: ignore[override]
        existing = _PROM_REGISTRY._names_to_collectors.get(name)
        if existing is not None:
            try:
                _PROM_REGISTRY.unregister(existing)
            except Exception:
                pass
        return _PromGauge(name, documentation, labelnames=labelnames or (), **kw)

    CollectorRegistry = _PromCollectorRegistry  # type: ignore
    start_http_server = _start_http_server  # type: ignore
    _USING_STUB = False
except Exception as exc:  # pragma: no cover - optional dependency missing
    logger = logging.getLogger(__name__)
    logger.warning(
        "prometheus_client missing; using lightweight metrics server. "
        "Advanced features disabled: %s",
        exc,
    )
    _USING_STUB = True

    import threading

    class _ValueWrapper:
        def __init__(self, value: float = 0.0) -> None:
            self._value = value
            self._lock = threading.Lock()

        def set(self, value: float) -> None:
            with self._lock:
                self._value = value

        def add(self, delta: float) -> None:
            with self._lock:
                self._value += delta

        def get(self) -> float:
            with self._lock:
                return self._value

    class Registry:
        """Minimal registry for Gauge objects."""

        def __init__(self) -> None:
            self._gauges: List["Gauge"] = []

        def register(self, gauge: "Gauge") -> None:
            self._gauges.append(gauge)

        def collect(self) -> Iterable["Gauge"]:
            return list(self._gauges)

    CollectorRegistry = Registry

    _DEFAULT_REGISTRY = Registry()

    def start_http_server(
        port: int = 8000,
        addr: str = "0.0.0.0",
        *,
        registry: Registry | None = None,
    ) -> "HTTPServer":
        """Expose metrics via a minimal HTTP endpoint."""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        import atexit

        reg = registry or _DEFAULT_REGISTRY

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                body = "\n".join(g._prometheus() for g in reg.collect()) + "\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args: object) -> None:  # pragma: no cover - silence
                return

        server = HTTPServer((addr, port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        global _METRICS_SERVER
        _METRICS_SERVER = server

        def _shutdown() -> None:
            global _METRICS_SERVER
            if _METRICS_SERVER:
                try:
                    _METRICS_SERVER.shutdown()
                    _METRICS_SERVER.server_close()
                except Exception as exc:  # pragma: no cover - runtime issues
                    logger.exception(
                        "failed to stop fallback metrics server: %s", exc
                    )
                finally:
                    _METRICS_SERVER = None

        atexit.register(_shutdown)
        return server

    class _GaugeChild:
        def __init__(self, wrapper: _ValueWrapper) -> None:
            self._wrapper = wrapper

        def set(self, value: float) -> None:
            self._wrapper.set(value)

        def inc(self, amount: float = 1.0) -> None:
            self._wrapper.add(amount)

        def dec(self, amount: float = 1.0) -> None:
            self._wrapper.add(-amount)

        def get(self) -> float:
            return self._wrapper.get()

    class Gauge:
        """Minimal Gauge fallback exposing Prometheus text format."""

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: Sequence[str] | None = None,
            *,
            registry: Registry | None = None,
        ) -> None:
            self.name = name
            self.documentation = documentation
            self._labelnames: Tuple[str, ...] = tuple(labelnames or [])
            self._values: Dict[Tuple[str, ...], _ValueWrapper] = {}
            self._lock = threading.Lock()
            (registry or _DEFAULT_REGISTRY).register(self)

        def _get_child(self, key: Tuple[str, ...]) -> _GaugeChild:
            with self._lock:
                if key not in self._values:
                    self._values[key] = _ValueWrapper()
                wrapper = self._values[key]
            return _GaugeChild(wrapper)

        def labels(self, *labelvalues: str, **labelkwargs: str) -> _GaugeChild:
            if labelkwargs:
                if labelvalues:
                    raise ValueError("cannot mix label methods")
                labelvalues = tuple(labelkwargs[name] for name in self._labelnames)
            if len(labelvalues) != len(self._labelnames):
                raise ValueError("incorrect label count")
            key = tuple(str(v) for v in labelvalues)
            return self._get_child(key)

        def set(self, value: float) -> None:
            self.labels().set(value)

        def inc(self, amount: float = 1.0) -> None:
            self.labels().inc(amount)

        def dec(self, amount: float = 1.0) -> None:
            self.labels().dec(amount)

        def _prometheus(self) -> str:
            lines = [
                f"# HELP {self.name} {self.documentation}",
                f"# TYPE {self.name} gauge",
            ]
            with self._lock:
                items = list(self._values.items())
            for key, wrapper in items:
                value = wrapper.get()
                if self._labelnames:
                    labels = ",".join(
                        f"{n}=\"{v}\"" for n, v in zip(self._labelnames, key)
                    )
                    lines.append(f"{self.name}{{{labels}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
            return "\n".join(lines)

learning_cv_score: Optional[Gauge]
learning_holdout_score: Optional[Gauge]
evolution_cycle_count: Optional[Gauge]
experiment_best_roi: Optional[Gauge]
learning_cv_score = Gauge(
    "learning_cv_score", "Cross validation score of LearningEngine"
)
learning_holdout_score = Gauge(
    "learning_holdout_score", "Holdout test score of LearningEngine"
)
evolution_cycle_count = Gauge(
    "evolution_cycle_count", "Total evolution cycles executed"
)
experiment_best_roi = Gauge(
    "experiment_best_roi", "ROI of current best experiment variant"
)

# Gauge tracking the latest ROI forecast value
roi_forecast_gauge = Gauge(
    "roi_forecast",
    "Latest ROI forecast value",
)

# Gauge tracking the latest synergy ROI forecast value
synergy_forecast_gauge = Gauge(
    "synergy_forecast",
    "Latest synergy ROI forecast value",
)

# Gauges tracking adaptive thresholds
roi_threshold_gauge = Gauge(
    "roi_threshold",
    "Current ROI threshold for diminishing returns",
)
synergy_threshold_gauge = Gauge(
    "synergy_threshold",
    "Current synergy convergence threshold",
)

# Gauges for prediction accuracy
prediction_error = Gauge(
    "prediction_error",
    "Latest absolute prediction error",
    ["metric"],
)
prediction_mae = Gauge(
    "prediction_mae",
    "Rolling mean absolute error for predictions",
    ["metric"],
)

prediction_reliability = Gauge(
    "prediction_reliability",
    "Rolling reliability score for predictions",
    ["metric"],
)

# New gauges for extended metrics
security_score_gauge = Gauge(
    "security_score", "Overall security score of the system"
)
safety_rating_gauge = Gauge(
    "safety_rating", "Overall safety rating of the system"
)
adaptability_gauge = Gauge(
    "adaptability", "Adaptability score of the system"
)
antifragility_gauge = Gauge(
    "antifragility", "Antifragility metric"
)
shannon_entropy_gauge = Gauge(
    "shannon_entropy", "Shannon entropy of outputs"
)
efficiency_gauge = Gauge(
    "efficiency_metric", "Overall efficiency metric"
)
flexibility_gauge = Gauge(
    "flexibility_metric", "Overall flexibility metric"
)
projected_lucrativity_gauge = Gauge(
    "projected_lucrativity", "Projected lucrativity"
)


# Gauges for vector store and retrieval metrics
embedding_tokens_total = Gauge(
    "embedding_tokens_total", "Total tokens processed during embedding",
)
embedding_wall_time_seconds = Gauge(
    "embedding_wall_time_seconds", "Wall time spent generating embeddings",
)
vector_store_latency_seconds = Gauge(
    "vector_store_latency_seconds", "Latency of vector store operations",
)
embedding_stale_cost_seconds = Gauge(
    "embedding_stale_cost_seconds", "Age in seconds of retrieved embedding", ["origin_db"],
)
retrieval_hits_total = Gauge(
    "retrieval_hits_total", "Number of retrieval hits",
)
retrieval_win_rate = Gauge(
    "retrieval_win_rate", "Win rate of retrieval operations",
)
patch_success_total = Gauge(
    "patch_success_total", "Number of successful patch applications",
)
patch_failure_total = Gauge(
    "patch_failure_total", "Number of failed patch applications",
)


# Metrics for visual agent utilisation
visual_agent_wait_time = Gauge(
    "visual_agent_wait_time_seconds",
    "Average wait time for VisualAgentClient tasks",
)
visual_agent_queue_depth = Gauge(
    "visual_agent_queue_depth", "Queue length reported by VisualAgentClient"
)
visual_agent_recoveries_total = Gauge(
    "visual_agent_recoveries_total",
    "Total number of visual agent queue recoveries",
)
visual_agent_watchdog_recoveries_total = Gauge(
    "visual_agent_watchdog_recoveries_total",
    "Recoveries triggered by the internal queue watchdog",
)

# Gauges for workflow benchmarking
workflow_duration_gauge = Gauge(
    "workflow_duration_seconds",
    "Execution time of a workflow run",
    labelnames=["workflow"],
)
workflow_cpu_percent_gauge = Gauge(
    "workflow_cpu_percent",
    "Average CPU utilisation during the run",
    labelnames=["workflow"],
)
workflow_memory_gauge = Gauge(
    "workflow_memory_mb",
    "Resident memory usage after execution",
    labelnames=["workflow"],
)
workflow_memory_usage_gauge = Gauge(
    "workflow_memory_usage_mb",
    "Resident memory at end of workflow",
    labelnames=["workflow"],
)
workflow_cpu_time_gauge = Gauge(
    "workflow_cpu_time_seconds",
    "User+system CPU time consumed by the workflow",
    labelnames=["workflow"],
)
workflow_cpu_user_time_gauge = Gauge(
    "workflow_cpu_user_time_seconds",
    "User CPU time consumed by the workflow",
    labelnames=["workflow"],
)
workflow_cpu_system_time_gauge = Gauge(
    "workflow_cpu_system_time_seconds",
    "System CPU time consumed by the workflow",
    labelnames=["workflow"],
)
workflow_net_io_gauge = Gauge(
    "workflow_network_bytes",
    "Network I/O during the run",
    labelnames=["workflow"],
)
workflow_disk_read_gauge = Gauge(
    "workflow_disk_read_bytes",
    "Bytes read from disk",
    labelnames=["workflow"],
)
workflow_disk_write_gauge = Gauge(
    "workflow_disk_write_bytes",
    "Bytes written to disk",
    labelnames=["workflow"],
)
workflow_latency_p95_gauge = Gauge(
    "workflow_latency_p95_seconds",
    "95th percentile latency for recent runs",
    labelnames=["workflow"],
)
workflow_latency_median_gauge = Gauge(
    "workflow_latency_median_seconds",
    "Median workflow latency",
    labelnames=["workflow"],
)
workflow_latency_min_gauge = Gauge(
    "workflow_latency_min_seconds",
    "Minimum workflow latency",
    labelnames=["workflow"],
)
workflow_latency_max_gauge = Gauge(
    "workflow_latency_max_seconds",
    "Maximum workflow latency",
    labelnames=["workflow"],
)
workflow_peak_memory_gauge = Gauge(
    "workflow_peak_memory_mb",
    "Approximate peak resident memory during execution",
    labelnames=["workflow"],
)

# Additional counters for failure visibility
error_bot_exceptions = Gauge(
    "error_bot_exceptions_total", "Exceptions encountered in ErrorBot"
)
learning_engine_exceptions = Gauge(
    "learning_engine_exceptions_total",
    "Exceptions encountered in UnifiedLearningEngine"
)
synergy_weight_updates_total = Gauge(
    "synergy_weight_updates_total", "Total number of successful synergy weight updates"
)
synergy_weight_update_failures_total = Gauge(
    "synergy_weight_update_failures_total", "Total number of failed synergy weight update attempts"
)
synergy_weight_update_alerts_total = Gauge(
    "synergy_weight_update_alerts_total",
    "Total number of synergy weight update failure alerts dispatched",
)

# Counters tracking forecast failures
roi_forecast_failures_total = Gauge(
    "roi_forecast_failures_total",
    "Total ROI forecast exceptions encountered",
)
synergy_forecast_failures_total = Gauge(
    "synergy_forecast_failures_total",
    "Total synergy forecast exceptions encountered",
)

# Counter tracking preset adaptation decisions
synergy_adaptation_actions_total = Gauge(
    "synergy_adaptation_actions_total",
    "Total preset adaptation decisions",
    labelnames=["action"],
)
orphan_modules_integrated_total = Gauge(
    "orphan_modules_integrated_total",
    "Total number of orphan modules upgraded from orphan to integrated status",
)
# Backwards compatibility aliases
orphan_modules_reintroduced_total = orphan_modules_integrated_total
orphan_modules_passed_total = orphan_modules_integrated_total
orphan_modules_tested_total = Gauge(
    "orphan_modules_tested_total",
    "Total number of orphan modules tested",
)
orphan_modules_failed_total = Gauge(
    "orphan_modules_failed_total",
    "Total number of orphan modules failing tests",
)
orphan_modules_reclassified_total = Gauge(
    "orphan_modules_reclassified_total",
    "Total number of orphan modules reclassified as candidates after tests",
)
orphan_modules_redundant_total = Gauge(
    "orphan_modules_redundant_total",
    "Total number of orphan modules marked redundant",
)
orphan_modules_legacy_total = Gauge(
    "orphan_modules_legacy_total",
    "Total number of orphan modules marked legacy",
)
orphan_modules_side_effects_total = Gauge(
    "orphan_modules_side_effects_total",
    "Total number of orphan modules skipped due to side effect scores",
)
isolated_modules_discovered_total = Gauge(
    "isolated_modules_discovered_total",
    "Total number of isolated modules discovered for potential inclusion",
)
isolated_modules_integrated_total = Gauge(
    "isolated_modules_integrated_total",
    "Total number of isolated modules successfully integrated",
)
sandbox_restart_total = Gauge(
    "sandbox_restart_total", "Total number of sandbox restarts"
)
sandbox_last_failure_ts = Gauge(
    "sandbox_last_failure_ts", "Timestamp of last sandbox failure"
)

# Gauges for sandbox cleanup statistics
cleanup_idle = Gauge(
    "cleanup_idle", "Containers removed after exceeding the idle timeout"
)
cleanup_unhealthy = Gauge(
    "cleanup_unhealthy", "Containers removed due to failed health checks"
)
cleanup_lifetime = Gauge(
    "cleanup_lifetime", "Containers purged for exceeding the maximum lifetime"
)
cleanup_disk = Gauge(
    "cleanup_disk", "Containers removed for exceeding disk usage limits"
)
stale_containers_removed = Gauge(
    "stale_containers_removed", "Containers removed by purge_leftovers or cleanup"
)
stale_vms_removed = Gauge(
    "stale_vms_removed", "VM overlay directories removed during cleanup"
)
cleanup_failures = Gauge(
    "cleanup_failures", "Failed attempts to stop or remove containers"
)
force_kills = Gauge(
    "force_kills", "Containers forcefully terminated via CLI fallback"
)
runtime_vms_removed = Gauge(
    "runtime_vms_removed", "VM overlays deleted while the sandbox is running"
)

# Gauge tracking container creation skips due to the active limit
active_container_limit_reached = Gauge(
    "active_container_limit_reached",
    "Container creations skipped because the active limit was reached",
)

# Gauges tracking container creation attempts
container_creation_failures_total = Gauge(
    "container_creation_failures_total",
    "Total container creation failures by image",
    labelnames=["image"],
)
container_creation_success_total = Gauge(
    "container_creation_success_total",
    "Total successful container creations by image",
    labelnames=["image"],
)
container_creation_alerts_total = Gauge(
    "container_creation_alerts_total",
    "Total container creation alerts dispatched by image",
    labelnames=["image"],
)
container_creation_seconds = Gauge(
    "container_creation_seconds",
    "Duration of the last container creation attempt by image",
    labelnames=["image"],
)

# Gauges tracking current resource usage
active_containers = Gauge(
    "active_containers",
    "Number of active sandbox containers",
)
active_overlays = Gauge(
    "active_overlays",
    "Number of active overlay directories",
)

# Gauge tracking overlay cleanup failures
overlay_cleanup_failures = Gauge(
    "overlay_cleanup_failures",
    "Failed attempts to delete VM overlay directories",
)

# Gauges tracking retry attempts for failed cleanup items
cleanup_retry_successes = Gauge(
    "cleanup_retry_successes",
    "Failed cleanup entries successfully retried",
)
cleanup_retry_failures = Gauge(
    "cleanup_retry_failures",
    "Failed cleanup retry attempts that did not succeed",
)

# Gauge tracking duration of cleanup workers
cleanup_duration_gauge = Gauge(
    "cleanup_duration_seconds",
    "Time spent in cleanup worker iterations",
    labelnames=["worker"],
)

# Gauge tracking age of last automatic purge
hours_since_autopurge = Gauge(
    "hours_since_autopurge",
    "Hours since purge_leftovers last ran automatically",
)


_METRICS_SERVER: "HTTPServer" | None = None


def stop_metrics_server() -> None:
    """Stop the fallback metrics HTTP server if it is running."""
    global _METRICS_SERVER
    if _METRICS_SERVER:
        try:
            _METRICS_SERVER.shutdown()
            _METRICS_SERVER.server_close()
        finally:
            _METRICS_SERVER = None


def start_metrics_server(port: int = 8001, *, registry: Registry | None = None) -> None:
    """Start an HTTP metrics server.

    When ``prometheus_client`` is installed the official server is started.
    Otherwise a very small text server is launched that exposes the registered
    metrics in Prometheus format.  The server is shut down gracefully on
    process exit.
    """
    logger = logging.getLogger(__name__)
    if _USING_STUB:
        logger.warning(
            "starting fallback metrics server; advanced features disabled"
        )
    try:
        kwargs = {"registry": registry} if registry is not None else {}
        start_http_server(port, **kwargs)
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.exception("failed to start metrics server: %s", exc)
        raise

__all__ = [
    "CollectorRegistry",
    "start_metrics_server",
    "stop_metrics_server",
    "learning_cv_score",
    "learning_holdout_score",
    "evolution_cycle_count",
    "experiment_best_roi",
    "roi_forecast_gauge",
    "synergy_forecast_gauge",
    "roi_threshold_gauge",
    "synergy_threshold_gauge",
    "prediction_error",
    "prediction_mae",
    "prediction_reliability",
    "error_bot_exceptions",
    "learning_engine_exceptions",
    "synergy_weight_updates_total",
    "synergy_weight_update_failures_total",
    "synergy_weight_update_alerts_total",
    "roi_forecast_failures_total",
    "synergy_forecast_failures_total",
    "synergy_adaptation_actions_total",
    "security_score_gauge",
    "safety_rating_gauge",
    "adaptability_gauge",
    "antifragility_gauge",
    "shannon_entropy_gauge",
    "efficiency_gauge",
    "flexibility_gauge",
    "projected_lucrativity_gauge",
    "visual_agent_wait_time",
    "visual_agent_queue_depth",
    "visual_agent_recoveries_total",
    "visual_agent_watchdog_recoveries_total",
    "isolated_modules_discovered_total",
    "isolated_modules_integrated_total",
    "sandbox_restart_total",
    "sandbox_last_failure_ts",
    "cleanup_idle",
    "cleanup_unhealthy",
    "cleanup_lifetime",
    "cleanup_disk",
    "stale_containers_removed",
    "stale_vms_removed",
    "cleanup_failures",
    "force_kills",
    "runtime_vms_removed",
    "overlay_cleanup_failures",
    "active_container_limit_reached",
    "active_containers",
    "active_overlays",
    "container_creation_failures_total",
    "container_creation_success_total",
    "container_creation_alerts_total",
    "container_creation_seconds",
    "cleanup_retry_successes",
    "cleanup_retry_failures",
    "hours_since_autopurge",
    "cleanup_duration_gauge",
    "workflow_duration_gauge",
    "workflow_cpu_percent_gauge",
    "workflow_memory_gauge",
    "workflow_memory_usage_gauge",
    "workflow_cpu_time_gauge",
    "workflow_cpu_user_time_gauge",
    "workflow_cpu_system_time_gauge",
    "workflow_net_io_gauge",
    "workflow_disk_read_gauge",
    "workflow_disk_write_gauge",
    "workflow_latency_p95_gauge",
    "workflow_latency_median_gauge",
    "workflow_latency_min_gauge",
    "workflow_latency_max_gauge",
    "workflow_peak_memory_gauge",
]

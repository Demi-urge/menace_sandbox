from __future__ import annotations

"""Restart sandbox runs when unexpected failures occur."""

try:  # pragma: no cover - lightweight bootstrap when run as a script
    from import_compat import bootstrap as _bootstrap
except Exception:  # pragma: no cover - import compatibility helper unavailable
    from pathlib import Path
    import sys

    _bootstrap = None  # type: ignore
    _here = Path(__file__).resolve()
    for _candidate in (_here.parent, *_here.parents):
        compat_path = _candidate / "import_compat.py"
        if compat_path.exists():
            candidate_str = str(_candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                from import_compat import bootstrap as _bootstrap  # type: ignore
            except Exception:
                _bootstrap = None  # type: ignore
            else:
                break

if "_bootstrap" in globals() and _bootstrap is not None:  # pragma: no cover - script usage
    _bootstrap(__name__, __file__)

from typing import Any, Callable, Dict, List
import inspect
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

try:
    from .logging_utils import set_correlation_id
except ImportError as exc:  # pragma: no cover - module not a package
    logging.getLogger(__name__).debug(
        "fallback to top-level logging_utils due to import error", exc_info=exc
    )
    from logging_utils import set_correlation_id  # type: ignore

try:
    from .resilience import (
        CircuitBreaker,
        CircuitOpenError,
        ResilienceError,
    )
except ImportError as exc:  # pragma: no cover - module not a package
    logging.getLogger(__name__).debug(
        "fallback to top-level resilience due to import error", exc_info=exc
    )
    from resilience import (
        CircuitBreaker,  # type: ignore
        CircuitOpenError,  # type: ignore
        ResilienceError,  # type: ignore
    )

try:
    from .metrics_exporter import CollectorRegistry, Gauge
except ImportError as exc:  # pragma: no cover - module may not be a package
    logging.getLogger(__name__).debug(
        "fallback to top-level metrics_exporter due to import error", exc_info=exc
    )
    from metrics_exporter import CollectorRegistry, Gauge  # type: ignore

try:
    from .sandbox_settings import SandboxSettings
except ImportError as exc:  # pragma: no cover - module may not be a package
    logging.getLogger(__name__).debug(
        "fallback to top-level sandbox_settings due to import error", exc_info=exc
    )
    from sandbox_settings import SandboxSettings  # type: ignore

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

recovery_failure_total = Gauge(
    "sandbox_recovery_failure_total",
    "Total number of sandbox recovery failures by severity",
    labelnames=["severity"],
)

logger = logging.getLogger(__name__)


class SandboxRecoveryError(ResilienceError):
    """Raised when sandbox recovery cannot proceed."""


class RecoveryMetricsRecorder:
    """Record metrics via :mod:`metrics_exporter` or a local file."""

    def __init__(self) -> None:
        try:
            try:
                from . import metrics_exporter as _me
            except ImportError as exc:  # pragma: no cover - package not available
                logger.debug("metrics_exporter relative import failed", exc_info=exc)
                import metrics_exporter as _me  # type: ignore

            self._using_exporter = not getattr(_me, "_USING_STUB", False)
            if self._using_exporter:
                self._restart_gauge = _me.sandbox_restart_total
                self._failure_gauge = _me.sandbox_last_failure_ts
            else:
                self._restart_gauge = None
                self._failure_gauge = None
        except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency missing
            logger.debug("metrics exporter unavailable; falling back to file", exc_info=exc)
            self._using_exporter = False
            self._restart_gauge = None
            self._failure_gauge = None

    def record(
        self,
        restart_count: int,
        last_failure_ts: float | None,
        data_dir: Path,
        *,
        service_name: str,
        reason: str,
    ) -> None:
        ts = float(last_failure_ts) if last_failure_ts is not None else 0.0
        if self._using_exporter and self._restart_gauge and self._failure_gauge:
            try:
                self._restart_gauge.labels(service=service_name, reason=reason).set(
                    float(restart_count)
                )
                self._failure_gauge.set(ts)
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - runtime issues
                logger.exception("failed to update metrics", exc_info=exc)
            return

        payload = {
            "sandbox_restart_total": float(restart_count),
            "sandbox_last_failure_ts": ts,
            "service": service_name,
            "reason": reason,
        }
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_dir / "recovery.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError as exc:  # pragma: no cover - runtime issues
            logger.exception("failed to write recovery metrics", exc_info=exc)


class SandboxRecoveryManager:
    """Run ``sandbox_main`` with retry and circuit breaker handling.

    The manager retries failed runs with exponential backoff, records
    recoverable and fatal failure metrics and halts execution when the
    underlying :class:`~resilience.CircuitBreaker` opens.
    """

    def __init__(
        self,
        sandbox_main: Callable[..., Any],
        *,
        retry_delay: float | None = None,
        max_retries: int | None = None,
        on_retry: Callable[[Exception, float], None] | None = None,
        registry: "CollectorRegistry" | None = None,
        circuit_max_failures: int | None = None,
        circuit_reset_timeout: float | None = None,
        settings: SandboxSettings | None = None,
        service_name: str | None = None,
    ) -> None:
        self.sandbox_main = sandbox_main
        self._supports_builder, self._requires_builder = self._inspect_sandbox_main(sandbox_main)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.on_retry = on_retry
        self.restart_count = 0
        self.last_failure_time: float | None = None
        self.settings = settings or SandboxSettings()
        self.retry_delay = (
            retry_delay
            if retry_delay is not None
            else self.settings.sandbox_retry_delay
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else self.settings.sandbox_max_retries
        )
        circuit_max_failures = (
            circuit_max_failures
            if circuit_max_failures is not None
            else self.settings.sandbox_circuit_max_failures
        )
        circuit_reset_timeout = (
            circuit_reset_timeout
            if circuit_reset_timeout is not None
            else self.settings.sandbox_circuit_reset_timeout
        )
        self.service_name = service_name or sandbox_main.__name__
        self._circuit = CircuitBreaker(
            max_failures=circuit_max_failures, reset_timeout=circuit_reset_timeout
        )

        self._metrics_recorder = RecoveryMetricsRecorder()

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> Dict[str, float | None]:
        """Return restart count and last failure time."""
        return {
            "sandbox_restart_total": float(self.restart_count),
            "sandbox_last_failure_ts": self.last_failure_time,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def load_last_tracker(data_dir: str | Path):
        """Return :class:`ROITracker` loaded from ``data_dir`` or ``None``."""
        try:
            from menace.roi_tracker import ROITracker
        except ImportError as exc:  # pragma: no cover - fallback
            logger.debug("roi_tracker relative import failed", exc_info=exc)
            from roi_tracker import ROITracker  # type: ignore

        path = Path(data_dir) / "roi_history.json"
        tracker = ROITracker()
        try:
            tracker.load_history(str(path))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception(
                "failed to load tracker history", extra={"path": str(path)}, exc_info=exc
            )
            return None
        return tracker

    # ------------------------------------------------------------------
    @staticmethod
    def _inspect_sandbox_main(func: Callable[..., Any]) -> tuple[bool, bool]:
        """Return ``(supports_builder, requires_builder)`` for ``func``."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True, False

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True, False

        positional = [
            p
            for p in params
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if len(positional) < 3:
            return False, False

        third = positional[2]
        requires = third.default is inspect._empty
        return True, requires

    # ------------------------------------------------------------------
    def run(
        self,
        preset: Dict[str, Any],
        args: argparse.Namespace,
        builder: object | None = None,
    ):
        """Execute ``sandbox_main`` with retry and circuit-breaker support.

        Recoverable failures trigger exponential backoff retries while fatal
        errors—such as an open circuit or exceeding ``max_retries``—raise
        :class:`SandboxRecoveryError`.
        """
        data_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
        attempts = 0
        delay = self.retry_delay
        while True:
            cid = uuid.uuid4().hex
            set_correlation_id(cid)
            start = time.monotonic()
            try:
                call: Callable[[], Any]
                if builder is not None and self._supports_builder:
                    call = lambda: self.sandbox_main(preset, args, builder)
                else:
                    if builder is None or not self._requires_builder:
                        call = lambda: self.sandbox_main(preset, args)
                    else:
                        raise TypeError(
                            "sandbox_main expects a context builder but none was provided"
                        )
                return self._circuit.call(call)
            except CircuitOpenError as exc:
                self.logger.error("recovery circuit open", exc_info=exc)
                recovery_failure_total.labels(severity="fatal").inc()
                raise SandboxRecoveryError("circuit open") from exc
            except ResilienceError as exc:
                self.logger.error("sandbox resilience error", exc_info=exc)
                recovery_failure_total.labels(severity="fatal").inc()
                raise SandboxRecoveryError("resilience failure") from exc
            except Exception as exc:  # pragma: no cover - rare
                attempts += 1
                self.restart_count += 1
                self.last_failure_time = time.time()
                runtime = time.monotonic() - start
                recovery_failure_total.labels(severity="recoverable").inc()
                self.logger.exception(
                    "sandbox run crashed; restarting",
                    extra={"attempt": attempts, "cid": cid, "runtime": runtime},
                )

                log_dir = Path(
                    getattr(args, "sandbox_data_dir", None) or data_dir
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "recovery.log"
                tb = traceback.format_exc()
                with open(log_file, "a", encoding="utf-8") as fh:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    fh.write(
                        f"{ts} cid={cid} attempt={attempts} runtime={runtime:.2f}s\n{tb}\n"
                    )

                if self.on_retry:
                    try:
                        self.on_retry(exc, runtime)
                    except Exception as cb_exc:
                        self.logger.exception("on_retry callback failed", exc_info=cb_exc)
                self._metrics_recorder.record(
                    self.restart_count,
                    self.last_failure_time,
                    log_dir,
                    service_name=self.service_name,
                    reason=exc.__class__.__name__,
                )

                if self.max_retries is not None and attempts >= self.max_retries:
                    recovery_failure_total.labels(severity="fatal").inc()
                    raise SandboxRecoveryError("maximum retries reached") from exc
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            finally:
                set_correlation_id(None)


__all__ = [
    "SandboxRecoveryError",
    "SandboxRecoveryManager",
    "cli",
    "load_metrics",
    "load_last_tracker",
]


def load_metrics(path: Path) -> Dict[str, float]:
    """Return metrics stored in ``path`` as ``float`` values."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "failed to load recovery metrics", extra={"path": str(path)}, exc_info=exc
        )
        return {}
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        mapping = {
            "restart_count": "sandbox_restart_total",
            "last_failure_time": "sandbox_last_failure_ts",
        }
        for k, v in data.items():
            name = mapping.get(str(k), str(k))
            try:
                out[name] = float(v)
            except (TypeError, ValueError):
                out[name] = 0.0
    return out


def cli(argv: List[str] | None = None) -> int:
    """Print sandbox recovery metrics."""
    parser = argparse.ArgumentParser(description=cli.__doc__)
    parser.add_argument(
        "--file",
        default=str(resolve_path("sandbox_data") / "recovery.json"),
        help="Path to recovery.json",
    )
    args = parser.parse_args(argv)

    try:
        data = load_metrics(Path(args.file))
    except OSError as exc:  # pragma: no cover - runtime issues
        logger.error("failed to read %s", args.file, exc_info=exc)
        return 1

    for k, v in data.items():
        logger.info("%s: %s", k, v)
    return 0


def main(argv: List[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

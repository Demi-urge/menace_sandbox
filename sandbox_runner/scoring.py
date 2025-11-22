from __future__ import annotations
"""Aggregate sandbox run metrics and persist them.

The :func:`record_run` helper collects runtime information such as success
status, runtime, entropy deltas and error traces. Each invocation is appended
as a JSON line under ``sandbox_data`` and a cumulative summary is maintained for
scorecard generation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import threading

from .logging_utils import get_logger, log_record
from sandbox_results_logger import record_run as _db_record_run

try:  # pragma: no cover - optional dependency during tests
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore


_LOG_DIR = Path(resolve_path("sandbox_data"))
_RUN_LOG = _LOG_DIR / "run_metrics.jsonl"
_SUMMARY_FILE = _LOG_DIR / "run_summary.json"
_lock = threading.Lock()

logger = get_logger(__name__)


def _serialise(obj: Any) -> Any:
    """Return a JSON serialisable representation of *obj*."""

    try:
        json.dumps(obj)
        return obj
    except Exception:  # pragma: no cover - defensive
        return str(obj)


def record_run(result: Any, metrics: Dict[str, Any]) -> None:
    """Record a sandbox run *result* with additional *metrics*.

    Parameters
    ----------
    result:
        Object describing the run outcome. It may be a boolean or an object
        exposing ``success``/``duration``/``failure`` attributes.
    metrics:
        Mapping containing optional ``roi``, ``coverage`` (mapping of files to
        executed function names), ``executed_functions`` (flattened
        ``"file:function"`` entries), ``entropy_delta`` or ``runtime``
        overrides.
    """

    success = bool(getattr(result, "success", result))
    runtime = float(metrics.get("runtime", getattr(result, "duration", 0.0)))
    entropy_delta = metrics.get("entropy_delta")
    roi = metrics.get("roi")
    coverage = metrics.get("coverage")
    executed_functions = metrics.get("executed_functions")
    if executed_functions is None:
        if isinstance(coverage, dict) and "executed_functions" in coverage:
            executed_functions = coverage.get("executed_functions")
        elif isinstance(coverage, dict):
            executed_functions = [
                f"{path}:{fn}"
                for path, funcs in coverage.items()
                for fn in funcs
            ]
        else:
            cov_attr = getattr(result, "coverage", None)
            if isinstance(cov_attr, dict):
                executed_functions = cov_attr.get("executed_functions")
    functions_hit = len(executed_functions) if executed_functions is not None else None

    failure = getattr(result, "failure", None)
    error_trace: str | None = None
    if failure:
        if isinstance(failure, dict):
            error_trace = failure.get("trace") or json.dumps(_serialise(failure))
        else:
            error_trace = str(failure)
    else:
        stderr = getattr(result, "stderr", None)
        if stderr:
            error_trace = str(stderr)

    record = {
        "ts": datetime.utcnow().isoformat(),
        "success": success,
        "runtime": runtime,
        "entropy_delta": entropy_delta,
        "roi": roi,
        "coverage": _serialise(coverage) if coverage is not None else None,
        "functions_hit": functions_hit,
        "executed_functions": _serialise(executed_functions)
        if executed_functions is not None
        else None,
        "error": error_trace,
    }
    logger.info("run", extra=log_record(**record))

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        try:
            with _RUN_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            summary = {}
            if _SUMMARY_FILE.exists():
                try:
                    summary = json.loads(_SUMMARY_FILE.read_text())
                except Exception:
                    summary = {}
            summary["runs"] = summary.get("runs", 0) + 1
            summary["successes"] = summary.get("successes", 0) + (1 if success else 0)
            summary["failures"] = summary.get("failures", 0) + (0 if success else 1)
            summary["runtime_total"] = summary.get("runtime_total", 0.0) + runtime
            if entropy_delta is not None:
                summary["entropy_total"] = summary.get("entropy_total", 0.0) + float(entropy_delta)
            if functions_hit is not None:
                summary["functions_hit_total"] = summary.get(
                    "functions_hit_total", 0
                ) + int(functions_hit)
            _SUMMARY_FILE.write_text(json.dumps(summary))

            try:  # propagate to SQLite logger
                _db_record_run(record)
            except Exception:  # pragma: no cover - don't fail caller
                logger.exception("failed to forward run metrics to legacy logger")
        except Exception:  # pragma: no cover - logging is best effort
            logger.exception("failed to persist run metrics")


def load_summary() -> Dict[str, Any]:
    """Return cumulative run information if available."""

    if _SUMMARY_FILE.exists():
        try:
            return json.loads(_SUMMARY_FILE.read_text())
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load run summary")
    return {}


__all__ = ["record_run", "load_summary"]

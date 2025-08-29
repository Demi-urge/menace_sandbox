"""Tests for the internal sandbox error logger fallback."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from metrics_exporter import environment_failure_total


def _metric_value(child) -> float:
    """Return the numeric value from a Gauge child."""

    try:
        return child.get()  # prometheus_client stub
    except AttributeError:  # pragma: no cover - real prometheus client
        return child._value.get()  # type: ignore[attr-defined]


def test_missing_error_logger_records_errors(monkeypatch, tmp_path):
    """Errors are persisted and metrics updated when error_logger is absent."""

    fake = types.ModuleType("error_logger")
    monkeypatch.setitem(sys.modules, "error_logger", fake)

    log_path = tmp_path / "errors.log"
    monkeypatch.setenv("SANDBOX_ERROR_LOG", str(log_path))

    package_path = Path(__file__).resolve().parents[1] / "sandbox_runner"
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment", package_path / "environment.py"
    )
    env = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env)
    assert spec.loader is not None
    spec.loader.exec_module(env)

    child = environment_failure_total.labels(reason="semantic_bug")
    child.set(0)

    env.record_error(ValueError("boom"))

    assert log_path.exists()
    assert "boom" in log_path.read_text()
    assert env.ERROR_CATEGORY_COUNTS["semantic_bug"] == 1
    assert _metric_value(child) == 1


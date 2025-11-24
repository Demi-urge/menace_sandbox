import sqlite3
import time
from types import SimpleNamespace

import pytest

from sandbox_runner import bootstrap


def _stub_settings(tmp_path):
    return SimpleNamespace(
        menace_env_file=tmp_path / ".env",
        sandbox_data_dir=tmp_path,
        alignment_baseline_metrics_path="",
        sandbox_required_db_files=("blocked.db",),
        optional_service_versions={},
        optional_modules=(),
        required_optional_modules=(),
        required_env_vars=(),
        required_system_tools=(),
        required_python_packages=(),
        optional_python_packages=(),
        auto_install_dependencies=False,
    )


def test_ensure_sqlite_db_times_out(monkeypatch, tmp_path):
    recorded = {}

    def _connect(path, timeout):
        recorded["timeout"] = timeout
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(bootstrap.sqlite3, "connect", _connect)

    with pytest.raises(RuntimeError) as excinfo:
        bootstrap._ensure_sqlite_db(tmp_path / "blocked.db")

    assert "blocked.db" in str(excinfo.value)
    assert recorded["timeout"] == bootstrap._DB_INIT_TIMEOUT_SECONDS


def test_initialize_sandbox_fails_fast_on_db_error(monkeypatch, tmp_path):
    settings = _stub_settings(tmp_path)
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_LAST_ERROR", None)
    monkeypatch.setattr(bootstrap, "auto_configure_env", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_optional_modules", lambda *_a, **_k: set())
    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)

    def _fail_fast(path):
        raise RuntimeError("database initialisation watchdog expired")

    monkeypatch.setattr(bootstrap, "_ensure_sqlite_db", _fail_fast)

    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        bootstrap.initialize_autonomous_sandbox(
            settings, start_services=False, start_self_improvement=False
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 30

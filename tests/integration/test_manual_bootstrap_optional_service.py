from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


class _DummyThread:
    """Lightweight stand-in for the self-improvement background worker."""

    def __init__(self) -> None:
        self._alive = False
        self._thread = self

    def start(self) -> None:
        self._alive = True

    def join(self, timeout: float | None = None) -> None:
        if timeout is None or timeout > 0:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive


def test_manual_bootstrap_handles_missing_quick_fix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Manual bootstrap should tolerate optional service dependency failures."""

    sys.modules.pop("dynamic_path_router", None)
    import dynamic_path_router  # noqa: F401
    from tests import test_bootstrap_service_deps as bootstrap_helpers  # noqa: F401

    bootstrap = bootstrap_helpers.bootstrap

    class _StubDefaultConfigManager:
        def __init__(self, *_a, **_k) -> None:
            pass

        def apply_defaults(self) -> None:
            return None

    auto_env_stub = types.SimpleNamespace(ensure_env=lambda *_: None)
    monkeypatch.setattr(bootstrap, "ensure_env", auto_env_stub.ensure_env)
    monkeypatch.setattr(bootstrap, "DefaultConfigManager", _StubDefaultConfigManager)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)

    api_stub = types.SimpleNamespace(
        init_self_improvement=lambda settings: None,
        start_self_improvement_cycle=lambda warmups: _DummyThread(),
        stop_self_improvement_cycle=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "self_improvement.api", api_stub)
    monkeypatch.setitem(sys.modules, "self_improvement", types.SimpleNamespace(api=api_stub))

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    env_file = tmp_path / "sandbox.env"

    class FakeSettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(data_dir)
            self.sandbox_repo_path = str(tmp_path)
            self.alignment_baseline_metrics_path = ""
            self.sandbox_required_db_files: list[str] = []
            self.optional_service_versions = {
                "relevancy_radar": "1.0.0",
                "quick_fix_engine": "1.0.0",
            }
            self.menace_env_file = str(env_file)
            self.required_env_vars = ["OPENAI_API_KEY", "DATABASE_URL", "MODELS"]
            self.menace_mode = "test"

    fake_settings = FakeSettings()
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: fake_settings)
    monkeypatch.setattr(bootstrap, "SandboxSettings", FakeSettings)

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("MODELS", str(models_dir))

    relevancy_mod = types.ModuleType("relevancy_radar")
    relevancy_mod.__version__ = "1.0.0"
    monkeypatch.setitem(sys.modules, "relevancy_radar", relevancy_mod)
    monkeypatch.setitem(sys.modules, f"{bootstrap._REPO_PACKAGE}.relevancy_radar", relevancy_mod)

    original_import = bootstrap.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in {
            "quick_fix_engine",
            f"{bootstrap._REPO_PACKAGE}.quick_fix_engine",
        }:
            raise ImportError("Self-coding engine is required for operation")
        return original_import(name, package)

    monkeypatch.setattr(bootstrap.importlib, "import_module", fake_import)

    bootstrap._INITIALISED = False
    bootstrap._OPTIONAL_MODULE_CACHE.clear()
    bootstrap._MISSING_OPTIONAL.clear()
    bootstrap._OPTIONAL_DEPENDENCY_WARNED.clear()

    caplog.set_level("WARNING")
    try:
        bootstrap.initialize_autonomous_sandbox(fake_settings)
        manual_bootstrap = importlib.import_module("manual_bootstrap")
        exit_code = manual_bootstrap.main(["--skip-sandbox", "--skip-environment"])
    finally:
        bootstrap.shutdown_autonomous_sandbox()
        bootstrap._OPTIONAL_MODULE_CACHE.clear()
        bootstrap._MISSING_OPTIONAL.clear()
        bootstrap._OPTIONAL_DEPENDENCY_WARNED.clear()

    assert exit_code == 0
    assert any("quick_fix_engine" in record.getMessage() for record in caplog.records)

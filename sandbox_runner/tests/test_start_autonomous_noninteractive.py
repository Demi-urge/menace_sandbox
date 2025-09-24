import sys
import threading
import types
from pathlib import Path
from dynamic_path_router import resolve_path

import logging
import pytest

# ensure repository root on path
sys.path.append(str(resolve_path("")))

from sandbox_settings import SandboxSettings  # noqa: E402


def _setup_base_packages() -> None:
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace"] = menace_pkg

    def fake_ensure_env(path: str) -> None:
        Path(path).touch()

    sys.modules["menace.auto_env_setup"] = types.SimpleNamespace(ensure_env=fake_ensure_env)
    sys.modules["menace.default_config_manager"] = types.SimpleNamespace(
        DefaultConfigManager=lambda *a, **k: types.SimpleNamespace(apply_defaults=lambda: None)
    )
    sys.modules["menace.environment_generator"] = types.SimpleNamespace(
        _CPU_LIMITS={}, _MEMORY_LIMITS={}
    )
    sys.modules["sandbox_runner.cli"] = types.SimpleNamespace(main=lambda *a, **k: None)
    sys.modules["sandbox_runner.cycle"] = types.SimpleNamespace(ensure_vector_service=lambda: None)
    sys.modules["self_improvement"] = types.ModuleType("self_improvement")


_setup_base_packages()


def test_start_autonomous_sandbox_noninteractive(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(repo))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)

    started = threading.Event()
    stop_event = threading.Event()

    def init_self_improvement(settings):
        return None

    def fake_start_self_improvement_cycle(workflows):
        def run():
            started.set()
            stop_event.wait()
        t = threading.Thread(target=run)
        return t

    def fake_stop_self_improvement_cycle():
        stop_event.set()

    api_stub = types.SimpleNamespace(
        init_self_improvement=init_self_improvement,
        start_self_improvement_cycle=fake_start_self_improvement_cycle,
        stop_self_improvement_cycle=fake_stop_self_improvement_cycle,
    )
    sys.modules["self_improvement.api"] = api_stub

    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.menace_env_file = str(tmp_path / ".env")
    settings.optional_service_versions = {}
    settings.sandbox_central_logging = False
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)

    import start_autonomous_sandbox as sas

    sas.main([])

    assert Path(settings.menace_env_file).exists()
    for name in ("metrics.db", "patch_history.db", "agent_queue.db"):
        assert (Path(settings.sandbox_data_dir) / name).exists()

    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    stop_event.set()
    thread.join(timeout=1)
    assert started.is_set()
    assert not getattr(thread, "_thread", thread).is_alive()
    bootstrap.shutdown_autonomous_sandbox()


def test_start_autonomous_sandbox_health_check(tmp_path, monkeypatch, caplog):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(repo))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)

    started = threading.Event()
    stop_event = threading.Event()

    def init_self_improvement(settings):
        return None

    def fake_start_self_improvement_cycle(workflows):
        def run():
            started.set()
            stop_event.wait()
        t = threading.Thread(target=run)
        return t

    def fake_stop_self_improvement_cycle():
        stop_event.set()

    api_stub = types.SimpleNamespace(
        init_self_improvement=init_self_improvement,
        start_self_improvement_cycle=fake_start_self_improvement_cycle,
        stop_self_improvement_cycle=fake_stop_self_improvement_cycle,
    )
    sys.modules["self_improvement.api"] = api_stub

    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.menace_env_file = str(tmp_path / ".env")
    settings.optional_service_versions = {}
    settings.sandbox_central_logging = False
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)

    import start_autonomous_sandbox as sas

    with caplog.at_level(logging.INFO):
        sas.main(["--health-check"])

    assert "Sandbox health" in caplog.text
    assert bootstrap._SELF_IMPROVEMENT_THREAD is None

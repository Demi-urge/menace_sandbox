import os
import sys
import threading
import types
from pathlib import Path
from dynamic_path_router import resolve_path

import pytest

sys.path.append(str(resolve_path("")))

from sandbox_settings import SandboxSettings  # noqa: E402


def _setup_base_packages():
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace"] = menace_pkg
    sys.modules["menace.auto_env_setup"] = types.SimpleNamespace(ensure_env=lambda *a, **k: None)
    sys.modules["menace.default_config_manager"] = types.SimpleNamespace(
        DefaultConfigManager=lambda *a, **k: types.SimpleNamespace(apply_defaults=lambda: None)
    )
    sys.modules["menace.environment_generator"] = types.SimpleNamespace(
        _CPU_LIMITS={}, _MEMORY_LIMITS={}
    )
    sys.modules["sandbox_runner.cli"] = types.SimpleNamespace(main=lambda *a, **k: None)
    sys.modules["sandbox_runner.cycle"] = types.SimpleNamespace(
        ensure_vector_service=lambda: None
    )
    si_pkg = types.ModuleType("self_improvement")
    sys.modules["self_improvement"] = si_pkg

    os.environ.setdefault("OPENAI_API_KEY", "test")
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("MODELS", str(Path.cwd()))


_setup_base_packages()


def _install_engine_stub():
    engine_module = types.ModuleType("self_improvement.engine")
    launch_calls = []
    skipped_calls = []

    def launch_autonomous_sandbox(*args, **kwargs):
        if getattr(engine_module, "_MANUAL_LAUNCH_TRIGGERED", False):
            skipped_calls.append((args, kwargs))
            return None

        engine_module._MANUAL_LAUNCH_TRIGGERED = True
        launch_calls.append((args, kwargs))
        return None

    engine_module.launch_autonomous_sandbox = launch_autonomous_sandbox  # type: ignore[attr-defined]
    engine_module._MANUAL_LAUNCH_TRIGGERED = False  # type: ignore[attr-defined]
    sys.modules["self_improvement.engine"] = engine_module
    si_pkg = sys.modules.get("self_improvement")
    if si_pkg is None:
        si_pkg = types.ModuleType("self_improvement")
        sys.modules["self_improvement"] = si_pkg
    setattr(si_pkg, "engine", engine_module)
    return engine_module, launch_calls, skipped_calls


def test_launch_sandbox_starts_self_improvement_thread(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
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

    sas.launch_sandbox()

    assert os.environ["SANDBOX_REPO_PATH"] == str(repo)
    assert os.environ["SANDBOX_DATA_DIR"] == str(data)

    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    stop_event.set()
    thread.join(timeout=1)
    assert started.is_set()
    assert not getattr(thread, "_thread", thread).is_alive()
    bootstrap.shutdown_autonomous_sandbox()


def test_launch_sandbox_runs_warmup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})

    warmup_called = threading.Event()

    def warmup():
        warmup_called.set()

    monkeypatch.setattr(bootstrap, "_self_improvement_warmup", warmup)

    started = threading.Event()
    stop_event = threading.Event()

    def init_self_improvement(settings):
        return None

    def fake_start_self_improvement_cycle(workflows):
        def run():
            workflows["bootstrap"]()
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

    sas.launch_sandbox()

    assert warmup_called.is_set()
    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()
    assert not getattr(thread, "_thread", thread).is_alive()
    bootstrap.shutdown_autonomous_sandbox()


def test_initialize_triggers_autonomous_launch_once(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_optional_modules", lambda *a, **k: set())
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    engine_module, launch_calls, skipped_calls = _install_engine_stub()

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

    bootstrap.initialize_autonomous_sandbox(settings)

    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()

    assert len(launch_calls) == 1
    args, kwargs = launch_calls[0]
    assert kwargs == {"background": True, "force": True}
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True
    assert skipped_calls == []

    bootstrap.shutdown_autonomous_sandbox()


def test_initialize_respects_manual_launch_guard(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_optional_modules", lambda *a, **k: set())
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    engine_module, launch_calls, skipped_calls = _install_engine_stub()
    engine_module._MANUAL_LAUNCH_TRIGGERED = True  # type: ignore[attr-defined]

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

    bootstrap.initialize_autonomous_sandbox(settings)

    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()

    assert launch_calls == []
    assert len(skipped_calls) == 1
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True

    bootstrap.shutdown_autonomous_sandbox()


def test_initialize_autonomous_sandbox_raises_on_dead_thread(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    class DeadThread:
        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    api_stub = types.SimpleNamespace(
        init_self_improvement=lambda s: None,
        start_self_improvement_cycle=lambda workflows: DeadThread(),
    )
    sys.modules["self_improvement.api"] = api_stub

    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.menace_env_file = str(tmp_path / ".env")
    settings.optional_service_versions = {}
    settings.sandbox_central_logging = False
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)

    with pytest.raises(RuntimeError):
        bootstrap.initialize_autonomous_sandbox(settings)


def test_initialize_autonomous_sandbox_warmup_failure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    def warmup():
        raise RuntimeError("boom")

    monkeypatch.setattr(bootstrap, "_self_improvement_warmup", warmup)

    class DummyThread:
        def start(self):
            pass

        def join(self, timeout=None):
            pass

    def fake_start_self_improvement_cycle(workflows):
        workflows["bootstrap"]()
        return DummyThread()

    api_stub = types.SimpleNamespace(
        init_self_improvement=lambda s: None,
        start_self_improvement_cycle=fake_start_self_improvement_cycle,
    )
    sys.modules["self_improvement.api"] = api_stub

    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.menace_env_file = str(tmp_path / ".env")
    settings.optional_service_versions = {}
    settings.sandbox_central_logging = False
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)

    with pytest.raises(RuntimeError):
        bootstrap.initialize_autonomous_sandbox(settings)


def test_shutdown_autonomous_sandbox_joins_thread(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "_start_optional_services", lambda *a, **k: None
    )
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

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

    bootstrap.initialize_autonomous_sandbox(settings)
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    assert getattr(thread, "_thread", thread).is_alive()

    bootstrap.shutdown_autonomous_sandbox()
    assert bootstrap._SELF_IMPROVEMENT_THREAD is None
    assert not getattr(thread, "_thread", thread).is_alive()

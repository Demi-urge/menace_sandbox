import importlib
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
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda *a, **k: {})
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)
    monkeypatch.setattr(bootstrap, "ensure_autonomous_launch", lambda *a, **k: True)

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

    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)
    monkeypatch.setattr(bootstrap, "ensure_autonomous_launch", lambda *a, **k: True)

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
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda *a, **k: {})

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

    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)
    monkeypatch.setattr(bootstrap, "ensure_autonomous_launch", lambda *a, **k: True)

    sas.launch_sandbox()

    assert warmup_called.is_set()
    stop_event.set()


def test_ensure_autonomous_launch_retries_on_import_error(monkeypatch):
    import sandbox_runner.bootstrap as bootstrap

    engine_module, launch_calls, _ = _install_engine_stub()
    engine_module._MANUAL_LAUNCH_TRIGGERED = False  # type: ignore[attr-defined]

    bootstrap._AUTONOMOUS_LAUNCH_RETRY = None
    bootstrap._SELF_IMPROVEMENT_THREAD = None

    class DummyTimer:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    schedule_calls: list[tuple[bool, bool, object]] = []
    timers: list[DummyTimer] = []

    def fake_schedule(*, background: bool, force: bool, thread: object) -> DummyTimer:
        schedule_calls.append((background, force, thread))
        timer = DummyTimer()
        timers.append(timer)
        bootstrap._AUTONOMOUS_LAUNCH_RETRY = timer
        return timer

    monkeypatch.setattr(
        bootstrap, "_schedule_autonomous_launch_retry", fake_schedule
    )

    call_count = 0

    def fake_import(name: str):
        nonlocal call_count
        assert name == "self_improvement.engine"
        call_count += 1
        if call_count == 1:
            raise ImportError("boom")
        return engine_module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    target_thread = types.SimpleNamespace(is_alive=lambda: True)

    first_result = bootstrap.ensure_autonomous_launch(thread=target_thread)
    assert first_result is False
    assert schedule_calls == [(True, False, target_thread)]
    assert timers and not timers[0].cancelled
    assert bootstrap._AUTONOMOUS_LAUNCH_RETRY is timers[0]
    assert call_count == 1

    second_result = bootstrap.ensure_autonomous_launch()
    assert second_result is True
    assert len(launch_calls) == 1
    assert timers[0].cancelled is True
    assert bootstrap._AUTONOMOUS_LAUNCH_RETRY is None
    assert call_count == 2

    bootstrap._SELF_IMPROVEMENT_THREAD = None
    bootstrap._AUTONOMOUS_LAUNCH_RETRY = None


def test_ensure_autonomous_launch_import_error_coalesces_retry(monkeypatch):
    import sandbox_runner.bootstrap as bootstrap

    engine_module, _, _ = _install_engine_stub()
    engine_module._MANUAL_LAUNCH_TRIGGERED = False  # type: ignore[attr-defined]

    bootstrap._AUTONOMOUS_LAUNCH_RETRY = None
    bootstrap._SELF_IMPROVEMENT_THREAD = None

    class DummyTimer:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    schedule_calls: list[tuple[bool, bool, object]] = []
    timers: list[DummyTimer] = []

    def fake_schedule(*, background: bool, force: bool, thread: object) -> None:
        schedule_calls.append((background, force, thread))
        timer = DummyTimer()
        timers.append(timer)
        bootstrap._AUTONOMOUS_LAUNCH_RETRY = timer

    monkeypatch.setattr(
        bootstrap, "_schedule_autonomous_launch_retry", fake_schedule
    )

    call_count = 0

    def fake_import(name: str):
        nonlocal call_count
        assert name == "self_improvement.engine"
        call_count += 1
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import)

    target_thread = types.SimpleNamespace(is_alive=lambda: True)

    first_result = bootstrap.ensure_autonomous_launch(thread=target_thread)
    assert first_result is False
    assert schedule_calls == [(True, False, target_thread)]
    assert timers and bootstrap._AUTONOMOUS_LAUNCH_RETRY is timers[0]
    assert call_count == 1

    second_result = bootstrap.ensure_autonomous_launch()
    assert second_result is False
    assert schedule_calls == [(True, False, target_thread)]
    assert bootstrap._AUTONOMOUS_LAUNCH_RETRY is timers[0]
    assert call_count == 2

    bootstrap._SELF_IMPROVEMENT_THREAD = None
    bootstrap._AUTONOMOUS_LAUNCH_RETRY = None


def test_initialize_does_not_trigger_autonomous_launch(tmp_path, monkeypatch):
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
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda *a, **k: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)
    monkeypatch.setattr(bootstrap, "ensure_env", lambda *_a, **_k: None)
    monkeypatch.setattr(
        bootstrap,
        "DefaultConfigManager",
        lambda *_a, **_k: types.SimpleNamespace(apply_defaults=lambda: None),
    )

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

    bootstrap.bootstrap_environment(settings)

    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()

    assert launch_calls == []
    assert skipped_calls == []

    bootstrap.shutdown_autonomous_sandbox()


def test_launch_sandbox_triggers_autonomous_launch_once(tmp_path, monkeypatch):
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
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda *a, **k: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)

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

    import start_autonomous_sandbox as sas

    sas.launch_sandbox()

    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()

    assert len(launch_calls) == 1
    args, kwargs = launch_calls[0]
    assert kwargs == {"background": True, "force": True}
    assert skipped_calls == []

    bootstrap.shutdown_autonomous_sandbox()


def test_launch_sandbox_respects_manual_launch_guard(tmp_path, monkeypatch):
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
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda *a, **k: {})
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)

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

    import start_autonomous_sandbox as sas

    sas.launch_sandbox()

    stop_event.set()
    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()

    assert launch_calls == []
    assert len(skipped_calls) == 1
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True

    bootstrap.shutdown_autonomous_sandbox()


def test_ensure_autonomous_launch_forces_dead_thread(monkeypatch):
    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)
    engine_module, launch_calls, skipped_calls = _install_engine_stub()

    dead_thread = types.SimpleNamespace(
        _thread=types.SimpleNamespace(is_alive=lambda: False)
    )

    result = bootstrap.ensure_autonomous_launch(
        background=False, force=True, thread=dead_thread
    )

    assert result is True
    assert bootstrap._SELF_IMPROVEMENT_THREAD is dead_thread
    assert len(launch_calls) == 1
    args, kwargs = launch_calls[0]
    assert args == ()
    assert kwargs == {"background": False, "force": True}
    assert skipped_calls == []
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True

    second = bootstrap.ensure_autonomous_launch(
        background=False, force=True, thread=dead_thread
    )

    assert second is True
    assert len(launch_calls) == 1
    assert len(skipped_calls) == 1


def test_ensure_autonomous_launch_retries_until_available(monkeypatch):
    import sandbox_runner.bootstrap as bootstrap

    thread = types.SimpleNamespace(
        _thread=types.SimpleNamespace(is_alive=lambda: True)
    )
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", thread)

    engine_module, launch_calls, skipped_calls = _install_engine_stub()
    launch_impl = engine_module.launch_autonomous_sandbox
    delattr(engine_module, "launch_autonomous_sandbox")

    scheduled = []

    def fake_schedule(*, background: bool, force: bool, thread, delay: float = 0.5):
        scheduled.append((background, force, thread, delay))
        engine_module.launch_autonomous_sandbox = launch_impl  # type: ignore[attr-defined]
        result = bootstrap.ensure_autonomous_launch(
            background=background, force=force, thread=thread
        )
        assert result is True

    monkeypatch.setattr(
        bootstrap, "_schedule_autonomous_launch_retry", fake_schedule
    )
    monkeypatch.setattr(bootstrap, "_AUTONOMOUS_LAUNCH_RETRY", None)

    result = bootstrap.ensure_autonomous_launch(background=False, force=False)

    assert result is False
    assert scheduled
    assert len(scheduled) == 1
    assert len(launch_calls) == 1
    assert skipped_calls == []
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True


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

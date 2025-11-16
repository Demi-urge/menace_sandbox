import sys
import threading
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.path.append(str(resolve_path("")))

from sandbox_settings import SandboxSettings  # noqa: E402


def _setup_base(monkeypatch):
    monkeypatch.setattr(
        "sandbox_runner.bootstrap._start_optional_services",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("sandbox_runner.bootstrap.ensure_vector_service", lambda: None)
    monkeypatch.setattr("sandbox_runner.bootstrap._verify_required_dependencies", lambda s: {})


def test_shutdown_cleans_caches(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    _setup_base(monkeypatch)

    import sandbox_runner.bootstrap as bootstrap
    import sandbox_runner.generative_stub_provider as gsp
    import self_improvement.utils as si_utils

    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    flags = {"gf": 0, "gc": 0, "cf": 0, "cc": 0}

    def _inc(name: str) -> None:
        flags[name] += 1

    monkeypatch.setattr(gsp, "flush_caches", lambda config=None: _inc("gf"))
    monkeypatch.setattr(gsp, "cleanup_cache_files", lambda config=None: _inc("gc"))
    monkeypatch.setattr(si_utils, "clear_import_cache", lambda: _inc("cf"))
    monkeypatch.setattr(si_utils, "remove_import_cache_files", lambda base=None: _inc("cc"))

    stop_event = threading.Event()

    def init_self_improvement(settings):
        return None

    def fake_start_self_improvement_cycle(workflows):
        def run():
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
    bootstrap.shutdown_autonomous_sandbox()
    assert flags == {"gf": 1, "gc": 1, "cf": 1, "cc": 1}

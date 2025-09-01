import os
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

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


_setup_base_packages()


def test_launch_sandbox_starts_self_improvement_thread(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted"))

    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda mods: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)

    started = threading.Event()

    def init_self_improvement(settings):
        return None

    def fake_start_self_improvement_cycle(workflows):
        def run():
            started.set()
        t = threading.Thread(target=run)
        return t

    api_stub = types.SimpleNamespace(
        init_self_improvement=init_self_improvement,
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

    import start_autonomous_sandbox as sas

    sas.launch_sandbox()

    assert os.environ["SANDBOX_REPO_PATH"] == str(repo)
    assert os.environ["SANDBOX_DATA_DIR"] == str(data)

    thread = bootstrap._SELF_IMPROVEMENT_THREAD
    assert thread is not None
    thread.join(timeout=1)
    assert started.is_set()
    assert not thread.is_alive()

import json
import os
import sys
import threading
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.path.append(str(resolve_path("")))

import dynamic_path_router as _dynamic_path_router  # noqa: E402  (ensure repo_root exists for imports)

if not hasattr(_dynamic_path_router, "repo_root"):
    _dynamic_path_router.repo_root = _dynamic_path_router.get_project_root  # type: ignore[attr-defined]


# Minimal stubs for external modules used during sandbox bootstrap

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
    sys.modules["sandbox_runner.generative_stub_provider"] = types.SimpleNamespace(
        _GENERATOR=object(),
        flush_caches=lambda config=None: Path(
            os.getenv("SANDBOX_STUB_CACHE", str(Path.cwd() / "stub_cache.json"))
        ).write_text("[]"),
        cleanup_cache_files=lambda config=None: None,
    )
    si_pkg = types.ModuleType("self_improvement")
    sys.modules["self_improvement"] = si_pkg

    os.environ.setdefault("OPENAI_API_KEY", "test")
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("MODELS", str(Path.cwd()))


_setup_base_packages()

from sandbox_settings import SandboxSettings  # noqa: E402
import sandbox_runner.bootstrap as bootstrap  # noqa: E402
import start_autonomous_sandbox as sas  # noqa: E402


def test_optional_import_retries_with_package(monkeypatch):
    cache: dict[str, types.ModuleType] = {}
    monkeypatch.setattr(bootstrap, "_OPTIONAL_MODULE_CACHE", cache)

    import_calls: list[str] = []
    module = types.ModuleType("menace_sandbox.fallback_demo_mod")

    def fake_import(name: str):
        import_calls.append(name)
        if name == "fallback_demo_mod":
            raise ImportError("attempted relative import with no known parent package")
        if name == "menace_sandbox.fallback_demo_mod":
            return module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bootstrap.importlib, "import_module", fake_import)
    monkeypatch.delitem(sys.modules, "fallback_demo_mod", raising=False)
    monkeypatch.delitem(sys.modules, "sandbox_runner.fallback_demo_mod", raising=False)
    monkeypatch.delitem(sys.modules, "menace_sandbox.fallback_demo_mod", raising=False)

    loaded = bootstrap._import_optional_module("fallback_demo_mod")

    assert loaded is module
    assert import_calls[0] == "fallback_demo_mod"
    assert import_calls[-1] == "menace_sandbox.fallback_demo_mod"
    assert import_calls.index("menace_sandbox.fallback_demo_mod") > 0
    assert sys.modules["fallback_demo_mod"] is module
    assert cache["fallback_demo_mod"] is module

    import_calls.clear()
    again = bootstrap._import_optional_module("fallback_demo_mod")
    assert again is module
    assert import_calls == []


def test_sandbox_health_and_artifacts(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setenv("MODELS", str(repo))
    monkeypatch.setenv("SYNERGY_WEIGHT_FILE", str(data / "synergy_weights.json"))
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(data / "stub_cache.json"))
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "demo-model")

    # Stub model registry so the stub generator initialises without heavy deps
    sys.modules["model_registry"] = types.SimpleNamespace(get_client=lambda *a, **k: object())

    # Provide lightweight self-improvement API to create synergy weights and thread
    started = threading.Event()
    stop_event = threading.Event()

    def init_self_improvement(settings):
        path = Path(settings.synergy_weight_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_doc": "Default synergy weights", **settings.default_synergy_weights}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def start_self_improvement_cycle(workflows):
        def run():
            started.set()
            stop_event.wait()

        t = threading.Thread(target=run)
        return t

    def stop_self_improvement_cycle():
        stop_event.set()

    api_stub = types.SimpleNamespace(
        init_self_improvement=init_self_improvement,
        start_self_improvement_cycle=start_self_improvement_cycle,
        stop_self_improvement_cycle=stop_self_improvement_cycle,
    )
    sys.modules["self_improvement.api"] = api_stub

    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.synergy_weight_file = str(data / "synergy_weights.json")
    settings.synergy_weights_path = settings.synergy_weight_file
    settings.menace_env_file = str(tmp_path / ".env")
    settings.optional_service_versions = {}
    settings.sandbox_central_logging = False
    settings.sandbox_stub_model = "demo-model"

    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda mods: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", lambda s: {})
    monkeypatch.setattr(bootstrap, "_verify_optional_modules", lambda *a, **k: set())
    monkeypatch.setattr(bootstrap, "_cli_main", lambda args: None)

    sas.main([])

    import sandbox_runner.generative_stub_provider as gsp

    # Persist stub cache to disk
    gsp.flush_caches()

    health = bootstrap.sandbox_health()
    assert health == {
        "self_improvement_thread_alive": True,
        "databases_accessible": True,
        "database_errors": {},
        "stub_generator_initialized": True,
    }

    assert (data / "synergy_weights.json").exists()
    assert (data / "stub_cache.json").exists()
    for name in settings.sandbox_required_db_files:
        assert (data / name).exists()

    bootstrap.shutdown_autonomous_sandbox()

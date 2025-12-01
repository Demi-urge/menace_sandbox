import asyncio
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import types

def _reload_bootstrap(monkeypatch, tmp_path):
    marker = tmp_path / "marker"
    lock = tmp_path / "lock"
    monkeypatch.setenv("MENACE_BOOTSTRAP_MARKER", str(marker))
    monkeypatch.setenv("MENACE_BOOTSTRAP_LOCK", str(lock))
    module_name = "menace_sandbox.environment_bootstrap"
    pkg_name = "menace_sandbox"
    sys.modules.pop(module_name, None)
    sys.modules.pop(pkg_name, None)

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
    sys.modules[pkg_name] = pkg

    vector_pkg = types.ModuleType(f"{pkg_name}.vector_service")
    vector_pkg.__path__ = []
    sys.modules[f"{pkg_name}.vector_service"] = vector_pkg
    scheduler_stub = types.ModuleType(
        f"{pkg_name}.vector_service.embedding_scheduler"
    )
    scheduler_stub.start_scheduler_from_env = lambda: None
    sys.modules[f"{pkg_name}.vector_service.embedding_scheduler"] = scheduler_stub

    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parent.parent / "environment_bootstrap.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ensure_bootstrapped_runs_once(monkeypatch, tmp_path, caplog):
    eb = _reload_bootstrap(monkeypatch, tmp_path)
    calls: list[dict] = []

    class _StubBootstrapper(eb.EnvironmentBootstrapper):
        def __init__(self):
            pass

        def bootstrap(self, **kwargs):
            calls.append(kwargs)

    caplog.set_level(logging.INFO)
    result = eb.ensure_bootstrapped(bootstrapper=_StubBootstrapper())
    assert result["ready"] is False or result["ready"] is True
    assert (tmp_path / "marker").exists()
    assert eb.ensure_bootstrapped(bootstrapper=_StubBootstrapper()) == result
    assert len(calls) == 1
    assert any("bootstrap already completed" in rec.message for rec in caplog.records)


def test_ensure_bootstrapped_async(monkeypatch, tmp_path, caplog):
    eb = _reload_bootstrap(monkeypatch, tmp_path)
    calls: list[dict] = []

    class _StubBootstrapper(eb.EnvironmentBootstrapper):
        def __init__(self):
            pass

        def bootstrap(self, **kwargs):
            calls.append(kwargs)

    caplog.set_level(logging.INFO)
    async_result = asyncio.run(eb.ensure_bootstrapped_async(bootstrapper=_StubBootstrapper()))
    assert async_result["ready"] is False or async_result["ready"] is True
    assert (
        asyncio.run(eb.ensure_bootstrapped_async(bootstrapper=_StubBootstrapper()))
        == async_result
    )
    assert len(calls) == 1
    assert any("bootstrap already completed" in rec.message for rec in caplog.records)

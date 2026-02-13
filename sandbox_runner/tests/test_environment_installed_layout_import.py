from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path


def test_resolve_self_debugger_sandbox_class_in_installed_like_layout(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    installed_root = tmp_path / "site_packages"
    installed_root.mkdir()

    shutil.copytree(repo_root / "sandbox_runner", installed_root / "sandbox_runner")

    menace_pkg = installed_root / "menace"
    menace_pkg.mkdir()
    (menace_pkg / "__init__.py").write_text("", encoding="utf-8")
    (menace_pkg / "self_debugger_sandbox.py").write_text(
        "class SelfDebuggerSandbox:\n    pass\n",
        encoding="utf-8",
    )

    class _ContextBuilder:
        pass

    vector_service_pkg = types.ModuleType("vector_service")
    context_builder_mod = types.ModuleType("vector_service.context_builder")
    context_builder_mod.ContextBuilder = _ContextBuilder

    log_mod = types.ModuleType("logging_utils")
    log_mod.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    log_mod.log_record = lambda *a, **k: None

    counter = types.SimpleNamespace(labels=lambda *a, **k: counter, inc=lambda *a, **k: None)
    metrics_mod = types.ModuleType("metrics_exporter")
    metrics_mod.sandbox_crashes_total = counter
    metrics_mod.environment_failure_total = counter
    metrics_mod.Gauge = lambda *a, **k: counter

    alert_mod = types.ModuleType("alert_dispatcher")
    alert_mod.dispatch_alert = lambda *a, **k: None

    path_mod = types.ModuleType("dynamic_path_router")
    path_mod.resolve_path = lambda value: Path(value)
    path_mod.repo_root = lambda: repo_root
    path_mod.path_for_prompt = lambda value: str(value)
    path_mod.get_project_root = lambda: repo_root

    ontology_mod = types.ModuleType("error_ontology")
    ontology_mod.ErrorCategory = object
    ontology_mod.classify_error = lambda *a, **k: None

    class _DummyLock:
        def __init__(self, *_a, **_k):
            pass

        def acquire(self, *_a, **_k):
            return None

        def release(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

    lock_mod = types.ModuleType("lock_utils")
    lock_mod.SandboxLock = _DummyLock
    lock_mod.Timeout = RuntimeError

    settings_mod = types.ModuleType("sandbox_settings")
    settings_mod.SandboxSettings = object

    monkeypatch.syspath_prepend(str(installed_root))
    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", context_builder_mod)
    monkeypatch.setitem(sys.modules, "logging_utils", log_mod)
    monkeypatch.setitem(sys.modules, "metrics_exporter", metrics_mod)
    monkeypatch.setitem(sys.modules, "alert_dispatcher", alert_mod)
    monkeypatch.setitem(sys.modules, "dynamic_path_router", path_mod)
    monkeypatch.setitem(sys.modules, "error_ontology", ontology_mod)
    monkeypatch.setitem(sys.modules, "lock_utils", lock_mod)
    monkeypatch.setitem(sys.modules, "sandbox_settings", settings_mod)

    orphan_mod = types.ModuleType("sandbox_runner.orphan_integration")
    orphan_mod.integrate_and_graph_orphans = lambda *a, **k: {}
    scoring_mod = types.ModuleType("sandbox_runner.scoring")
    scoring_mod.record_run = lambda *a, **k: None
    scoring_mod.load_summary = lambda *a, **k: {}

    for name in [
        "sandbox_runner",
        "sandbox_runner.environment",
        "sandbox_runner.import_candidates",
        "menace",
        "menace.self_debugger_sandbox",
    ]:
        sys.modules.pop(name, None)

    sandbox_runner_pkg = types.ModuleType("sandbox_runner")
    sandbox_runner_pkg.__path__ = [str(installed_root / "sandbox_runner")]
    monkeypatch.setitem(sys.modules, "sandbox_runner", sandbox_runner_pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", orphan_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.scoring", scoring_mod)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment",
        installed_root / "sandbox_runner" / "environment.py",
    )
    assert spec and spec.loader
    env = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env)
    spec.loader.exec_module(env)
    resolved = env._resolve_self_debugger_sandbox_class()

    assert resolved.__name__ == "SelfDebuggerSandbox"
    assert resolved.__module__ == "menace.self_debugger_sandbox"

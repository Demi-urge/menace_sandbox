import sys
import shutil
from pathlib import Path
import importlib
import types

import dynamic_path_router  # noqa: E402
from dynamic_path_router import clear_cache, resolve_path  # noqa: E402


def test_simulate_full_environment_uses_resolved_path(tmp_path, monkeypatch):
    original = Path(resolve_path("sandbox_runner.py"))
    sandbox_file = original.name
    alt_root = tmp_path / "alt_root"
    alt_root.mkdir()
    (alt_root / "sandbox_data").mkdir()
    shutil.copy2(original, alt_root / sandbox_file)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(alt_root))
    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.delenv("OS_TYPE", raising=False)
    clear_cache()
    monkeypatch.setattr(
        dynamic_path_router, "resolve_path", lambda name: (alt_root / name)
    )

    class DummyErrorLogger:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules, "error_logger", types.SimpleNamespace(ErrorLogger=DummyErrorLogger)
    )
    env_mod = importlib.import_module("sandbox_runner.environment")
    importlib.reload(env_mod)

    calls = {}

    def fake_run(cmd, cwd=None, env=None, check=None, stdout=None, stderr=None):
        calls['cmd'] = cmd
        calls['cwd'] = cwd

        class Dummy:
            pass

        return Dummy()

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

    class DummyTracker:
        diagnostics = {}

        def load_history(self, *_args, **_kwargs):
            pass
    monkeypatch.setitem(
        sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker)
    )

    env_mod.simulate_full_environment({})

    assert Path(calls['cmd'][1]) == alt_root / sandbox_file
    assert Path(calls['cwd']) == alt_root


def test_resolvers_fallback_to_flat_modules(monkeypatch):
    env_mod = importlib.import_module("sandbox_runner.environment")

    real_import_module = importlib.import_module

    fake_modules = {
        "task_handoff_bot": types.SimpleNamespace(
            WorkflowDB=type("WorkflowDB", (), {}),
            WorkflowRecord=type("WorkflowRecord", (), {}),
        ),
        "self_coding_engine": types.SimpleNamespace(
            SelfCodingEngine=type("SelfCodingEngine", (), {}),
        ),
        "code_database": types.SimpleNamespace(
            CodeDB=type("CodeDB", (), {}),
        ),
        "menace_memory_manager": types.SimpleNamespace(
            MenaceMemoryManager=type("MenaceMemoryManager", (), {}),
        ),
    }

    def fake_import(name: str, package: str | None = None):
        if name.startswith("menace."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        if name in fake_modules:
            return fake_modules[name]
        return real_import_module(name, package)

    monkeypatch.setattr(env_mod.importlib, "import_module", fake_import)

    workflow_db, workflow_record = env_mod._resolve_workflow_db_types()
    assert workflow_db is fake_modules["task_handoff_bot"].WorkflowDB
    assert workflow_record is fake_modules["task_handoff_bot"].WorkflowRecord
    assert env_mod._resolve_self_coding_engine_class() is fake_modules["self_coding_engine"].SelfCodingEngine
    assert env_mod._resolve_code_db_class() is fake_modules["code_database"].CodeDB
    assert env_mod._resolve_memory_manager_class() is fake_modules["menace_memory_manager"].MenaceMemoryManager


def test_workflow_db_resolver_error_includes_both_attempts(monkeypatch):
    env_mod = importlib.import_module("sandbox_runner.environment")

    def fail_import(name: str, package: str | None = None):
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(env_mod.importlib, "import_module", fail_import)

    with __import__("pytest").raises(ModuleNotFoundError) as exc:
        env_mod._resolve_workflow_db_types()

    message = str(exc.value)
    assert "menace.task_handoff_bot" in message
    assert "task_handoff_bot" in message
    assert "Package attempt error" in message
    assert "Flat attempt error" in message

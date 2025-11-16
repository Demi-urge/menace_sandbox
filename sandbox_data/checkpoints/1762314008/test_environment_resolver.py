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

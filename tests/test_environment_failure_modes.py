from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "sqlalchemy" not in sys.modules:
    sa = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = object
    sa.engine = engine_mod
    sys.modules["sqlalchemy"] = sa
    sys.modules["sqlalchemy.engine"] = engine_mod
if "pyroute2" not in sys.modules:
    pr2 = types.ModuleType("pyroute2")
    pr2.IPRoute = pr2.NSPopen = pr2.netns = object
    sys.modules["pyroute2"] = pr2

if "error_logger" not in sys.modules:
    el = types.ModuleType("error_logger")
    el.ErrorLogger = type("ErrorLogger", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["error_logger"] = el

sys.modules.pop("sandbox_runner", None)
sys.modules.pop("sandbox_runner.environment", None)
import sandbox_runner.environment as env  # noqa: E402


def test_cpu_spike_failure_surfaces_error(monkeypatch):
    import threading

    class BoomThread:
        def __init__(self, *a, **k):
            raise RuntimeError("boom thread")

    monkeypatch.setattr(threading, "Thread", BoomThread)
    snippet = env._inject_failure_modes("", {"cpu_spike"})
    with pytest.raises(RuntimeError, match="boom thread"):
        exec(snippet, {})


def test_concurrency_spike_failure_surfaces_error(monkeypatch, capsys):
    import asyncio

    def boom_run(_):
        raise RuntimeError("burst fail")

    monkeypatch.setattr(asyncio, "run", boom_run)
    snippet = env._inject_failure_modes("", {"concurrency_spike"})
    with pytest.raises(RuntimeError, match="burst fail"):
        exec(snippet, {})
    assert "concurrency spike burst failed" in capsys.readouterr().err


def test_misuse_provider_failure_surfaces_error(monkeypatch):
    class DummySettings:
        misuse_stubs = True

        def __init__(self):
            pass

    monkeypatch.setattr(env, "SandboxSettings", DummySettings)

    def boom(*a, **k):
        raise ValueError("hostile fail")

    monkeypatch.setattr(env, "_hostile_strategy", boom)
    with pytest.raises(ValueError, match="hostile fail"):
        env._misuse_provider([], {})

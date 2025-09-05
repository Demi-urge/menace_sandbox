import logging
import sys
import types
from pathlib import Path


def _load_radar_snippet():
    src = (Path(__file__).resolve().parents[1] / "sandbox_runner" / "environment.py").read_text()  # path-ignore
    start = src.index("# Relevancy radar integration")
    end = src.index("import builtins", start)
    snippet = "from __future__ import annotations\n" + src[start:end]
    ns: dict[str, object] = {
        "os": types.SimpleNamespace(getenv=lambda _k: "1"),
        "queue": types.SimpleNamespace(Queue=lambda: None),
        "threading": types.SimpleNamespace(Thread=lambda *a, **k: None),
        "record_error": lambda _e: None,
        "atexit": types.SimpleNamespace(register=lambda _f: None),
        "logger": logging.getLogger("test"),
    }
    sys.modules["relevancy_radar"] = types.ModuleType("relevancy_radar")
    exec(snippet, ns)
    return ns


def test_warns_once_when_radar_missing(caplog):
    env = _load_radar_snippet()
    env["_RADAR_WARNING_EMITTED"] = False
    with caplog.at_level(logging.WARNING):
        env["_async_radar_track"]("foo")
        env["_async_radar_track"]("bar")
    messages = [r.getMessage() for r in caplog.records if "relevancy_radar" in r.getMessage()]
    assert messages == ["relevancy_radar dependency missing; tracking disabled"]
    assert not env["RELEVANCY_RADAR_AVAILABLE"]


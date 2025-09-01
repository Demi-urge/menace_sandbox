import os
import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.pop("sandbox_runner", None)

import types

err_log_mod = types.SimpleNamespace(ErrorLogger=lambda *a, **k: None)
sys.modules.setdefault("error_logger", err_log_mod)
sys.modules.setdefault("menace.error_logger", err_log_mod)

import sandbox_runner.environment as env  # noqa: E402


def test_generate_input_stubs_respects_dynamic_strategy(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    monkeypatch.setenv("SANDBOX_MISUSE_STUBS", "0")
    importlib.reload(env)

    captured = []

    def recorder(stubs, ctx):
        captured.append(ctx.get("strategy"))
        return stubs

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "random")
    env.generate_input_stubs(1, providers=[recorder])
    assert captured[-1] == "random"

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "hostile")
    env.generate_input_stubs(1, providers=[recorder])
    assert captured[-1] == "hostile"


def test_generate_input_stubs_respects_dynamic_misuse(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "hostile")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    importlib.reload(env)

    monkeypatch.setenv("SANDBOX_MISUSE_STUBS", "0")
    stubs1 = env.generate_input_stubs(1)
    assert len(stubs1) == 1

    monkeypatch.setenv("SANDBOX_MISUSE_STUBS", "1")
    stubs2 = env.generate_input_stubs(1)
    assert len(stubs2) > len(stubs1)

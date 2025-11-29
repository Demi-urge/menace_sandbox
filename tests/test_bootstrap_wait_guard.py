import importlib
import os

from bootstrap_timeout_policy import (
    _BOOTSTRAP_TIMEOUT_MINIMUMS,
    guard_bootstrap_wait_env,
)


def test_guard_bootstrap_wait_env_clamps_missing_or_low(monkeypatch):
    local_env: dict[str, str] = {}

    resolved_missing = guard_bootstrap_wait_env(env=local_env)
    assert resolved_missing["MENACE_BOOTSTRAP_WAIT_SECS"] == _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_WAIT_SECS"
    ]
    assert resolved_missing["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"] == _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"
    ]

    local_env.update(
        {
            "MENACE_BOOTSTRAP_WAIT_SECS": "12",
            "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": "15",
        }
    )
    resolved_low = guard_bootstrap_wait_env(env=local_env)
    assert resolved_low["MENACE_BOOTSTRAP_WAIT_SECS"] == _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_WAIT_SECS"
    ]
    assert resolved_low["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"] == _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"
    ]


def test_manual_bootstrap_hydrates_wait_env(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "1")
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "2")

    module = importlib.import_module("manual_bootstrap")
    importlib.reload(module)

    assert float(os.environ["MENACE_BOOTSTRAP_WAIT_SECS"]) >= _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_WAIT_SECS"
    ]
    assert float(
        os.environ["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"]
    ) >= _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"]

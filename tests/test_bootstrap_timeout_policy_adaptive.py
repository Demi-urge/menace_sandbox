from __future__ import annotations

import importlib
import json
from pathlib import Path


def _reload_policy(monkeypatch, state_path: Path):
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(state_path))
    import bootstrap_timeout_policy as btp

    return importlib.reload(btp)


def test_component_overruns_persist_adaptive_floors(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)

    telemetry = {
        "shared_timeout": {
            "timeline": [
                {
                    "label": "vectorizer warmup",
                    "elapsed": 520.0,
                    "effective": 300.0,
                    "vector_heavy": True,
                },
                {
                    "label": "vectorizer reload",
                    "elapsed": 480.0,
                    "effective": 300.0,
                    "vector_heavy": True,
                },
            ]
        }
    }

    budgets = policy.compute_prepare_pipeline_component_budgets(telemetry=telemetry)

    assert budgets["vectorizers"] >= 550.0

    state = json.loads(state_path.read_text())
    host_state = state.get(policy._state_host_key(), {})  # type: ignore[attr-defined]
    component_floors = host_state.get("component_floors", {})

    assert component_floors.get("vectorizers", 0) >= 520.0

    policy_again = _reload_policy(monkeypatch, state_path)
    next_budgets = policy_again.compute_prepare_pipeline_component_budgets()

    assert next_budgets["vectorizers"] >= component_floors["vectorizers"]


def test_persisted_floors_feed_component_coordinator(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)

    overrun_state = {
        policy._state_host_key(): {  # type: ignore[attr-defined]
            "component_overruns": {
                "retrievers": {
                    "overruns": 3,
                    "max_elapsed": 400.0,
                    "expected_floor": 320.0,
                    "suggested_floor": 360.0,
                }
            }
        }
    }

    state_path.write_text(json.dumps(overrun_state))

    policy = _reload_policy(monkeypatch, state_path)
    budgets = policy.compute_prepare_pipeline_component_budgets()

    coordinator = policy.SharedTimeoutCoordinator(
        sum(budgets.values()),
        component_floors=policy.load_component_timeout_floors(),
        component_budgets={"retrievers": budgets["retrievers"]},
    )

    windows = coordinator.start_component_timers(
        {"retrievers": budgets["retrievers"]}, minimum=0.0
    )

    assert windows["retrievers"]["budget"] >= 360.0

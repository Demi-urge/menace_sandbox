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


def test_global_window_scales_with_component_complexity(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)

    monkeypatch.setattr(policy, "_cluster_budget_scale", lambda **_: (1.0, {}))
    monkeypatch.setattr(policy, "_host_load_scale", lambda _load=None: 1.0)
    monkeypatch.setattr(policy, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(policy, "_save_timeout_state", lambda state: state_path.write_text(json.dumps(state)))

    base_budgets = policy.compute_prepare_pipeline_component_budgets(
        pipeline_complexity={"vectorizers": ["a"], "retrievers": 1}
    )
    base_window, base_inputs = policy.load_last_global_bootstrap_window()

    assert base_budgets
    assert base_window is not None
    assert base_inputs.get("component_complexity")

    expanded_budgets = policy.compute_prepare_pipeline_component_budgets(
        pipeline_complexity={"vectorizers": ["a", "b", "c"], "db_indexes": [1, 2]},
    )
    expanded_window, _ = policy.load_last_global_bootstrap_window()

    assert expanded_budgets
    assert expanded_window is not None
    assert expanded_window > base_window


def test_adaptive_component_floors_include_guard_and_peer_state(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)

    monkeypatch.setenv("MENACE_BOOTSTRAP_LOAD_THRESHOLD", "1.0")
    monkeypatch.setenv("MENACE_BOOTSTRAP_COMPONENT_FLOOR_MAX_SCALE", "2.0")

    peer_state = {
        "peer-host": {
            "component_floors": {"retrievers": 900.0},
            "updated_at": 0,
        }
    }
    state_path.write_text(json.dumps(peer_state))

    policy._record_bootstrap_guard(30.0, 2.0, source="test", host_load=2.5)  # type: ignore[attr-defined]
    floors = policy.load_component_timeout_floors()

    assert floors["vectorizers"] >= policy._COMPONENT_TIMEOUT_MINIMUMS[  # type: ignore[attr-defined]
        "vectorizers"
    ] * 2.0
    assert floors["retrievers"] == policy._COMPONENT_TIMEOUT_MINIMUMS[  # type: ignore[attr-defined]
        "retrievers"
    ] * 2.0

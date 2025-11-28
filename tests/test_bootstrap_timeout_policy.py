from __future__ import annotations

import json
import os
import importlib

import bootstrap_timeout_policy as btp
import coding_bot_interface as cbi


def test_component_budgets_expand_with_overruns(monkeypatch):
    floors = {
        "vectorizers": 120.0,
        "retrievers": 90.0,
        "db_indexes": 60.0,
        "orchestrator_state": 50.0,
    }

    monkeypatch.setattr(btp, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(btp, "_save_timeout_state", lambda state: None)
    monkeypatch.setattr(
        btp,
        "_load_timeout_state",
        lambda: {
            "test-host": {
                "component_overruns": {
                    "retrievers": {
                        "overruns": 3,
                        "max_elapsed": 180.0,
                        "expected_floor": 120.0,
                        "suggested_floor": 200.0,
                    }
                }
            }
        },
    )

    telemetry = {
        "shared_timeout": {
            "timeline": [
                {
                    "label": "vectorizer warmup",
                    "elapsed": 150.0,
                    "effective": 100.0,
                }
            ]
        }
    }

    budgets = btp.compute_prepare_pipeline_component_budgets(
        component_floors=floors, telemetry=telemetry, load_average=2.0
    )

    assert budgets["vectorizers"] > floors["vectorizers"]
    assert budgets["retrievers"] > floors["retrievers"]


def test_vector_heavy_budgets_extend_bootstrap_wait(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(state_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "300")
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    importlib.reload(btp)
    importlib.reload(cbi)

    budgets = {
        "vectorizers": 720.0,
        "retrievers": 420.0,
        "db_indexes": 260.0,
        "orchestrator_state": 190.0,
        "pipeline_config": 140.0,
        "background_loops": 120.0,
    }

    def _persisted_compute(**_kwargs: object) -> dict[str, float]:
        host_state = {
            btp._state_host_key(): {  # type: ignore[attr-defined]
                "component_floors": dict(budgets),
                "last_component_budgets": dict(budgets),
                "last_component_budget_total": sum(budgets.values()),
            }
        }
        btp._save_timeout_state(host_state)  # type: ignore[attr-defined]
        return dict(budgets)

    monkeypatch.setattr(cbi, "compute_prepare_pipeline_component_budgets", _persisted_compute)

    cbi._refresh_bootstrap_wait_timeouts()

    aggregated_timeout = cbi._BOOTSTRAP_WAIT_TIMEOUT
    assert aggregated_timeout is not None
    assert aggregated_timeout >= sum(budgets.values())

    state = json.loads(state_path.read_text())
    host_state = state.get(btp._state_host_key(), {})  # type: ignore[attr-defined]
    window_state = host_state.get("bootstrap_wait_windows", {}).get("general", {})

    assert window_state.get("component_total") >= sum(budgets.values())

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "60")
    monkeypatch.setattr(cbi, "compute_prepare_pipeline_component_budgets", lambda **_: {})

    cbi._refresh_bootstrap_wait_timeouts()

    assert cbi._BOOTSTRAP_WAIT_TIMEOUT is not None
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT >= sum(budgets.values())


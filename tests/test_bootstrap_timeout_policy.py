from __future__ import annotations

import importlib
import json
import os
import time

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


def test_component_budgets_scale_with_live_backlog(monkeypatch):
    monkeypatch.setattr(btp, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(btp, "_save_timeout_state", lambda state: None)
    monkeypatch.setattr(btp, "_load_timeout_state", lambda: {})
    monkeypatch.setattr(btp, "_recent_peer_activity", lambda max_age=900.0: [{"pid": 1, "ts": time.time()}])

    base_budgets = btp.compute_prepare_pipeline_component_budgets(
        load_average=1.0, host_telemetry={"ts": time.time()}
    )
    backlog_budgets = btp.compute_prepare_pipeline_component_budgets(
        load_average=1.0,
        host_telemetry={
            "ts": time.time(),
            "queue_depth": 3,
            "pending_background_loops": 4,
        },
    )

    assert sum(backlog_budgets.values()) > sum(base_budgets.values())


def test_component_budgets_include_defaults(monkeypatch):
    monkeypatch.setattr(btp, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(btp, "_save_timeout_state", lambda state: None)
    monkeypatch.setattr(btp, "_load_timeout_state", lambda: {})

    budgets = btp.compute_prepare_pipeline_component_budgets(
        telemetry={}, host_telemetry={"ts": time.time()}
    )

    expected_components = {
        "vectorizers": btp._COMPONENT_TIMEOUT_MINIMUMS["vectorizers"],
        "retrievers": btp._COMPONENT_TIMEOUT_MINIMUMS["retrievers"],
        "db_indexes": btp._COMPONENT_TIMEOUT_MINIMUMS["db_indexes"],
        "orchestrator_state": btp._COMPONENT_TIMEOUT_MINIMUMS["orchestrator_state"],
        "pipeline_config": btp._COMPONENT_TIMEOUT_MINIMUMS["pipeline_config"],
        "background_loops": btp._DEFERRED_COMPONENT_TIMEOUT_MINIMUMS["background_loops"],
    }

    for gate, minimum in expected_components.items():
        assert gate in budgets
        assert budgets[gate] >= minimum


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
        "retrievers": 360.0,
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
    assert aggregated_timeout >= sum(
        value for key, value in budgets.items() if key not in btp.DEFERRED_COMPONENTS
    )

    state = json.loads(state_path.read_text())
    host_state = state.get(btp._state_host_key(), {})  # type: ignore[attr-defined]
    window_state = host_state.get("bootstrap_wait_windows", {}).get("general", {})

    core_total = sum(value for key, value in budgets.items() if key not in btp.DEFERRED_COMPONENTS)
    assert window_state.get("component_total") >= core_total

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "60")
    monkeypatch.setattr(cbi, "compute_prepare_pipeline_component_budgets", lambda **_: {})

    cbi._refresh_bootstrap_wait_timeouts()

    assert cbi._BOOTSTRAP_WAIT_TIMEOUT is not None
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT + 15.0 >= sum(budgets.values())


def test_component_overruns_cover_all_taxonomy():
    telemetry = {
        "shared_timeout": {
            "timeline": [
                {"label": "vector service warmup", "elapsed": 75.0, "effective": 30.0},
                {
                    "label": "retriever handshake",
                    "elapsed": 65.0,
                    "effective": 25.0,
                    "components": ["retrievers"],
                },
                {
                    "label": "db index sync",
                    "elapsed": 55.0,
                    "effective": 20.0,
                    "component": "db_indexes",
                },
                {
                    "label": "orchestrator recovery",
                    "elapsed": 45.0,
                    "effective": 15.0,
                },
                {
                    "label": "pipeline config",
                    "elapsed": 35.0,
                    "effective": 10.0,
                },
                {
                    "label": "background scheduler loop",
                    "elapsed": 30.0,
                    "effective": 8.0,
                },
            ]
        }
    }

    overruns = btp._summarize_component_overruns(telemetry)

    expected_by_gate = {
        "vectorizers": 30.0,
        "retrievers": 25.0,
        "db_indexes": 20.0,
        "orchestrator_state": 15.0,
        "pipeline_config": 10.0,
        "background_loops": 8.0,
    }

    for gate, effective in expected_by_gate.items():
        assert gate in overruns
        assert overruns[gate]["expected_floor"] >= effective


def test_guard_scale_only_inflates_overrun_components(monkeypatch):
    floors = {
        "vectorizers": 120.0,
        "retrievers": 90.0,
    }

    monkeypatch.setattr(btp, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(btp, "_save_timeout_state", lambda state: None)
    monkeypatch.setattr(btp, "_load_timeout_state", lambda: {})
    monkeypatch.setattr(btp, "_recent_peer_activity", lambda max_age=900.0: [])
    monkeypatch.setattr(btp, "get_bootstrap_guard_context", lambda: {"budget_scale": 2.0})

    telemetry = {
        "shared_timeout": {
            "timeline": [
                {"label": "vectorizer warmup", "elapsed": 180.0, "effective": 90.0},
            ]
        }
    }

    budgets = btp.compute_prepare_pipeline_component_budgets(
        component_floors=floors,
        telemetry=telemetry,
        load_average=0.0,
        host_telemetry={},
    )

    assert budgets["vectorizers"] >= floors["vectorizers"] * 2
    assert budgets["retrievers"] >= btp._COMPONENT_TIMEOUT_MINIMUMS["retrievers"]  # type: ignore[attr-defined]
    assert budgets["retrievers"] < budgets["vectorizers"]


def test_bootstrap_wait_persists_component_pools(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(state_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_BACKGROUND_UNLIMITED", "1")
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))

    importlib.reload(btp)
    importlib.reload(cbi)

    pools = {
        "vectorizers": 400.0,
        "retrievers": 160.0,
        "background_loops": 600.0,
    }

    monkeypatch.setattr(cbi, "compute_prepare_pipeline_component_budgets", lambda: {})
    monkeypatch.setattr(cbi, "load_component_budget_pools", lambda: dict(pools))

    cbi._refresh_bootstrap_wait_timeouts()

    state = json.loads(state_path.read_text())
    host_state = state.get(btp._state_host_key(), {})  # type: ignore[attr-defined]
    window_state = host_state.get("bootstrap_wait_windows", {}).get("general", {})
    deadlines = window_state.get("meta.per_component_deadlines") or {}

    assert cbi._BOOTSTRAP_WAIT_TIMEOUT is not None
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT >= pools["vectorizers"]
    assert deadlines.get("background_loops") is None
    assert deadlines.get("vectorizers") >= pools["vectorizers"]
def test_timeout_hints_include_budget_clauses():
    components = {
        "vectorizers": {"budget": 60.0, "remaining": -5.0},
        "retrievers": {"budget": 45.0, "remaining": 0.0},
        "db_indexes": {"budget": 35.0, "remaining": 2.0},
        "orchestrator_state": {"budget": 25.0, "remaining": 1.0},
        "pipeline_config": {"budget": 20.0, "remaining": 5.0},
        "background_loops": {"budget": 15.0, "remaining": 7.5},
    }

    hints = btp.render_prepare_pipeline_timeout_hints(
        vector_heavy=False, components=components
    )

    for gate in components:
        gate_hints = [hint for hint in hints if gate in hint]
        assert gate_hints
        assert any("budget" in hint or "remaining" in hint for hint in gate_hints)


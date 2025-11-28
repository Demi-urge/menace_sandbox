import time

import bootstrap_timeout_policy
from bootstrap_timeout_policy import SharedTimeoutCoordinator


def test_component_budgets_not_constrained_by_global_window():
    component_budgets = {
        "vectorizers": 1.2,
        "retrievers": 0.9,
        "db_indexes": 0.8,
        "orchestrator_state": 0.7,
    }

    coordinator = SharedTimeoutCoordinator(
        total_budget=None,
        component_floors={gate: 0.5 for gate in component_budgets},
        component_budgets=component_budgets,
    )

    windows = coordinator.start_component_timers(component_budgets, minimum=0.5)

    # Each stage should preserve its own budget instead of competing for a single
    # shared deadline; record staggered progress to validate the per-gate timers
    # remain intact.
    for gate, window in windows.items():
        assert window["budget"] == component_budgets[gate]
        coordinator.record_progress(
            gate,
            elapsed=0.1,
            remaining=max(0.0, component_budgets[gate] - 0.1),
            metadata={"test_gate": gate},
        )
        time.sleep(0.01)

    snapshot = coordinator.snapshot()

    assert snapshot["remaining_budget"] >= sum(component_budgets.values())
    for gate, window in snapshot["component_windows"].items():
        assert window["remaining"] <= component_budgets[gate]


def test_overlapping_component_windows_extend_deadline(monkeypatch):
    component_budgets = {
        "vectorizers": 1.5,
        "retrievers": 1.0,
    }

    monkeypatch.setattr(bootstrap_timeout_policy, "_host_load_average", lambda: 1.0)

    coordinator = SharedTimeoutCoordinator(
        total_budget=1.0,
        component_floors={gate: 0.5 for gate in component_budgets},
        component_budgets=component_budgets,
    )

    windows = coordinator.start_component_timers(component_budgets, minimum=0.5)
    assert set(windows) == set(component_budgets)

    snapshot = coordinator.snapshot()
    assert snapshot["expanded_global_window"] >= sum(component_budgets.values())
    assert snapshot["deadline_extensions"]
    assert snapshot["deadline_extensions"][0]["gate"] in component_budgets

    for gate, window in snapshot["component_windows"].items():
        assert window["remaining"] == component_budgets[gate]


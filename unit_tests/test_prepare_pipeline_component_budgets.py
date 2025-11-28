import time

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

    assert snapshot["remaining_budget"] is None
    for gate, window in snapshot["component_windows"].items():
        assert window["remaining"] <= component_budgets[gate]


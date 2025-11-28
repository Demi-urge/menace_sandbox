import bootstrap_timeout_policy
from bootstrap_readiness import minimal_online
from bootstrap_timeout_policy import SharedTimeoutCoordinator


def test_minimal_online_accepts_quorum_without_full_completion(monkeypatch):
    monkeypatch.delenv("MENACE_DEGRADED_CORE_QUORUM", raising=False)
    state = {
        "components": {
            "vector_seeding": "ready",
            "retriever_hydration": "warming",
            "db_index_load": "pending",
        }
    }

    ready, lagging, degraded, degraded_online = minimal_online(state)

    assert ready
    assert degraded_online
    assert lagging == {"db_index_load"}
    assert degraded == {"retriever_hydration"}


def test_timeout_coordinator_extends_window_under_load(monkeypatch):
    monkeypatch.setattr(bootstrap_timeout_policy, "_host_load_average", lambda: 2.5)
    coordinator = SharedTimeoutCoordinator(
        10.0,
        component_budgets={"vectorizers": 5.0, "retrievers": 5.0},
        component_floors={"vectorizers": 3.0, "retrievers": 3.0},
    )

    coordinator.start_component_timers({"vectorizers": 5.0, "retrievers": 5.0})

    snapshot = coordinator.snapshot()
    assert snapshot["global_window"] > 10.0
    assert snapshot["expanded_global_window"] > 10.0
    assert snapshot["deadline_extensions"]


def test_readiness_promotes_when_straggler_finishes(monkeypatch):
    monkeypatch.setenv("MENACE_DEGRADED_CORE_QUORUM", "2")
    state = {
        "components": {
            "vector_seeding": "ready",
            "retriever_hydration": "ready",
            "db_index_load": "pending",
        }
    }

    ready, lagging, degraded, degraded_online = minimal_online(state)
    assert ready
    assert degraded_online
    assert lagging == {"db_index_load"}

    state["components"]["db_index_load"] = "ready"
    ready, lagging, degraded, degraded_online = minimal_online(state)

    assert ready
    assert not lagging
    assert not degraded_online
    assert degraded == set()

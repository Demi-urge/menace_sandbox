from bootstrap_readiness import CORE_COMPONENTS, minimal_online


def test_minimal_online_marks_degraded_online(monkeypatch):
    monkeypatch.setenv("MENACE_DEGRADED_CORE_QUORUM", "1")
    state = {
        "components": {
            "vector_seeding": "warming",
            "retriever_hydration": "pending",
            "db_index_load": "pending",
        }
    }

    ready, lagging, degraded, degraded_online = minimal_online(state)

    assert ready
    assert degraded_online
    assert degraded == {"vector_seeding"}
    assert "retriever_hydration" in lagging
    assert "db_index_load" in lagging


def test_minimal_online_respects_quorum_override(monkeypatch):
    monkeypatch.setenv("MENACE_DEGRADED_CORE_QUORUM", str(len(CORE_COMPONENTS)))
    state = {"components": {component: "warming" for component in CORE_COMPONENTS}}

    ready, lagging, degraded, degraded_online = minimal_online(state)

    assert ready
    assert not degraded_online
    assert degraded == set(CORE_COMPONENTS)
    assert not lagging

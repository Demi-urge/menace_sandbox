import threading
import time

import start_autonomous_sandbox as sas


class _StubThread:
    def __init__(self) -> None:
        self._alive = True
        self.name = "stub"
        self.ident = 123

    def is_alive(self) -> bool:
        return self._alive

    def join(self, _timeout: float) -> None:
        self._alive = False


def test_monitor_bootstrap_allows_degraded_online(monkeypatch):
    sas.SHUTDOWN_EVENT.clear()
    sas.BOOTSTRAP_PROGRESS["last_step"] = "context_builder"
    sas.BOOTSTRAP_ONLINE_STATE["components"] = {
        "vector_seeding": "warming",
        "retriever_hydration": "pending",
        "db_index_load": "pending",
    }
    monkeypatch.setenv("MENACE_DEGRADED_CORE_QUORUM", "1")

    stub_thread = _StubThread()
    timed_out, _, _, _, _, _, _, stage_timeout_context = sas._monitor_bootstrap_thread(
        bootstrap_thread=stub_thread,
        bootstrap_stop_event=threading.Event(),
        bootstrap_start=time.monotonic(),
        step_budgets={"context_builder": 0.0},
        adaptive_grace={},
        completed_steps=set(),
        vector_heavy_steps=set(),
        stage_policy={
            "db_index_load": {
                "deadline": 0.0,
                "optional": False,
                "enforced": True,
            }
        },
    )

    assert not timed_out
    assert sas.BOOTSTRAP_ONLINE_STATE.get("core_degraded_online")
    assert stage_timeout_context
    assert stage_timeout_context.get("stage") == "db_index_load"

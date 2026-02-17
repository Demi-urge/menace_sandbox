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


def test_self_debug_guardian_expected_misuse_is_bounded(monkeypatch):
    monkeypatch.setenv("FAILURE_MODES", "user_misuse")
    sas._SELF_DEBUG_EXPECTED_MISUSE_TELEMETRY.update(
        {"events": 0.0, "checks": 0.0, "ratio": 0.0}
    )

    checks: list[int] = []

    def expected_misuse_condition(_logger):
        checks.append(1)
        return (
            False,
            "workflow_discovery_failed",
            {"scenario": "user_misuse", "detail": "simulated_user_misuse_len_probe"},
        )

    run_calls: list[int] = []
    monkeypatch.setattr(sas, "_run_self_debug_once", lambda *_a, **_k: run_calls.append(1) or 0)

    ok = sas.run_self_debug_until_stable(
        settings=sas.load_sandbox_settings(),
        logger=sas.logger,
        max_attempts=5,
        backoff=0.0,
        cadence=0.0,
        success_conditions=[expected_misuse_condition],
    )

    assert ok is True
    assert len(checks) == 1
    assert run_calls == []
    assert sas._SELF_DEBUG_EXPECTED_MISUSE_TELEMETRY["events"] >= 1.0


def test_self_debug_health_loop_expected_misuse_skips_retry(monkeypatch):
    monkeypatch.setenv("FAILURE_MODES", "user_misuse")
    sas._SELF_DEBUG_EXPECTED_MISUSE_TELEMETRY.update(
        {"events": 0.0, "checks": 0.0, "ratio": 0.0}
    )

    stop_event = threading.Event()

    def expected_misuse_check(_logger):
        stop_event.set()
        return False, "module_resolution_failed", {"scenario": "user_misuse", "reason": "misuse"}

    retries: list[int] = []
    monkeypatch.setattr(
        sas,
        "run_self_debug_until_stable",
        lambda **_kwargs: retries.append(1) or False,
    )

    sas._self_debug_health_loop(
        settings=sas.load_sandbox_settings(),
        logger=sas.logger,
        stop_event=stop_event,
        checks=[expected_misuse_check],
        cadence=0.0,
        max_attempts=3,
        backoff=0.0,
    )

    assert retries == []

import logging
import os
import threading
import time

import pytest

import coding_bot_interface as cbi
import bootstrap_timeout_policy as btp


class _DummyPipeline:
    def __init__(self, **_kwargs: object) -> None:
        return


def test_prepare_timeout_logs_standard_hints(monkeypatch, caplog):
    stop_event = threading.Event()
    stop_event.set()

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "30")
    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})

    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        with pytest.raises(TimeoutError) as excinfo:
            cbi._prepare_pipeline_for_bootstrap_impl(
                pipeline_cls=_DummyPipeline,
                context_builder=object(),
                bot_registry=object(),
                data_bot=object(),
                stop_event=stop_event,
                timeout=30.0,
                deadline=time.perf_counter() + 30.0,
            )

    combined_output = caplog.text + str(excinfo.value)

    assert "MENACE_BOOTSTRAP_WAIT_SECS=" in combined_output
    assert "BOOTSTRAP_VECTOR_STEP_TIMEOUT=" in combined_output
    assert "Stagger concurrent bootstraps" in combined_output


def test_bootstrap_policy_refreshes_cached_wait_timeout(monkeypatch, caplog):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "600")
    cbi._refresh_bootstrap_wait_timeouts()
    minimum_component_window = sum(
        value
        for key, value in btp._BOOTSTRAP_TIMEOUT_MINIMUMS.items()
        if key.startswith("PREPARE_PIPELINE_") and key.endswith("_BUDGET_SECS")
    )
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT >= minimum_component_window

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "30")
    stop_event = threading.Event()
    stop_event.set()

    class _Pipeline:
        def __init__(self, **_kwargs: object) -> None:
            return

    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        with pytest.raises(TimeoutError):
            cbi._prepare_pipeline_for_bootstrap_impl(
                pipeline_cls=_Pipeline,
                context_builder=object(),
                bot_registry=object(),
                data_bot=object(),
                stop_event=stop_event,
                timeout=0.0,
            )

    assert float(os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")) >= cbi._BOOTSTRAP_TIMEOUT_FLOOR
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT >= cbi._BOOTSTRAP_TIMEOUT_FLOOR
    assert cbi._BOOTSTRAP_WAIT_TIMEOUT >= minimum_component_window
    assert cbi._resolve_bootstrap_wait_timeout(False) == cbi._BOOTSTRAP_WAIT_TIMEOUT


def test_bootstrap_stage_floors_clamp_to_policy_minimums(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "30")
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "45")
    monkeypatch.setenv("BOOTSTRAP_STEP_TIMEOUT", "60")
    monkeypatch.setenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT", "90")
    monkeypatch.setenv(
        "PREPARE_PIPELINE_STAGE_BUDGETS",
        "vectorizers:1,retrievers:1,db_indexes:1,orchestrator_state:1,pipeline_config:1",
    )
    monkeypatch.setenv(
        "PREPARE_PIPELINE_VECTOR_STAGE_BUDGETS",
        "vectorizers:2,retrievers:2,db_indexes:2,orchestrator_state:2,pipeline_config:2",
    )

    cbi._refresh_bootstrap_wait_timeouts()

    minima = btp._BOOTSTRAP_TIMEOUT_MINIMUMS
    component_minima = btp._COMPONENT_TIMEOUT_MINIMUMS

    assert cbi._BOOTSTRAP_TIMEOUT_FLOOR >= minima["MENACE_BOOTSTRAP_WAIT_SECS"]
    assert cbi._VECTOR_BOOTSTRAP_TIMEOUT_FLOOR >= minima["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"]
    assert cbi._MIN_STAGE_TIMEOUT >= minima["BOOTSTRAP_STEP_TIMEOUT"]
    assert cbi._MIN_STAGE_TIMEOUT_VECTOR >= minima["BOOTSTRAP_VECTOR_STEP_TIMEOUT"]

    for gate, minimum in component_minima.items():
        assert cbi._PREPARE_STAGE_BUDGETS[gate] >= minimum
        assert cbi._PREPARE_VECTOR_STAGE_BUDGETS[gate] >= minimum


def test_shared_timeout_records_stage_gates(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "90")
    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})

    class _VectorHeavyPipeline:
        vector_bootstrap_heavy = True

        def __init__(self, **_kwargs: object) -> None:
            return

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_VectorHeavyPipeline,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        vectorizer_budget=5.0,
        retriever_budget=3.0,
        db_warmup_budget=2.0,
    )

    promote(object())

    shared = cbi._PREPARE_PIPELINE_WATCHDOG.get("shared_timeout", {})
    timeline = shared.get("timeline", [])

    assert timeline
    assert any(
        "vectorizers" in entry.get("meta.gates", ())
        or entry.get("label") == "vectorizers"
        for entry in timeline
    ), timeline
    assert any(
        entry.get("label") in {"pipeline_component_warmup", "orchestrator_state_warmup"}
        for entry in timeline
    )


def test_optional_budget_overrun_degrades_instead_of_fails(monkeypatch):
    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})
    monkeypatch.setattr(cbi, "_MIN_STAGE_TIMEOUT", 0.0)
    monkeypatch.setattr(cbi, "_MIN_STAGE_TIMEOUT_VECTOR", 0.0)
    monkeypatch.setattr(cbi, "_BOOTSTRAP_TIMEOUT_FLOOR", 0.0)
    monkeypatch.setattr(cbi, "_BOOTSTRAP_WAIT_TIMEOUT", 0.0)
    monkeypatch.setattr(cbi, "_VECTOR_STAGE_GRACE_PERIOD", 0.0)
    monkeypatch.setattr(
        cbi,
        "_derive_readiness_gates",
        lambda _label, _gates=None: {"vectorizers"},
    )

    def _force_timeout_deadline(deadline, *, start_time, vector_heavy, stage_label=None):
        telemetry = {
            "ready_gates": [],
            "pending_gates": ["vectorizers"],
            "readiness_ratio": 0.0,
        }
        return start_time, 0.0, telemetry

    monkeypatch.setattr(cbi, "_normalize_watchdog_timeout", _force_timeout_deadline)

    class _Pipeline:
        def __init__(self, **_kwargs: object) -> None:
            return

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        component_timeouts={"vectorizers": 0.0},
    )

    promote(object())

    degraded = cbi._PREPARE_PIPELINE_WATCHDOG.get("degraded") or {}
    assert degraded.get("degraded_reason") in {
        "optional_phase_overrun",
        "optional_window_exhausted",
    }


def test_overlapping_optional_gates_defer_before_watchdog(monkeypatch):
    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})
    monkeypatch.setattr(cbi, "_MIN_STAGE_TIMEOUT", 0.0)
    monkeypatch.setattr(cbi, "_MIN_STAGE_TIMEOUT_VECTOR", 0.0)
    monkeypatch.setattr(cbi, "_BOOTSTRAP_TIMEOUT_FLOOR", 0.0)
    monkeypatch.setattr(cbi, "_BOOTSTRAP_WAIT_TIMEOUT", 0.0)
    monkeypatch.setattr(cbi, "_VECTOR_STAGE_GRACE_PERIOD", 0.0)
    monkeypatch.setattr(
        cbi,
        "_derive_readiness_gates",
        lambda _label, _gates=None: {"vectorizers", "retrievers"},
    )

    def _force_timeout_deadline(deadline, *, start_time, vector_heavy, stage_label=None):
        telemetry = {
            "ready_gates": [],
            "pending_gates": ["vectorizers", "retrievers"],
            "readiness_ratio": 0.0,
        }
        return start_time, 0.0, telemetry

    monkeypatch.setattr(cbi, "_normalize_watchdog_timeout", _force_timeout_deadline)

    class _Pipeline:
        def __init__(self, **_kwargs: object) -> None:
            return

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        component_timeouts={"vectorizers": 0.0, "retrievers": 0.0},
    )

    promote(object())

    deferred = cbi._PREPARE_PIPELINE_WATCHDOG.get("deferred") or []
    assert deferred, "expected overlapping optional gates to be deferred"
    deferred_gates = {
        gate for payload in deferred for gate in payload.get("deferred_gates", ())
    }
    assert {"vectorizers", "retrievers"} <= deferred_gates

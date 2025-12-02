import time
import types

import bootstrap_timeout_policy
import vector_metrics_db


def test_vector_metrics_db_stubbed_when_bootstrap_timer_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "600")

    class _SentinelRouter:
        def resolve_path(self, *_, **__):  # pragma: no cover - should not be called
            raise AssertionError("path resolution should be skipped during bootstrap timers")

        def get_project_root(self):
            return tmp_path

    def _sentinel_db_router():  # pragma: no cover - should not be called
        raise AssertionError("router should not be initialised during bootstrap timers")

    monkeypatch.setattr(vector_metrics_db, "_dynamic_path_router", lambda: _SentinelRouter())
    monkeypatch.setattr(vector_metrics_db, "_db_router", _sentinel_db_router)

    start = time.perf_counter()
    vm = vector_metrics_db.VectorMetricsDB(tmp_path / "vm.db")
    elapsed = time.perf_counter() - start

    assert vm._boot_stub_active
    assert vm._conn is vm._stub_conn
    assert not (tmp_path / "vm.db").exists()
    assert elapsed < 0.1


def test_timeout_policy_sets_vector_metrics_warmup(monkeypatch):
    monkeypatch.delenv("VECTOR_METRICS_BOOTSTRAP_WARMUP", raising=False)

    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "bootstrap_manager",
        types.SimpleNamespace(run_once=lambda *_args, **_kwargs: _args[1]()),
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_collect_timeout_telemetry",
        lambda: {"env": {}},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "load_escalated_timeout_floors",
        lambda: dict(bootstrap_timeout_policy._BOOTSTRAP_TIMEOUT_MINIMUMS),
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "load_component_timeout_floors",
        lambda: {},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "compute_prepare_pipeline_component_budgets",
        lambda component_floors, telemetry: {},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_summarize_component_overruns",
        lambda telemetry: {},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_load_timeout_state",
        lambda: {},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_save_timeout_state",
        lambda state: None,
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_merge_consumption_overruns",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_apply_success_decay",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_maybe_escalate_timeout_floors",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_derive_timeout_setting",
        lambda env_name, effective_minimum, value, allow_unsafe, state_value, prompt_override, logger: {
            "value": effective_minimum
        },
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_component_timeout_setting",
        lambda component, telemetry_value, floor, allow_unsafe, logger: {"value": floor},
    )
    monkeypatch.setattr(
        bootstrap_timeout_policy,
        "_summarize_component_telemetry",
        lambda telemetry: {},
    )

    bootstrap_timeout_policy.enforce_bootstrap_timeout_policy()

    assert bootstrap_timeout_policy.os.getenv("VECTOR_METRICS_BOOTSTRAP_WARMUP") == "1"

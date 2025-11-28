import threading

import bootstrap_timeout_policy
import environment_bootstrap


class _StubInfrastructureBootstrapper:
    def __init__(self, *_args, **_kwargs) -> None:
        return

    def bootstrap(self) -> None:
        return None

    def run_continuous(self, interval: float, stop_event: threading.Event) -> threading.Thread:
        return threading.Thread()


class _StubPolicy:
    name = "stub"
    required_commands: list[str] = []
    additional_python_dependencies: list[str] = []
    enforce_systemd = False
    enforce_remote_checks = False
    run_database_migrations = False
    provision_vector_assets = True
    ensure_apscheduler = False


class _LightweightBootstrapper(environment_bootstrap.EnvironmentBootstrapper):
    def __init__(self, *args, **kwargs) -> None:
        self.optional_release = threading.Event()
        self.optional_started = threading.Event()
        super().__init__(*args, **kwargs)

    def _critical_prerequisites(self, check_budget, phase_budget=None) -> None:
        check_budget()
        if phase_budget:
            phase_budget.mark_online_ready(reason="critical-core")

    def _provisioning_phase(self, check_budget, phase_budget=None, *, skip_db_init: bool) -> None:  # type: ignore[override]
        check_budget()
        if phase_budget:
            phase_budget.mark_online_ready(reason="provisioning-core")

    def _optional_tail(self, check_budget) -> None:
        self.optional_started.set()
        self.optional_release.wait(timeout=1)
        check_budget()



def test_partial_then_full_readiness(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))
    monkeypatch.setattr(
        environment_bootstrap, "InfrastructureBootstrapper", _StubInfrastructureBootstrapper
    )

    bootstrapper = _LightweightBootstrapper(policy=_StubPolicy())
    bootstrapper.bootstrap(halt_background=False)

    assert bootstrapper._phase_readiness.get("online") is True
    assert bootstrapper._phase_readiness.get("full_ready") is False

    assert bootstrapper.optional_started.wait(timeout=1)

    heartbeat = bootstrap_timeout_policy.read_bootstrap_heartbeat(max_age=10)
    assert heartbeat is not None
    assert heartbeat.get("status") in {"ready", "start"}

    bootstrapper.optional_release.set()
    for thread in list(bootstrapper._background_threads):
        thread.join(timeout=1)

    assert bootstrapper.readiness_state().get("full_ready") is True
    final_heartbeat = bootstrap_timeout_policy.read_bootstrap_heartbeat(max_age=10)
    assert final_heartbeat is not None
    assert final_heartbeat.get("phase") in {"optional", "full_ready"}

    bootstrapper.shutdown()

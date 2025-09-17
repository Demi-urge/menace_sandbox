import logging
import os
import types
from pathlib import Path

import pytest

os.environ["MENACE_LOCAL_DB_PATH"] = "/tmp/menace_local.db"
os.environ["MENACE_SHARED_DB_PATH"] = "/tmp/menace_shared.db"

import sandbox_runner
import menace_sandbox.service_supervisor as ss


class DummyBuilder:
    def refresh_db_weights(self) -> None:
        pass


class DummyManager:
    def __init__(self) -> None:
        self.summary_payload: dict[str, object] = {}
        self.context_builder = None
        self.bot_name = "ServiceSupervisor"
        self.engine = types.SimpleNamespace(last_added_modules=[], added_modules=[])

    def auto_run_patch(self, path: Path, description: str):
        return {
            "summary": self.summary_payload,
            "patch_id": 1,
            "commit": "abc123",
            "result": None,
        }


@pytest.fixture(autouse=True)
def _stub_sandbox_runner(monkeypatch):
    monkeypatch.setattr(
        sandbox_runner,
        "try_integrate_into_workflows",
        lambda *a, **k: None,
        raising=False,
    )


def test_deploy_patch_requires_self_tests(monkeypatch, tmp_path):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.context_builder = DummyBuilder()
    supervisor.approval_policy = types.SimpleNamespace()
    supervisor.rollback_mgr = types.SimpleNamespace(auto_rollback=lambda *a, **k: None)
    logger = logging.getLogger("ServiceSupervisorTest")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    supervisor.logger = logger

    manager = DummyManager()

    monkeypatch.setattr(ss, "SelfCodingEngine", lambda *a, **k: manager.engine, raising=False)
    monkeypatch.setattr(ss, "ModelAutomationPipeline", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "DataBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "CapitalManagementBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        ss,
        "EvolutionOrchestrator",
        lambda *a, **k: types.SimpleNamespace(provenance_token="token"),
        raising=False,
    )
    monkeypatch.setattr(ss, "internalize_coding_bot", lambda *a, **k: manager, raising=False)
    monkeypatch.setattr(
        ss,
        "get_thresholds",
        lambda name: types.SimpleNamespace(
            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
        ),
        raising=False,
    )
    monkeypatch.setattr(ss, "persist_sc_thresholds", lambda *a, **k: None, raising=False)

    ss.bus = types.SimpleNamespace()
    ss.registry = types.SimpleNamespace()
    ss.data_bot = types.SimpleNamespace()

    target = tmp_path / "svc.py"
    target.write_text("print('hi')\n")

    with pytest.raises(RuntimeError, match="self test summary unavailable"):
        supervisor.deploy_patch(target, "desc")

    manager.summary_payload = {"self_tests": {"failed": 0}}
    supervisor.deploy_patch(target, "desc")

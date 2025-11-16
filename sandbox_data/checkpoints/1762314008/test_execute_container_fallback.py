import asyncio
import os
import sys
import types

import sandbox_runner.environment as env


def _stub_docker_fail_from_env(exc_cls):
    dummy = types.ModuleType("docker")
    dummy.errors = types.ModuleType("docker.errors")
    dummy.errors.DockerException = exc_cls
    dummy.errors.APIError = exc_cls
    def from_env():
        raise exc_cls("fail")
    dummy.from_env = from_env
    sys.modules["docker"] = dummy
    sys.modules["docker.errors"] = dummy.errors


def _stub_docker_retry_fail(exc_cls, attempts_rec):
    class DummyContainers:
        def run(self, *a, **k):
            attempts_rec.append(1)
            raise exc_cls("boom")

    class DummyClient:
        def __init__(self):
            self.containers = DummyContainers()
    dummy = types.ModuleType("docker")
    dummy.errors = types.ModuleType("docker.errors")
    dummy.errors.DockerException = exc_cls
    dummy.errors.APIError = exc_cls
    dummy.from_env = lambda: DummyClient()
    sys.modules["docker"] = dummy
    sys.modules["docker.errors"] = dummy.errors


def test_from_env_failure_local_exec(monkeypatch):
    class DummyErr(Exception):
        pass

    _stub_docker_fail_from_env(DummyErr)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)

    calls = []
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: calls.append((issue, success)))

    res = asyncio.run(env._execute_in_container("print('hi')", {}))
    assert res["exit_code"] == 0.0
    assert "container_error" in res
    assert not os.path.exists(res["stdout_log"])
    assert not os.path.exists(res["stderr_log"])
    assert calls == []


def test_retry_limit_backoff_and_fallback(monkeypatch):
    class DummyErr(Exception):
        pass

    attempts = []
    _stub_docker_retry_fail(DummyErr, attempts)

    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 3)
    monkeypatch.setattr(env, "_CREATE_BACKOFF_BASE", 1.0)

    sleeps = []
    monkeypatch.setattr(env.time, "sleep", lambda d: sleeps.append(d))

    logs = []
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.append((issue, success)))

    res = asyncio.run(env._execute_in_container("print('x')", {"CPU_LIMIT": "1"}))

    assert res["exit_code"] == 0.0
    assert "container_error" in res
    assert sleeps == [2.0, 4.0]
    assert logs[-1] == ("local_fallback", True)
    assert len([l for l in logs if not l[1]]) == 3
    assert not os.path.exists(res["stdout_log"])
    assert not os.path.exists(res["stderr_log"])


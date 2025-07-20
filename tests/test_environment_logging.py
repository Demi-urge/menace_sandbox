import os
import sys
import types
import logging
import asyncio
import builtins

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
dummy_jinja = types.ModuleType("jinja2")
dummy_jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", dummy_jinja)
dummy_yaml = types.ModuleType("yaml")
dummy_yaml.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", dummy_yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
sqlalchemy_mod.Boolean = sqlalchemy_mod.Column = sqlalchemy_mod.Float = lambda *a, **k: None
sqlalchemy_mod.ForeignKey = sqlalchemy_mod.Integer = sqlalchemy_mod.MetaData = lambda *a, **k: None
sqlalchemy_mod.String = sqlalchemy_mod.Table = sqlalchemy_mod.Text = lambda *a, **k: None
sqlalchemy_mod.create_engine = lambda *a, **k: None
engine_mod.Engine = object
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
flask_stub = types.ModuleType("flask")
class _DummyFlask:
    def add_url_rule(self, *a, **k):
        pass
    def run(self, host="0.0.0.0", port=0):
        pass
flask_stub.Flask = lambda *a, **k: _DummyFlask()
flask_stub.jsonify = lambda *a, **k: {}
sys.modules.setdefault("flask", flask_stub)
menace_pkg = types.ModuleType("menace")
md_mod = types.ModuleType("menace.metrics_dashboard")
md_mod.MetricsDashboard = lambda *a, **k: object()
menace_pkg.metrics_dashboard = md_mod
sys.modules.setdefault("menace", menace_pkg)
sys.modules.setdefault("menace.metrics_dashboard", md_mod)

import sandbox_runner.environment as env

class DummyDiag:
    def __init__(self):
        self.log = types.SimpleNamespace(add=self.bad)
        self.error_bot = types.SimpleNamespace(handle_error=lambda x: None)
    def bad(self, *a, **k):
        raise RuntimeError("boom")


def _stub_docker(image_holder):
    class DummyExec:
        def __init__(self, code=0):
            self.exit_code = code

    class DummyContainer:
        id = "dummy"
        def exec_run(self, *a, **k):
            return DummyExec(0)
        def wait(self):
            return {"StatusCode": 0}
        def stats(self, stream=False):
            return {
                "blkio_stats": {"io_service_bytes_recursive": []},
                "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                "memory_stats": {"max_usage": 1},
            }
        def remove(self):
            image_holder.append("removed")
        def stop(self, timeout=0):
            pass
    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            image_holder.append(image)
            return DummyContainer()
    class DummyClient:
        containers = DummyContainers()
    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient()
    dummy.types = types
    sys.modules["docker"] = dummy


def test_log_diagnostic_logs(monkeypatch, caplog):
    monkeypatch.setattr(env, "_DIAGNOSTIC", DummyDiag())
    caplog.set_level(logging.ERROR)
    env._log_diagnostic("x", False)
    assert "diagnostic logging failed" in caplog.text


def test_execute_local_rlimit_warning(monkeypatch, caplog):
    original_import = builtins.__import__
    def fake_import(name, *a, **kw):
        if name == "docker":
            raise ImportError("no docker")
        return original_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: True)
    monkeypatch.setattr(env.resource, "setrlimit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    msgs = []

    monkeypatch.setattr(env.logger, "warning", lambda msg, *a, **k: msgs.append(msg % a))

    class DummyPopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=None, env=None, cwd=None, preexec_fn=None):
            if preexec_fn:
                preexec_fn()
            self.pid = 1
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""
        def communicate(self, timeout=None):
            return ("", "")
        def kill(self):
            pass

    monkeypatch.setattr(env.subprocess, "Popen", DummyPopen)
    asyncio.run(env._execute_in_container("print('hi')", {"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}))
    assert any("failed to set CPU limit" in m or "failed to set memory limit" in m for m in msgs)


def test_execute_in_container_invalid_limits(monkeypatch, caplog):
    calls = []
    _stub_docker(calls)
    caplog.set_level(logging.WARNING)
    asyncio.run(env._execute_in_container("print('x')", {"CPU_LIMIT": "bad", "GPU_LIMIT": "bad"}))
    assert "invalid CPU limit" in caplog.text
    assert "GPU limit ignored" in caplog.text


def test_section_worker_limits_logging(monkeypatch, caplog):
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: True)
    monkeypatch.setattr(env.resource, "setrlimit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="", stderr="", returncode=0))
    msgs = []
    monkeypatch.setattr(env.logger, "warning", lambda msg, *a, **k: msgs.append(msg % a))

    def dummy_run(cmd, capture_output=False, text=False, env=None, timeout=None, preexec_fn=None):
        if preexec_fn:
            preexec_fn()
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(env.subprocess, "run", dummy_run)
    res, _ = asyncio.run(
        env._section_worker("print('x')", {"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}, 0.0)
    )
    assert res["exit_code"] == 0
    assert any("failed to set CPU limit" in m or "failed to set memory limit" in m for m in msgs)

import importlib.util
import pytest
import sys
import types
from pathlib import Path
import subprocess

if importlib.util.find_spec("cryptography") is None:
    pytest.skip("optional dependencies not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]

# Stub heavy optional dependencies
jinja_stub = types.ModuleType("jinja2")
jinja_stub.Template = object
sys.modules.setdefault("jinja2", jinja_stub)
env_mod = types.ModuleType("env_config")
env_mod.DATABASE_URL = "sqlite:///tmp.db"
sys.modules.setdefault("env_config", env_mod)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("networkx", types.ModuleType("networkx"))

# Create minimal menace package with required modules only
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

ss_mod = types.ModuleType("menace.service_supervisor")


class ServiceSupervisor:
    def __init__(
        self,
        check_interval: float = 5.0,
        *,
        context_builder=None,
    ) -> None:
        self.check_interval = check_interval
        self.context_builder = context_builder
        if context_builder and hasattr(context_builder, "refresh_db_weights"):
            context_builder.refresh_db_weights()

    def start_all(self) -> None:
        pass

    def _monitor(self) -> None:
        pass


ss_mod.ServiceSupervisor = ServiceSupervisor
sys.modules["menace.service_supervisor"] = ss_mod

spec_cs = importlib.util.spec_from_file_location(
    "menace.cluster_supervisor", ROOT / "cluster_supervisor.py"  # path-ignore
)
cs = importlib.util.module_from_spec(spec_cs)
spec_cs.loader.exec_module(cs)
sys.modules["menace.cluster_supervisor"] = cs

import menace.cluster_supervisor as cs
import menace.service_supervisor as ss


def test_cluster_supervisor_start(monkeypatch):
    calls = []

    class DummyProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(cmd)
        return DummyProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    sup = cs.ClusterServiceSupervisor(hosts=["h1"], context_builder=builder)
    monkeypatch.setattr(ss.ServiceSupervisor, "start_all", lambda self: None)
    monkeypatch.setattr(ss.ServiceSupervisor, "_monitor", lambda self: None)
    sup.start_all()
    assert ["ssh", "h1", "python3", "-m", "menace.service_supervisor"] in calls


def test_cluster_supervisor_start_docker(monkeypatch):
    calls = []

    class DummyProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(cmd)
        return DummyProc()

    monkeypatch.setenv("CLUSTER_BACKEND", "docker")
    monkeypatch.setenv("CLUSTER_DOCKER_IMAGE", "img")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    sup = cs.ClusterServiceSupervisor(hosts=["h1"], context_builder=builder)
    monkeypatch.setattr(ss.ServiceSupervisor, "start_all", lambda self: None)
    monkeypatch.setattr(ss.ServiceSupervisor, "_monitor", lambda self: None)
    sup.start_all()
    assert [
        "ssh",
        "h1",
        "docker",
        "run",
        "-d",
        "--name",
        "menace_supervisor",
        "img",
    ] in calls


def test_cluster_supervisor_start_k8s(monkeypatch):
    calls = []

    class DummyProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(cmd)
        return DummyProc()

    monkeypatch.setenv("CLUSTER_BACKEND", "k8s")
    monkeypatch.setenv("CLUSTER_K8S_NAMESPACE", "ns")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    sup = cs.ClusterServiceSupervisor(hosts=["dep"], context_builder=builder)
    monkeypatch.setattr(ss.ServiceSupervisor, "start_all", lambda self: None)
    monkeypatch.setattr(ss.ServiceSupervisor, "_monitor", lambda self: None)
    sup.start_all()
    assert ["kubectl", "-n", "ns", "rollout", "restart", "deployment/dep"] in calls


def test_cluster_supervisor_redeploy_unhealthy(monkeypatch):
    calls = []

    class DummyProc:
        def __init__(self, rc):
            self.rc = rc

        def poll(self):
            return self.rc

    def fake_popen(cmd, stdout=None, stderr=None):
        return DummyProc(rc=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    sup = cs.ClusterServiceSupervisor(hosts=["h1"], context_builder=builder)
    sup.remote_procs["h1"] = DummyProc(rc=1)

    def fake_start(self, host):
        calls.append(host)

    monkeypatch.setattr(sup, "_start_remote", fake_start.__get__(sup))
    monkeypatch.setattr(ss.ServiceSupervisor, "_monitor", lambda self: None)
    for host in sup.hosts:
        if not sup._check_remote(host):
            sup._start_remote(host)
    assert calls == ["h1"]


def test_cluster_supervisor_add_hosts(monkeypatch):
    calls = []

    class DummyProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(cmd)
        return DummyProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    sup = cs.ClusterServiceSupervisor(hosts=["h1"], context_builder=builder)
    sup.add_hosts(["h2", "h3"])
    assert "h2" in sup.hosts and "h3" in sup.hosts
    assert ["ssh", "h2", "python3", "-m", "menace.service_supervisor"] in calls

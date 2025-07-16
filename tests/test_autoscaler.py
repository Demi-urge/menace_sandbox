import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys_modules = sys.modules
sys_modules.setdefault(
    "menace.environment_bootstrap",
    types.SimpleNamespace(EnvironmentBootstrapper=type("EB", (), {"deploy_across_hosts": lambda self, hosts: None})),
)
import autoscaler as a


def test_default_provider(monkeypatch):
    monkeypatch.delenv("AUTOSCALER_PROVIDER", raising=False)
    auto = a.Autoscaler(cooldown=0)
    assert isinstance(auto.provider, a.LocalProvider)


def test_kubernetes_provider(monkeypatch):
    monkeypatch.setenv("AUTOSCALER_PROVIDER", "kubernetes")
    auto = a.Autoscaler(cooldown=0)
    assert isinstance(auto.provider, a.KubernetesProvider)


def test_swarm_provider(monkeypatch):
    monkeypatch.setenv("AUTOSCALER_PROVIDER", "swarm")
    auto = a.Autoscaler(cooldown=0)
    assert isinstance(auto.provider, a.DockerSwarmProvider)


class DummyProvider(a.Provider):
    def __init__(self) -> None:
        self.up = 0
        self.down = 0

    def scale_up(self, amount: int = 1) -> bool:
        self.up += amount
        return True

    def scale_down(self, amount: int = 1) -> bool:
        self.down += amount
        return True


def test_scale_budget(monkeypatch):
    monkeypatch.setenv("BUDGET_MAX_INSTANCES", "1")
    import importlib
    import env_config
    importlib.reload(env_config)
    importlib.reload(a)
    provider = DummyProvider()
    auto = a.Autoscaler(provider=provider, cooldown=0)
    auto.scale_up()
    auto.scale_up()
    assert provider.up == 1


def test_scale_memory_and_budget(monkeypatch):
    monkeypatch.setenv("BUDGET_MAX_INSTANCES", "2")
    import importlib
    import env_config
    importlib.reload(env_config)
    importlib.reload(a)
    provider = DummyProvider()
    auto = a.Autoscaler(provider=provider, cooldown=0)
    auto.scale({"cpu": 0.1, "memory": 0.9})
    auto.scale({"cpu": 0.9, "memory": 0.1})
    auto.scale({"cpu": 0.9, "memory": 0.9})
    assert provider.up == 2


def test_scale_down_low_load():
    provider = DummyProvider()
    auto = a.Autoscaler(provider=provider, cooldown=0)
    auto.instances = 2
    auto.scale({"cpu": 0.1, "memory": 0.1})
    assert provider.down == 1
    assert auto.instances == 1


def test_scaling_policy():
    policy = a.ScalingPolicy(window=3, max_instances=3)
    for _ in range(3):
        policy.record({"cpu": 0.9, "memory": 0.9})
    assert policy.desired_replicas(1) == 2
    policy = a.ScalingPolicy(window=3, max_instances=3)
    for _ in range(3):
        policy.record({"cpu": 0.1, "memory": 0.1})
    assert policy.desired_replicas(2) == 1


def test_scale_up_bootstraps_new_hosts(monkeypatch):
    provider = DummyProvider()
    calls = []

    def fake_deploy(self, hosts):
        calls.append(list(hosts))

    monkeypatch.setenv("NEW_HOSTS", '["h1","h2"]')
    monkeypatch.setattr(a.EnvironmentBootstrapper, "deploy_across_hosts", fake_deploy)
    auto = a.Autoscaler(provider=provider, cooldown=0)
    auto.scale_up(2)
    assert provider.up == 2
    assert calls == [["h1", "h2"]]
    assert os.getenv("NEW_HOSTS") == '["h1","h2"]'

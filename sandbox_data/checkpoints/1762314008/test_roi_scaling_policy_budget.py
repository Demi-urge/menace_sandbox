import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import sys
import types
sys.modules.setdefault(
    "menace.environment_bootstrap",
    types.SimpleNamespace(EnvironmentBootstrapper=type("EB", (), {"deploy_across_hosts": lambda self, hosts: None})),
)
import autoscaler as a

sys.modules['menace.resource_prediction_bot'] = types.SimpleNamespace(ResourcePredictionBot=object)
import menace.roi_scaling_policy as rsp

class DummyPredictor:
    def predict_roi(self, horizon: int = 24):
        return {"roi": 1.0}

class DummyProvider(a.Provider):
    def __init__(self):
        self.up = 0
        self.down = 0

    def scale_up(self, amount: int = 1) -> bool:
        self.up += amount
        return True

    def scale_down(self, amount: int = 1) -> bool:
        self.down += amount
        return True


def test_roi_scaling_budget(monkeypatch):
    monkeypatch.setenv("BUDGET_MAX_INSTANCES", "1")
    import importlib
    import env_config
    importlib.reload(env_config)
    importlib.reload(a)
    provider = DummyProvider()
    policy = rsp.ROIScalingPolicy(
        predictor=DummyPredictor(), autoscaler=a.Autoscaler(provider=provider, cooldown=0)
    )
    policy.evaluate_and_scale()
    policy.evaluate_and_scale()
    assert provider.up == 1

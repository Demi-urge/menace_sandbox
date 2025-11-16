import io
import sys
import types
import importlib
import json
from pathlib import Path

import pytest

import synergy_weight_cli as cli


class DummySettings:
    def __init__(self, dir_: Path) -> None:
        self.sandbox_data_dir = str(dir_)
        self.synergy_weights_lr = 0.1


def _stub_engine(monkeypatch, learner_cls):
    mod = types.ModuleType("menace.self_improvement")
    mod.SynergyWeightLearner = learner_cls
    mod.DQNSynergyLearner = learner_cls
    mod.DoubleDQNSynergyLearner = learner_cls
    mod.SACSynergyLearner = learner_cls
    mod.TD3SynergyLearner = learner_cls
    monkeypatch.setitem(sys.modules, "menace.self_improvement", mod)
    if "menace" in sys.modules:
        monkeypatch.setattr(sys.modules["menace"], "self_improvement", mod, raising=False)


class FakeLearner:
    def __init__(self, path: Path, lr: float = 0.1) -> None:
        self.path = Path(path)
        self.lr = lr
        self.weights = {"roi": 1.0}
        self.saved = False
        self.updates = []

    def update(self, roi_delta: float, entry: dict[str, float]) -> None:
        self.updates.append((roi_delta, dict(entry)))

    def save(self) -> None:
        self.saved = True


class FailingLearner(FakeLearner):
    def update(self, roi_delta: float, entry: dict[str, float]) -> None:  # pragma: no cover - behaviour tested
        raise RuntimeError("boom")


def test_train_from_history_logs(monkeypatch, tmp_path):
    _stub_engine(monkeypatch, FakeLearner)

    settings_mod = types.ModuleType("sandbox_settings")
    settings_mod.SandboxSettings = lambda: DummySettings(tmp_path)
    monkeypatch.setitem(sys.modules, "sandbox_settings", settings_mod)

    logs = []

    def fake_log(path, weights):
        logs.append((Path(path), dict(weights)))

    monkeypatch.setattr(cli, "_log_weights", fake_log)

    hist = [{"synergy_roi": 1.0}, {"synergy_roi": -0.5}]
    cli.train_from_history(hist, tmp_path / "weights.json")

    assert len(logs) == 1
    assert logs[0][0] == cli.LOG_PATH
    assert logs[0][1]["roi"] == 1.0


def test_train_update_failure(monkeypatch, tmp_path):
    _stub_engine(monkeypatch, FailingLearner)

    settings_mod = types.ModuleType("sandbox_settings")
    settings_mod.SandboxSettings = lambda: DummySettings(tmp_path)
    monkeypatch.setitem(sys.modules, "sandbox_settings", settings_mod)

    alerts = []

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(cli, "dispatch_alert", fake_alert)

    cli.synergy_weight_update_failures_total.set(0.0)
    cli.synergy_weight_update_alerts_total.set(0.0)

    hist = [{"synergy_roi": 1.0}]
    with pytest.raises(RuntimeError):
        cli.train_from_history(hist, tmp_path / "weights.json")

    assert cli.synergy_weight_update_failures_total.labels().get() == 1.0
    assert cli.synergy_weight_update_alerts_total.labels().get() == 1.0
    assert alerts
    assert alerts[0][0][0] == "synergy_weight_update_failure"

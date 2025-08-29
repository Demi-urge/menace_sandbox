import json
import types
import io
import sys
import pytest

import synergy_weight_cli as cli


class DummyLearner:
    def __init__(self):
        self.weights = {"roi": 1.0, "efficiency": 2.0, "resilience": 3.0, "antifragility": 4.0}
        self.saved = False

    def save(self):
        self.saved = True


class DummyEngine:
    def __init__(self, *a, **k):
        self.synergy_learner = DummyLearner()


def _make_history(tmp_path):
    hist_file = tmp_path / "hist.json"
    hist = [
        {"synergy_roi": 0.5, "synergy_efficiency": 0.1},
        {"synergy_roi": -0.2, "synergy_efficiency": -0.1},
    ]
    hist_file.write_text(json.dumps(hist))
    return hist_file


def test_show(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: DummyEngine())
    cli.cli(["show"])
    data = json.loads(capsys.readouterr().out)
    assert data["roi"] == 1.0


def test_export(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: DummyEngine())
    outp = tmp_path / "w.json"
    cli.cli(["export", "--out", str(outp)])
    dumped = json.loads(outp.read_text())
    assert dumped["efficiency"] == 2.0


def test_import(monkeypatch, tmp_path):
    engine = DummyEngine()
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: engine)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"roi": 9}))
    cli.cli(["import", str(inp)])
    assert engine.synergy_learner.weights["roi"] == 9.0
    assert engine.synergy_learner.saved


def test_train(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: DummyEngine())
    hist = _make_history(tmp_path)
    called = {}

    def fake_train(history, path):
        called["history"] = history
        called["path"] = path

    monkeypatch.setattr(cli, "train_from_history", fake_train)
    cli.cli(["train", str(hist)])
    assert isinstance(called.get("history"), list)
    assert len(called["history"]) == 2
    assert called["path"] is None


def test_reset(monkeypatch):
    engine = DummyEngine()
    engine.synergy_learner.weights["roi"] = 5.0
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: engine)
    log = io.StringIO()

    def fake_log(path, weights):
        log.write(json.dumps(weights))

    monkeypatch.setattr(cli, "_log_weights", fake_log)
    cli.cli(["reset"])
    assert engine.synergy_learner.weights["roi"] == 1.0
    assert engine.synergy_learner.saved
    assert "roi" in log.getvalue()


def test_history(monkeypatch, tmp_path, capsys):
    hist_file = tmp_path / "log.log"
    hist_file.write_text(json.dumps({"timestamp": 1, "roi": 2}) + "\n")
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: DummyEngine())
    cli.cli(["history", "--log", str(hist_file)])
    out = capsys.readouterr().out
    assert "roi" in out


def test_history_plot(monkeypatch, tmp_path):
    hist_file = tmp_path / "log.log"
    hist_file.write_text(json.dumps({"timestamp": 1, "roi": 2}) + "\n")
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: DummyEngine())
    called = {}
    monkeypatch.setattr(cli, "_plot_history", lambda p: called.setdefault("p", p))
    cli.cli(["history", "--log", str(hist_file), "--plot"])
    assert called["p"] == hist_file


@pytest.mark.parametrize(
    "env,cls_name",
    [
        ("dqn", "DQNSynergyLearner"),
        ("td3", "TD3SynergyLearner"),
    ],
)
def test_load_engine_env(monkeypatch, tmp_path, env, cls_name):
    mod = types.ModuleType("menace.self_improvement")

    class BaseLearner:
        def __init__(self, *a, **k):
            pass

    class DQN(BaseLearner):
        pass

    class Double(BaseLearner):
        pass

    class SAC(BaseLearner):
        pass

    class TD3(BaseLearner):
        pass

    class Engine:
        def __init__(self, *, interval=0, synergy_weights_path=None, synergy_learner_cls=BaseLearner):
            self.synergy_learner = synergy_learner_cls()

    mod.SelfImprovementEngine = Engine
    mod.SynergyWeightLearner = BaseLearner
    mod.DQNSynergyLearner = DQN
    mod.DoubleDQNSynergyLearner = Double
    mod.SACSynergyLearner = SAC
    mod.TD3SynergyLearner = TD3

    monkeypatch.setitem(sys.modules, "menace.self_improvement", mod)
    if "menace" in sys.modules:
        monkeypatch.setattr(sys.modules["menace"], "self_improvement", mod, raising=False)

    monkeypatch.setenv("SYNERGY_LEARNER", env)

    engine = cli._load_engine(str(tmp_path / "w.json"))

    expected = getattr(mod, cls_name)
    assert isinstance(engine.synergy_learner, expected)

    monkeypatch.delenv("SYNERGY_LEARNER", raising=False)

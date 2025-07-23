import json
import types

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
    engine = DummyEngine()
    monkeypatch.setattr(cli, "_load_engine", lambda p=None: engine)
    hist = _make_history(tmp_path)
    updates = []

    def record(*a, **k):
        updates.append(a)

    engine.synergy_learner.update = record
    cli.cli(["train", str(hist)])
    assert len(updates) == 2
    assert engine.synergy_learner.saved

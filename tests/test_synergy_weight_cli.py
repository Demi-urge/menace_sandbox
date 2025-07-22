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

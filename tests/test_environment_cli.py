import json
import environment_cli as cli


class DummyTracker:
    def __init__(self):
        self.loaded = None

    def load_history(self, path):
        self.loaded = path


def test_generate_cli(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "generate_presets", lambda n=None, profiles=None: [{"env": "A"}]
    )
    cli.cli(["generate", "--count", "1"])
    data = json.loads(capsys.readouterr().out)
    assert data == [{"env": "A"}]


def test_adapt_cli(monkeypatch, tmp_path):
    inp = tmp_path / "p.json"
    inp.write_text(json.dumps([{"env": "A"}]))
    outp = tmp_path / "out.json"

    called = {}

    def fake_adapt(tracker, presets):
        called["tracker"] = tracker
        called["presets"] = presets
        return [{"env": "B"}]

    monkeypatch.setattr(cli, "adapt_presets", fake_adapt)
    monkeypatch.setattr(cli, "_ROITracker", DummyTracker)

    cli.cli(["adapt", str(inp), "--out", str(outp), "--history", "hist.json"])

    assert json.loads(outp.read_text()) == [{"env": "B"}]
    assert isinstance(called.get("tracker"), DummyTracker)
    assert called.get("presets") == [{"env": "A"}]


def test_export_policy_cli(monkeypatch, tmp_path):
    agent_data = {(1, 2): {0: 0.5}}
    monkeypatch.setattr(cli, "export_preset_policy", lambda: agent_data)

    outp = tmp_path / "policy.json"
    cli.cli(["export-policy", "--out", str(outp)])

    dumped = json.loads(outp.read_text())
    assert dumped == {"1,2": {"0": 0.5}}


def test_import_policy_cli(monkeypatch, tmp_path):
    inp = tmp_path / "policy.json"
    inp.write_text(json.dumps({"1,2": {"0": 0.5}}))

    received = {}

    def fake_import(data):
        received.update(data)

    monkeypatch.setattr(cli, "import_preset_policy", fake_import)

    cli.cli(["import-policy", str(inp)])

    assert received == {(1, 2): {0: 0.5}}


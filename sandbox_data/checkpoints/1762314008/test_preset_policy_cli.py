import json
import sys

import preset_policy_cli as cli
import environment_generator as eg


class DummyAgent:
    def __init__(self, data=None):
        self.data = data or {(1, 2): {0: 0.5, 1: 0.2}}

    def export_policy(self):
        return self.data

    def import_policy(self, data):
        self.data = data


def test_round_trip(monkeypatch, tmp_path):
    agent = DummyAgent()
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", agent, raising=False)

    out = tmp_path / "p.json"
    cli.cli(["export", "--out", str(out)])
    dumped = json.loads(out.read_text())
    assert dumped == {"1,2": {"0": 0.5, "1": 0.2}}

    new_agent = DummyAgent(data={})
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", new_agent, raising=False)
    cli.cli(["import", str(out)])
    assert new_agent.data == agent.data

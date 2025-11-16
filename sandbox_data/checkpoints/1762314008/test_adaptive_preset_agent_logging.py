import builtins
import logging
import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.environment_generator import AdaptivePresetAgent


def test_load_state_logs_failure(monkeypatch, tmp_path, caplog):
    path = tmp_path / "policy.pkl"
    state_file = tmp_path / "policy.pkl.state.json"
    state_file.write_text("{}")

    orig_open = builtins.open

    def bad_open(file, *args, **kwargs):
        if str(file) in {str(state_file), str(state_file) + ".tmp"}:
            raise IOError("boom")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", bad_open)
    caplog.set_level(logging.WARNING)
    AdaptivePresetAgent(str(path))
    assert "Failed to load RL state" in caplog.text


def test_save_state_logs_failure(monkeypatch, tmp_path, caplog):
    path = tmp_path / "policy.pkl"
    agent = AdaptivePresetAgent(str(path))
    agent.prev_state = (0, 0, 0, 0, 0)
    agent.prev_action = 0
    state_file = tmp_path / "policy.pkl.state.json"

    orig_open = builtins.open

    def bad_open(file, *args, **kwargs):
        if str(file) in {str(state_file), str(state_file) + ".tmp"}:
            raise IOError("boom")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", bad_open)
    caplog.set_level(logging.WARNING)
    agent._save_state()
    assert "Failed to save RL state" in caplog.text

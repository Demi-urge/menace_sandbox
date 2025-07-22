import builtins
import logging
import os
import sys
import types

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules["numpy"].isscalar = lambda x: isinstance(x, (int, float, complex))
sys.modules["numpy"].bool_ = bool
sys.modules["numpy"].ndarray = type("ndarray", (), {})
np_random = types.ModuleType("numpy.random")
np_random.seed = lambda *a, **k: None
np_random.get_state = lambda: None
np_random.set_state = lambda state: None
sys.modules.setdefault("numpy.random", np_random)
sys.modules["numpy"].random = np_random
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
sys.modules.setdefault("sklearn.linear_model", types.ModuleType("sklearn.linear_model"))
sys.modules.setdefault("sklearn.preprocessing", types.ModuleType("sklearn.preprocessing"))
sys.modules["sklearn.linear_model"].LinearRegression = object
sys.modules["sklearn.preprocessing"].PolynomialFeatures = object

sys.modules.pop("torch", None)
sys.modules.pop("torch.nn", None)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.preset_rl_agent import PresetRLAgent


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
    PresetRLAgent(str(path))
    assert "Failed to load RL state" in caplog.text


def test_save_state_logs_failure(monkeypatch, tmp_path, caplog):
    path = tmp_path / "policy.pkl"
    agent = PresetRLAgent(str(path))
    agent.prev_state = (0, 0)
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


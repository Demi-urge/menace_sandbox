import os
import sys
import types

# Ensure optional heavy deps aren't required
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

# Simulate torch not installed
sys.modules.pop("torch", None)
sys.modules.pop("torch.nn", None)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.environment_generator import AdaptivePresetAgent  # noqa: E402
import menace_sandbox.self_improvement_policy as policy_mod  # noqa: E402

policy_mod.DoubleDQNStrategy = policy_mod.QLearningStrategy


class TempTracker:
    def __init__(self):
        self.roi_history = [0.0, 0.1]
        self.metrics_history = {
            "synergy_roi": [0.0, 0.0],
            "synergy_efficiency": [0.0, 0.0],
            "synergy_resilience": [0.0, 0.0],
            "cpu_usage": [0.0, 0.0],
            "memory_usage": [0.0, 0.0],
            "threat_intensity": [0.0, 0.0],
        }


def test_state_file_reload(tmp_path):
    path = tmp_path / "policy.pkl"
    tracker = TempTracker()
    agent = AdaptivePresetAgent(str(path))
    agent.decide(tracker)
    agent.decide(tracker)
    agent.save()

    state_file = path.parent / (path.name + ".state.json")
    assert state_file.exists()

    loaded = AdaptivePresetAgent(str(path))
    assert loaded.prev_state == agent.prev_state
    assert loaded.prev_action == agent.prev_action


def test_state_file_reload_ddqn(tmp_path):
    path = tmp_path / "policy.pkl"
    tracker = TempTracker()
    agent = AdaptivePresetAgent(str(path), strategy="double_dqn")
    agent.decide(tracker)
    agent.save()

    loaded = AdaptivePresetAgent(str(path), strategy="double_dqn")
    assert loaded.prev_state == agent.prev_state
    assert loaded.prev_action == agent.prev_action
    assert isinstance(loaded.policy.strategy, type(agent.policy.strategy))


def test_state_file_backup(tmp_path):
    path = tmp_path / "policy.pkl"
    agent = AdaptivePresetAgent(str(path))
    agent.prev_state = (1, 1, 1, 1, 1, 1, 1)
    agent.prev_action = 1
    agent._save_state()

    state_file = path.parent / (path.name + ".state.json")
    bak_file = path.parent / (path.name + ".state.json.bak")
    assert state_file.exists()
    assert not bak_file.exists()
    first = state_file.read_text()

    agent.prev_state = (2, 2, 2, 2, 2, 2, 2)
    agent.prev_action = 2
    agent._save_state()

    assert bak_file.exists()
    assert bak_file.read_text() == first
    assert not state_file.with_suffix(state_file.suffix + ".tmp").exists()


def test_load_state_from_backup(tmp_path):
    path = tmp_path / "policy.pkl"
    agent = AdaptivePresetAgent(str(path))
    agent.prev_state = (1, 1, 1, 1, 1, 1, 1)
    agent.prev_action = 1
    agent._save_state()

    agent.prev_state = (2, 2, 2, 2, 2, 2, 2)
    agent.prev_action = 2
    agent._save_state()

    state_file = path.parent / (path.name + ".state.json")
    state_file.write_text("{bad json")

    loaded = AdaptivePresetAgent(str(path))
    assert loaded.prev_state == (1, 1, 1, 1, 1, 1, 1)
    assert loaded.prev_action == 1


def test_backup_rotation(tmp_path):
    path = tmp_path / "policy.pkl"
    agent = AdaptivePresetAgent(str(path))
    agent.prev_state = (1, 1, 1, 1, 1, 1, 1)
    agent.prev_action = 1
    agent._save_state()
    state_file = path.parent / (path.name + ".state.json")
    first = state_file.read_text()

    agent.prev_state = (2, 2, 2, 2, 2, 2, 2)
    agent.prev_action = 2
    agent._save_state()
    second = state_file.read_text()

    agent.prev_state = (3, 3, 3, 3, 3, 3, 3)
    agent.prev_action = 3
    agent._save_state()

    bak_file = path.parent / (path.name + ".state.json.bak")
    bak1_file = path.parent / (path.name + ".state.json.bak.1")
    assert bak_file.read_text() == second
    assert bak1_file.read_text() == first

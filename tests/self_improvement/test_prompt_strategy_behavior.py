import importlib
from pathlib import Path
import json
import types
import pytest
import sys

ROOT = Path(__file__).resolve().parents[2]

pkg = types.ModuleType("self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules.setdefault("self_improvement", pkg)

sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: p, repo_root=lambda: ROOT),
)
boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

prompt_memory = importlib.import_module("self_improvement.prompt_memory")
PromptStrategyManager = importlib.import_module(
    "self_improvement.prompt_strategy_manager"
).PromptStrategyManager


@pytest.fixture
def strategy_templates():
    return ["alpha", "beta", "gamma"]


@pytest.fixture
def mock_roi_stats(strategy_templates):
    return {
        "alpha": {"success": 1, "avg_roi": 1.0, "trials": 1},
        "beta": {"success": 1, "avg_roi": 5.0, "trials": 1},
        "gamma": {"success": 1, "avg_roi": 2.0, "trials": 1},
    }


@pytest.fixture
def dummy_prompt(strategy_templates):
    return types.SimpleNamespace(
        system="s",
        user="u",
        examples=[],
        metadata={"prompt_id": strategy_templates[0]},
    )


def test_failure_reason_logged(tmp_path, monkeypatch, dummy_prompt):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    penalty_path = tmp_path / "penalties.json"
    monkeypatch.setattr(prompt_memory, "_penalty_path", penalty_path)
    from filelock import FileLock
    monkeypatch.setattr(
        prompt_memory, "_penalty_lock", FileLock(str(penalty_path) + ".lock")
    )
    monkeypatch.setattr(prompt_memory._settings, "prompt_failure_log_path", "fail.json")
    monkeypatch.setattr(prompt_memory._settings, "prompt_success_log_path", "succ.json")

    prompt_memory.log_prompt_attempt(
        dummy_prompt,
        success=False,
        exec_result={"error": "boom"},
        failure_reason="api_error",
    )

    log_path = tmp_path / "fail.json"
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["failure_reason"] == "api_error"
    assert entry["prompt_id"] == dummy_prompt.metadata["prompt_id"]


def test_strategies_rotate_after_failure(tmp_path, strategy_templates):
    mgr = PromptStrategyManager(strategy_templates, state_path=tmp_path / "state.json")
    first = mgr.select(lambda seq: seq[0])
    mgr.record_failure()
    second = mgr.select(lambda seq: seq[0])
    assert first == strategy_templates[0]
    assert second == strategy_templates[1]


def test_high_roi_favored_over_penalized(strategy_templates, mock_roi_stats):
    penalties = {s: 5 for s in strategy_templates}

    class MiniEngine:
        def __init__(self, stats):
            self.strategy_stats = stats
            self.deprioritized_strategies = set()

        def select(self, strategies, threshold=3, multiplier=0.5):
            eligible = []
            penalised = []
            for strat in strategies:
                if strat in self.deprioritized_strategies:
                    continue
                count = penalties.get(str(strat), 0)
                weight = multiplier if threshold and count >= threshold else 1.0
                stats = self.strategy_stats.get(str(strat))
                if stats:
                    roi_factor = stats.get("avg_roi", 0.0)
                    roi_factor = roi_factor if roi_factor > 0 else 0.1
                    weight *= roi_factor * max(stats.get("success", 0), 1)
                target = penalised if threshold and count >= threshold else eligible
                target.append((strat, weight))
            pool = eligible or penalised
            best = None
            best_weight = -1.0
            for strat, weight in pool:
                if weight > best_weight:
                    best_weight = weight
                    best = strat
            return best

    eng = MiniEngine(mock_roi_stats)
    selected = eng.select(strategy_templates)
    assert selected == "beta"


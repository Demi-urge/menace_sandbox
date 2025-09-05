import importlib
from pathlib import Path
import json
import types
import pytest
import sys

ROOT = Path(__file__).resolve().parents[2]

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: p, repo_root=lambda: ROOT),
)
boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

prompt_memory = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_memory"
)
import menace_sandbox.prompt_optimizer as prompt_optimizer  # noqa: E402
PromptStrategyManager = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_strategy_manager"
).PromptStrategyManager
from dynamic_path_router import resolve_path  # noqa: E402


@pytest.fixture
def strategy_templates():
    return ["alpha", "beta", "gamma"]


@pytest.fixture
def mock_roi_stats(strategy_templates):
    return {
        "alpha": {"score": 1.0},
        "beta": {"score": 5.0},
        "gamma": {"score": 2.0},
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
        sandbox_metrics={"s": 2},
    )

    log_path = tmp_path / "fail.json"
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["failure_reason"] == "api_error"
    assert entry["sandbox_metrics"] == {"s": 2}
    assert entry["s"] == 2
    assert entry["prompt_id"] == dummy_prompt.metadata["prompt_id"]


def test_strategies_rotate_after_failure(tmp_path, strategy_templates):
    mgr = PromptStrategyManager(strategy_templates, state_path=tmp_path / "state.json")
    first = mgr.select(lambda seq: seq[0])
    mgr.record_failure()
    second = mgr.select(lambda seq: seq[0])
    assert first == strategy_templates[0]
    assert second == strategy_templates[1]


def test_high_roi_favored_over_penalized(strategy_templates, mock_roi_stats, monkeypatch):
    penalties = {s: 5 for s in strategy_templates}

    class MiniEngine:
        def __init__(self):
            self.deprioritized_strategies = set()

        def select(self, strategies, threshold=3, multiplier=0.5):
            eligible = []
            penalised = []
            stats = prompt_optimizer.load_strategy_stats()
            for strat in strategies:
                if strat in self.deprioritized_strategies:
                    continue
                count = penalties.get(str(strat), 0)
                weight = multiplier if threshold and count >= threshold else 1.0
                rs = stats.get(str(strat))
                if rs:
                    score = rs.get("score", 0.0)
                    score = score if score > 0 else 0.1
                    weight *= score
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

    monkeypatch.setattr(
        prompt_optimizer, "load_strategy_stats", lambda: mock_roi_stats
    )
    eng = MiniEngine()
    selected = eng.select(strategy_templates)
    assert selected == "beta"


def test_roi_stats_logged(tmp_path, monkeypatch, dummy_prompt):
    path = Path(resolve_path(str(tmp_path / "stats.json")))
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory, "_strategy_stats_path", path)
    from filelock import FileLock
    monkeypatch.setattr(
        prompt_memory, "_strategy_lock", FileLock(str(path) + ".lock")
    )
    prompt_memory.log_prompt_attempt(
        dummy_prompt, True, exec_result={}, roi_meta={"roi_delta": 1.5}
    )
    stats = prompt_memory.load_strategy_roi_stats()
    assert pytest.approx(stats[dummy_prompt.metadata["prompt_id"]]["avg_roi"], 0.01) == 1.5

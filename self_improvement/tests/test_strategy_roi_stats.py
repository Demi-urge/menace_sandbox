import types
import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Minimal stubs for heavy dependencies
sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: p, repo_root=lambda: ROOT),
)
boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)
pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg
from menace_sandbox.sandbox_settings import SandboxSettings

class DummyPrompt:
    def __init__(self, strategy):
        self.metadata = {"strategy": strategy}

class MiniEngine:
    def __init__(self, path):
        self.strategy_stats_path = path
        self.strategy_stats = {}
        self.deprioritized_strategies = set()
        self.logger = logging.getLogger("test")

    def _save_strategy_stats(self):
        self.strategy_stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.strategy_stats_path.write_text(json.dumps(self.strategy_stats))

    def _update_strategy_stats(self, prompt, success, delta):
        strategy = getattr(prompt, "metadata", {}).get("strategy") if prompt else None
        if not strategy:
            return
        stats = self.strategy_stats.setdefault(strategy, {"success": 0, "avg_roi": 0.0, "trials": 0})
        roi = float(delta.get("roi", 0.0))
        stats["trials"] += 1
        stats["avg_roi"] = ((stats["avg_roi"] * (stats["trials"] - 1)) + roi) / stats["trials"]
        if success:
            stats["success"] += 1
        self._save_strategy_stats()

    def _select_prompt_strategy(self, strategies):
        penalties = {}
        settings = SandboxSettings()
        threshold = settings.prompt_failure_threshold
        eligible = []
        penalised = []
        for strat in strategies:
            if strat in self.deprioritized_strategies:
                continue
            count = penalties.get(str(strat), 0)
            weight = settings.prompt_penalty_multiplier if threshold and count >= threshold else 1.0
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

def test_roi_weighted_selection(tmp_path):
    stats_path = tmp_path / "stats.json"
    eng = MiniEngine(stats_path)
    eng._update_strategy_stats(DummyPrompt("s1"), True, {"roi": 2.0})
    eng._update_strategy_stats(DummyPrompt("s2"), True, {"roi": 0.5})
    assert json.loads(stats_path.read_text())["s1"]["avg_roi"] > 1.9
    choice = eng._select_prompt_strategy(["s1", "s2"])
    assert choice == "s1"
    eng2 = MiniEngine(stats_path)
    eng2.strategy_stats = json.loads(stats_path.read_text())
    assert eng2._select_prompt_strategy(["s1", "s2"]) == "s1"

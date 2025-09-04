import types
import sys
from pathlib import Path
import json

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

from menace_sandbox.sandbox_settings import SandboxSettings  # noqa: E402
import menace_sandbox.prompt_optimizer as prompt_optimizer  # noqa: E402
from dynamic_path_router import resolve_path  # noqa: E402


class MiniEngine:
    def __init__(self):
        self.deprioritized_strategies = set()

    def _select_prompt_strategy(self, strategies):
        penalties = {}
        settings = SandboxSettings()
        threshold = settings.prompt_failure_threshold
        eligible = []
        penalised = []
        stats = prompt_optimizer.load_strategy_stats()
        for strat in strategies:
            if strat in self.deprioritized_strategies:
                continue
            count = penalties.get(str(strat), 0)
            weight = (
                settings.prompt_penalty_multiplier
                if threshold and count >= threshold
                else 1.0
            )
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


def test_roi_weighted_selection(tmp_path, monkeypatch):
    path = Path(resolve_path(str(tmp_path / "stats.json")))
    data = {
        "s1": {
            "success": 1,
            "total": 1,
            "roi_sum": 2.0,
            "weighted_roi_sum": 2.0,
            "weight_sum": 1.0,
        },
        "s2": {
            "success": 1,
            "total": 1,
            "roi_sum": 0.5,
            "weighted_roi_sum": 0.5,
            "weight_sum": 1.0,
        },
    }
    path.write_text(json.dumps(data))
    monkeypatch.setattr(prompt_optimizer, "DEFAULT_STRATEGY_PATH", path)
    stats = prompt_optimizer.load_strategy_stats()
    assert stats["s1"]["weighted_roi"] > 1.9
    monkeypatch.setattr(
        prompt_optimizer, "load_strategy_stats", lambda p=None: stats
    )
    eng = MiniEngine()
    assert eng._select_prompt_strategy(["s1", "s2"]) == "s1"
    eng2 = MiniEngine()
    assert eng2._select_prompt_strategy(["s1", "s2"]) == "s1"

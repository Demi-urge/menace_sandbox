import types
import sys
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

from menace_sandbox.sandbox_settings import SandboxSettings  # noqa: E402
from menace_sandbox.self_improvement import prompt_memory  # noqa: E402
from filelock import FileLock  # noqa: E402
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
        stats = prompt_memory.load_strategy_roi_stats()
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
                roi_factor = rs.get("avg_roi", 0.0)
                roi_factor = roi_factor if roi_factor > 0 else 0.1
                weight *= roi_factor
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
    monkeypatch.setattr(prompt_memory, "_strategy_stats_path", path)
    monkeypatch.setattr(
        prompt_memory, "_strategy_lock", FileLock(str(path) + ".lock")
    )
    prompt_memory.update_strategy_roi("s1", 2.0)
    prompt_memory.update_strategy_roi("s2", 0.5)
    stats = prompt_memory.load_strategy_roi_stats()
    assert stats["s1"]["avg_roi"] > 1.9
    eng = MiniEngine()
    assert eng._select_prompt_strategy(["s1", "s2"]) == "s1"
    eng2 = MiniEngine()
    assert eng2._select_prompt_strategy(["s1", "s2"]) == "s1"

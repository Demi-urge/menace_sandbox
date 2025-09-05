import sys
import types
import json
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

import menace_sandbox.prompt_optimizer as prompt_optimizer  # noqa: E402
from menace_sandbox.self_improvement.prompt_strategy_manager import (  # noqa: E402
    PromptStrategyManager,
)
from dynamic_path_router import resolve_path  # noqa: E402


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
    mgr = PromptStrategyManager(stats_path=path, state_path=tmp_path / "state.json")
    mgr.set_strategies(["s1", "s2"])
    assert mgr.best_strategy(["s1", "s2"]) == "s1"
    mgr2 = PromptStrategyManager(stats_path=path, state_path=tmp_path / "state.json")
    mgr2.set_strategies(["s1", "s2"])
    assert mgr2.best_strategy(["s1", "s2"]) == "s1"

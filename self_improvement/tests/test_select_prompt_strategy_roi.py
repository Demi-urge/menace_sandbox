import sys
import types
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

router = types.ModuleType("dynamic_path_router")
router.resolve_path = lambda p: p
router.repo_root = lambda: ROOT
sys.modules.setdefault("dynamic_path_router", router)

from dynamic_path_router import resolve_path

boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

from menace_sandbox.self_improvement.prompt_strategy_manager import (  # noqa: E402
    PromptStrategyManager,
)

def test_best_strategy_prefers_high_roi(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"  # path-ignore
    data = {
        "s1": {
            "success": 1,
            "total": 1,
            "roi_sum": 0.5,
            "weighted_roi_sum": 0.5,
            "weight_sum": 1.0,
        },
        "s2": {
            "success": 1,
            "total": 1,
            "roi_sum": 1.5,
            "weighted_roi_sum": 1.5,
            "weight_sum": 1.0,
        },
    }
    stats_path.write_text(json.dumps(data))
    mgr = PromptStrategyManager(stats_path=stats_path, state_path=tmp_path / "state.json")  # path-ignore
    mgr.set_strategies(["s1", "s2"])
    assert mgr.best_strategy(["s1", "s2"]) == "s2"

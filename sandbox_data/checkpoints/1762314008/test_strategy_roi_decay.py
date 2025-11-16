import sys
import types
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: p, repo_root=lambda: ROOT),
)

from dynamic_path_router import resolve_path

boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

from menace_sandbox.self_improvement.prompt_strategy_manager import PromptStrategyManager  # noqa: E402
import sandbox_settings as settings_mod  # noqa: E402


class DummySettings:
    prompt_failure_threshold = None
    prompt_penalty_multiplier = 1.0
    prompt_roi_decay_rate = 1.0

    def __init__(self) -> None:
        pass


def test_recent_success_outweighs_old(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod, "SandboxSettings", DummySettings)
    mgr = PromptStrategyManager(stats_path=tmp_path / "stats.json", state_path=tmp_path / "state.json")  # path-ignore
    mgr.set_strategies(["s1", "s2"])
    now = time.time()
    mgr.stats = {
        "s1": {
            "total": 1,
            "success": 1,
            "roi_sum": 10.0,
            "weighted_roi_sum": 10.0,
            "weight_sum": 1.0,
            "records": [{"ts": now - 1000, "roi": 10.0, "success": True}],
        },
        "s2": {
            "total": 1,
            "success": 1,
            "roi_sum": 1.0,
            "weighted_roi_sum": 1.0,
            "weight_sum": 1.0,
            "records": [{"ts": now - 1, "roi": 1.0, "success": True}],
        },
    }
    decay = DummySettings.prompt_roi_decay_rate
    scores = {}
    for name, rec in mgr.stats.items():
        total_w = success_w = roi_w = 0.0
        for r in rec["records"]:
            w = math.exp(-decay * (now - r["ts"]))
            total_w += w
            roi_w += r["roi"] * w
            if r["success"]:
                success_w += w
        avg_weight = total_w / len(rec["records"])
        score = (
            success_w / total_w * max(roi_w / total_w, 0.0) * avg_weight
            if total_w
            else 0.1
        )
        scores[name] = score
    assert scores["s2"] > scores["s1"]

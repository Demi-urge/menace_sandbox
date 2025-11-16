import sys
import types
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

router = types.ModuleType("dynamic_path_router")
router.resolve_path = lambda p: p
router.repo_root = lambda: ROOT
sys.modules.setdefault("dynamic_path_router", router)

boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg
PromptStrategyManager = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_strategy_manager"
).PromptStrategyManager


def test_next_strategy_uses_roi_selection():
    mgr = PromptStrategyManager()
    captured: list[list[str]] = []

    def fake_best(seq):
        captured.append(list(seq))
        return seq[0] if seq else None

    mgr.best_strategy = fake_best
    current = mgr.strategies[0]
    nxt = mgr.record_failure(current, "fail")
    assert captured[0] == mgr.strategies[1:]
    assert nxt == mgr.strategies[1]

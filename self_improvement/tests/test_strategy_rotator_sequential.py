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


def test_next_strategy_rotates_through_templates():
    mgr = PromptStrategyManager()
    current = mgr.strategies[0]
    sequence = []
    for _ in range(len(mgr.strategies)):
        nxt = mgr.record_failure(current, "fail")
        sequence.append(nxt)
        current = nxt
    assert sequence == mgr.strategies[1:] + [mgr.strategies[0]]

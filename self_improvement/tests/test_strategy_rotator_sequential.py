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

strategy_rotator = importlib.import_module("menace_sandbox.self_improvement.strategy_rotator")


def test_next_strategy_rotates_through_templates():
    for key in list(strategy_rotator.failure_counts):
        strategy_rotator.failure_counts[key] = 0
        strategy_rotator.failure_limits[key] = 1

    current = strategy_rotator.TEMPLATES[0]
    sequence = []
    for _ in range(len(strategy_rotator.TEMPLATES)):
        nxt = strategy_rotator.next_strategy(current, "fail")
        sequence.append(nxt)
        current = nxt

    assert sequence == strategy_rotator.TEMPLATES[1:] + [strategy_rotator.TEMPLATES[0]]

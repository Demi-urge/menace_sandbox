import importlib
import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: p))
pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg
strategy_rotator = importlib.import_module(
    "menace_sandbox.self_improvement.strategy_rotator"
)


class DummyPrompt:
    def __init__(self, strategy: str):
        self.metadata = {"strategy": strategy}


class MiniEngine:
    def __init__(self):
        self.logger = logging.getLogger("test")
        self.pending_strategy = None
        self.strategy_manager = strategy_rotator.manager

    def _record_snapshot_delta(self, prompt, delta):
        strategy = getattr(prompt, "metadata", {}).get("strategy")
        if strategy:
            nxt = strategy_rotator.next_strategy(str(strategy), "regression")
            if delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0:
                self.pending_strategy = nxt

    def next_prompt_strategy(self):
        pending = self.pending_strategy
        if pending and pending in self.strategy_manager.strategies:
            self.pending_strategy = None
            return pending
        return self.strategy_manager.select(lambda seq: seq[0])


def test_rotation_and_pending_strategy():
    eng = MiniEngine()
    prompt = DummyPrompt("strict_fix")

    called: dict[str, tuple[str, str]] = {}

    def fake_next_strategy(current, reason):
        called["args"] = (current, reason)
        return "delete_rebuild"

    import menace_sandbox.self_improvement.strategy_rotator as sr

    original = sr.next_strategy
    try:
        sr.next_strategy = fake_next_strategy  # type: ignore
        eng._record_snapshot_delta(prompt, {"roi": -1})
    finally:
        sr.next_strategy = original  # type: ignore

    assert called["args"] == ("strict_fix", "regression")
    assert eng.pending_strategy == "delete_rebuild"

    choice = eng.next_prompt_strategy()
    assert choice == "delete_rebuild"
    assert eng.pending_strategy is None

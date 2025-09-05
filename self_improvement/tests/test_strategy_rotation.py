import logging
import importlib
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
PromptStrategyManager = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_strategy_manager"
).PromptStrategyManager


class DummyPrompt:
    def __init__(self, strategy: str):
        self.metadata = {"strategy": strategy}


class MiniEngine:
    def __init__(self):
        self.logger = logging.getLogger("test")
        self.pending_strategy = None
        self.strategy_manager = PromptStrategyManager()

    def _record_snapshot_delta(self, prompt, delta):
        success = not (delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0)
        if success:
            return
        strategy = getattr(prompt, "metadata", {}).get("strategy")
        if strategy:
            self.pending_strategy = self.strategy_manager.record_failure(
                str(strategy), "regression"
            )

    def next_prompt_strategy(self):
        pending = self.pending_strategy
        if pending and pending in self.strategy_manager.strategies:
            self.pending_strategy = None
            return pending
        return self.strategy_manager.select(lambda seq: seq[0])


def test_rotation_and_pending_strategy():
    eng = MiniEngine()
    prompt = DummyPrompt("strict_fix")
    eng._record_snapshot_delta(prompt, {"roi": -1})
    assert eng.strategy_manager.failure_counts["strict_fix"] == 1
    assert eng.pending_strategy == "delete_rebuild"

    choice = eng.next_prompt_strategy()
    assert choice == "delete_rebuild"
    assert eng.pending_strategy is None

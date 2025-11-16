import logging
import sys
import types
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

sys.modules.setdefault(
    "dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: p)
)
from dynamic_path_router import resolve_path  # noqa: E402

pkg_root = types.ModuleType("menace_sandbox")
pkg_root.__path__ = [str(ROOT)]
sys.modules["menace_sandbox"] = pkg_root

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

PromptStrategyManager = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_strategy_manager"
).PromptStrategyManager


class DummyPrompt:
    metadata = {"strategy": "s1"}


class MiniEngine:
    def __init__(self, state_path: Path):
        self.logger = logging.getLogger("test")
        self.prompt_strategy_manager = PromptStrategyManager(
            ["s1", "s2"], state_path=state_path, stats_path=state_path.with_name("stats.json")
        )

    def next_strategy(self) -> str | None:
        return self.prompt_strategy_manager.select(self._select_prompt_strategy)

    def _select_prompt_strategy(self, strategies):
        best = self.prompt_strategy_manager.best_strategy(strategies)
        return best if best else (strategies[0] if strategies else None)

    def _record_snapshot_delta(self, prompt, delta):
        success = not (delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0)
        if not success:
            self.prompt_strategy_manager.record_failure()


def test_strategy_rotation_persists(tmp_path):
    state_file = Path(resolve_path(tmp_path / "state.json"))
    eng = MiniEngine(state_file)
    assert eng.next_strategy() == "s1"
    eng._record_snapshot_delta(DummyPrompt(), {"roi": -1})
    assert eng.next_strategy() == "s2"

    # Reload to ensure rotation state persisted
    eng2 = MiniEngine(state_file)
    assert eng2.next_strategy() == "s2"
    eng2._record_snapshot_delta(DummyPrompt(), {"roi": -1})

    eng3 = MiniEngine(state_file)
    assert eng3.next_strategy() == "s1"


def test_failure_reason_selects_keyword_strategy(tmp_path):
    mgr = PromptStrategyManager(
        state_path=Path(resolve_path(tmp_path / "state.json")),
        stats_path=Path(resolve_path(tmp_path / "stats.json")),
    )
    mgr.set_strategies(["strict_fix", "delete_rebuild", "unit_test_rewrite"])
    mgr.index = mgr.strategies.index("delete_rebuild")
    nxt = mgr.record_failure("delete_rebuild", "needs refactor")
    assert nxt == "strict_fix"


def test_roi_fallback_used_when_no_keyword(tmp_path, monkeypatch):
    mgr = PromptStrategyManager(
        ["strict_fix", "delete_rebuild"],
        state_path=Path(resolve_path(tmp_path / "state.json")),
        stats_path=Path(resolve_path(tmp_path / "stats.json")),
    )

    called = {}

    def fake_best(strats):
        called["pool"] = list(strats)
        return "delete_rebuild"

    monkeypatch.setattr(mgr, "best_strategy", fake_best)
    nxt = mgr.record_failure("strict_fix", "unknown failure")
    assert called["pool"] == ["delete_rebuild"]
    assert nxt == "delete_rebuild"

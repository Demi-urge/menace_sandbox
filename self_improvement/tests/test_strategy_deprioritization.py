import logging
import logging
import importlib
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: p))
import sandbox_settings
from menace_sandbox.sandbox_settings import SandboxSettings
from menace_sandbox.dynamic_path_router import resolve_path

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(Path(resolve_path("self_improvement")))]
sys.modules["menace_sandbox.self_improvement"] = pkg
boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)
prompt_memory = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")
snapshot_tracker = importlib.import_module(
    "menace_sandbox.self_improvement.snapshot_tracker"
)
if "snapshot_history_db" in sys.modules:
    del sys.modules["snapshot_history_db"]
snapshot_history_db = importlib.import_module("menace_sandbox.snapshot_history_db")


class DummyPrompt:
    metadata = {"strategy": "s1"}


class MiniEngine:
    def __init__(self):
        self.deprioritized_strategies = set()
        self.logger = logging.getLogger("test")
        self.strategy_manager = snapshot_tracker.StrategyManager()

    def _record_snapshot_delta(self, prompt, delta):
        success = not (delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0)
        metadata = getattr(prompt, "metadata", {})
        strategy = None
        if isinstance(metadata, dict):
            strategy = metadata.get("strategy") or metadata.get("prompt_id")
        if strategy:
            if not success:
                count = snapshot_tracker.record_downgrade(str(strategy))
                if count >= SandboxSettings().prompt_failure_threshold:
                    self.deprioritized_strategies.add(str(strategy))
            self.strategy_manager.update(str(strategy), delta.get("roi", 0.0), success)

    def next_prompt_strategy(self, strategies):
        candidates = [s for s in strategies if s not in self.deprioritized_strategies]
        return self.strategy_manager.best_strategy(candidates)


def test_deprioritized_strategy_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(
        snapshot_tracker,
        "_downgrade_path",
        Path(resolve_path(str(tmp_path))) / Path("downgrades").with_suffix(".json"),
    )
    snapshot_tracker.downgrade_counts.clear()
    eng = MiniEngine()
    thr = SandboxSettings().prompt_failure_threshold
    for _ in range(thr):
        eng._record_snapshot_delta(DummyPrompt(), {"roi": -1})

    monkeypatch.setattr(prompt_memory, "load_prompt_penalties", lambda: {"s1": thr})

    assert "s1" in eng.deprioritized_strategies
    choice = eng.next_prompt_strategy(["s1", "s2"])
    assert choice == "s2"

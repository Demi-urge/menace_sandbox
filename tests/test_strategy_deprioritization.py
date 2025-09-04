import logging
import importlib
import types
import sys
from pathlib import Path

from menace_sandbox.sandbox_settings import SandboxSettings
from menace_sandbox.dynamic_path_router import resolve_path

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(Path(resolve_path("self_improvement")))]
sys.modules["menace_sandbox.self_improvement"] = pkg
snapshot_tracker = importlib.import_module(
    "menace_sandbox.self_improvement.snapshot_tracker"
)


class DummyPrompt:
    metadata = {"strategy": "s1"}


class MiniEngine:
    def __init__(self):
        self.deprioritized_strategies = set()
        self.logger = logging.getLogger("test")

    def _record_snapshot_delta(self, prompt, delta):
        success = not (delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0)
        if not success:
            strategy = None
            metadata = getattr(prompt, "metadata", {})
            if isinstance(metadata, dict):
                strategy = metadata.get("strategy") or metadata.get("prompt_id")
            if strategy:
                count = snapshot_tracker.record_downgrade(str(strategy))
                if count >= SandboxSettings().prompt_failure_threshold:
                    self.deprioritized_strategies.add(str(strategy))

    def _select_prompt_strategy(self, strategies):
        penalties = snapshot_tracker.downgrade_counts
        threshold = SandboxSettings().prompt_failure_threshold
        for strat in strategies:
            if strat in self.deprioritized_strategies:
                continue
            if penalties.get(str(strat), 0) >= threshold:
                continue
            return strat
        return None


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

    assert "s1" in eng.deprioritized_strategies
    assert snapshot_tracker.downgrade_counts["s1"] >= thr
    choice = eng._select_prompt_strategy(["s1", "s2"])
    assert choice == "s2"

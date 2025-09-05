import importlib
import logging
import types
import sys
from pathlib import Path

sys.modules.setdefault("dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: p))
from menace_sandbox.sandbox_settings import SandboxSettings  # noqa: E402
from menace_sandbox.dynamic_path_router import resolve_path  # noqa: E402

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
PromptStrategyManager = importlib.import_module(
    "menace_sandbox.self_improvement.prompt_strategy_manager"
).PromptStrategyManager
snapshot_history_db = importlib.import_module("menace_sandbox.snapshot_history_db")


class DummyPrompt:
    metadata = {"strategy": "s1"}


class MiniEngine:
    def __init__(self):
        self.deprioritized_strategies = set()
        self.logger = logging.getLogger("test")
        self.strategy_manager = PromptStrategyManager()

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


def test_log_regression_penalises_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    monkeypatch.setattr(prompt_memory, "_penalty_path", tmp_path / "penalties.json")
    monkeypatch.setattr(
        prompt_memory,
        "_penalty_lock",
        prompt_memory.FileLock(str(tmp_path / "penalties.json") + ".lock"),
    )
    monkeypatch.setattr(
        snapshot_history_db,
        "_db_path",
        lambda settings=None: tmp_path / "snapshot_history.db",
    )
    eng = MiniEngine()
    thr = SandboxSettings().prompt_failure_threshold
    for _ in range(thr):
        snapshot_history_db.log_regression("s1", None, {"roi": -1})
    choice = eng.next_prompt_strategy(["s1", "s2"])
    assert choice == "s2"

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: p))
from dynamic_path_router import resolve_path
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
        if strategy and (delta.get("roi", 0.0) < 0 or delta.get("entropy", 0.0) < 0):
            nxt = self.strategy_manager.record_failure(
                str(strategy), "regression", float(delta.get("roi", 0.0))
            )
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

    called: dict[str, tuple[str, str, float]] = {}

    def fake_record_failure(current, reason, roi):
        called["args"] = (current, reason, roi)
        return "delete_rebuild"

    import menace_sandbox.self_improvement.strategy_rotator as sr

    original = sr.manager.record_failure
    try:
        sr.manager.record_failure = fake_record_failure  # type: ignore
        eng._record_snapshot_delta(prompt, {"roi": -1})
    finally:
        sr.manager.record_failure = original  # type: ignore

    assert called["args"] == ("strict_fix", "regression", -1)
    assert eng.pending_strategy == "delete_rebuild"

    choice = eng.next_prompt_strategy()
    assert choice == "delete_rebuild"
    assert eng.pending_strategy is None


def _reload(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sr = importlib.reload(strategy_rotator)
    sr.manager.stats_path = Path(tmp_path) / "_strategy_stats.json"  # path-ignore
    sr.manager._stats_lock = FileLock(str(sr.manager.stats_path) + ".lock")
    return sr


def test_keyword_override(tmp_path, monkeypatch):
    sr = _reload(monkeypatch, tmp_path)
    nxt = sr.manager.record_failure("strict_fix", "tests_failed in module")
    assert nxt == "unit_test_rewrite"


def test_failure_counts_persist(tmp_path, monkeypatch):
    sr = _reload(monkeypatch, tmp_path)
    sr.manager.record_failure("strict_fix", "regression")
    assert sr.manager.failure_counts["strict_fix"] == 1
    sr = importlib.reload(sr)
    assert sr.manager.failure_counts["strict_fix"] == 1


def test_failure_limit_from_settings(tmp_path, monkeypatch):
    limits = json.dumps({"strict_fix": 1})
    sr = _reload(monkeypatch, tmp_path, STRATEGY_FAILURE_LIMITS=limits)
    current = "strict_fix"
    nxt = sr.manager.record_failure(current, "failure")
    assert nxt == "delete_rebuild"


def test_next_prefers_highest_roi(tmp_path, monkeypatch):
    sr = _reload(monkeypatch, tmp_path)
    sr.manager.ingest("strict_fix", roi_delta=1.0)
    sr.manager.ingest("delete_rebuild", failure_reason="bad", roi_delta=-1.0)
    choice = sr.manager.next()
    assert choice == "strict_fix"

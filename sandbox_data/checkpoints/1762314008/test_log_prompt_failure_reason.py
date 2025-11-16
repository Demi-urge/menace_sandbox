import json
import sys
import types
from pathlib import Path
import importlib
from dynamic_path_router import resolve_path

ROOT = Path(resolve_path(__file__)).resolve().parents[2]
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

prompt_memory = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")


def test_log_prompt_attempt_records_failure_reason(tmp_path, monkeypatch):
    def _mock_log_path(success: bool) -> Path:
        return tmp_path / ("success.jsonl" if success else "failure.jsonl")

    monkeypatch.setattr(prompt_memory, "_log_path", _mock_log_path)

    # Even if ``success`` is passed as True the presence of ``failure_reason``
    # should ensure the entry is written to the failure log.
    prompt_memory.log_prompt_attempt(
        prompt=None,
        success=True,
        exec_result={"detail": "x"},
        failure_reason="bad_result",
        sandbox_metrics={"m": 1, "sandbox_score": 0.2, "tests_passed": False, "entropy": 0.3},
        commit_hash="deadbeef",
    )

    failure_log = tmp_path / "failure.jsonl"
    assert failure_log.exists()
    entry = json.loads(failure_log.read_text().strip())
    assert entry["failure_reason"] == "bad_result"
    assert entry["sandbox_metrics"] == {
        "m": 1,
        "sandbox_score": 0.2,
        "tests_passed": False,
        "entropy": 0.3,
    }
    assert entry["score_delta"] == 0.2
    assert entry["test_status"] is False
    assert entry["entropy_delta"] == 0.3
    assert entry["m"] == 1
    assert entry["commit_hash"] == "deadbeef"
    # No success log should be written
    assert not (tmp_path / "success.jsonl").exists()


def test_uuid_and_strategy_and_raw_logged(tmp_path, monkeypatch):
    def _mock_log_path(success: bool) -> Path:
        return tmp_path / ("success.jsonl" if success else "failure.jsonl")

    monkeypatch.setattr(prompt_memory, "_log_path", _mock_log_path)

    prompt = types.SimpleNamespace(
        system="sys",
        user="usr",
        examples=["ex"],
        metadata={"strategy": "alpha"},
    )
    prompt_memory.log_prompt_attempt(
        prompt,
        success=False,
        exec_result={},
        failure_reason="oops",
    )

    entry = json.loads((tmp_path / "failure.jsonl").read_text().splitlines()[0])
    from uuid import UUID

    UUID(entry["prompt_id"])
    assert entry["strategy"] == "alpha"
    assert entry["failure_reason"] == "oops"
    assert entry["sandbox_metrics"] is None
    assert "sys" in entry["raw_prompt"]

    logs = list(prompt_memory.load_failures())
    assert logs[0]["prompt_id"] == entry["prompt_id"]

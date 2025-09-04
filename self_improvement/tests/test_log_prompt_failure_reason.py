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
    )

    failure_log = tmp_path / "failure.jsonl"
    assert failure_log.exists()
    entry = json.loads(failure_log.read_text().strip())
    assert entry["failure_reason"] == "bad_result"
    # No success log should be written
    assert not (tmp_path / "success.jsonl").exists()

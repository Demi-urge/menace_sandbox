import runpy
import sys
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from tests.test_visual_agent_auto_recover import _setup_va


def test_repair_running_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tombalolosvisualagent123")
    va = _setup_va(monkeypatch, tmp_path)

    va.job_status.clear()
    va.task_queue.clear()
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None, "status": "running"})
    va.job_status["a"] = {"status": "running", "prompt": "p", "branch": None}
    va._persist_state()

    monkeypatch.setattr(sys, "argv", ["menace_visual_agent_2", "--repair-running"])
    with pytest.raises(SystemExit):
        runpy.run_module("menace_visual_agent_2", run_name="__main__")

    q = va.VisualAgentQueue(tmp_path / "visual_agent_queue.db")
    status = q.get_status()
    assert status["a"]["status"] == "queued"

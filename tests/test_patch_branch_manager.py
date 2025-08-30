import json
from pathlib import Path
import subprocess

from patch_branch_manager import PatchBranchManager, finalize_patch_branch


class DummyTrail:
    def __init__(self):
        self.records = []

    def record(self, msg):
        self.records.append(msg)


def test_finalize_patch_branch_merges(monkeypatch):
    import patch_branch_manager as pbm

    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pbm.subprocess, "run", fake_run)
    trail = DummyTrail()
    branch = finalize_patch_branch("1", 0.9, 0.5, audit_trail=trail)
    assert any(cmd[-1].endswith("main") for cmd in calls)
    data = json.loads(trail.records[0])
    assert data["branch"] == "main"
    assert branch == "main"


def test_finalize_patch_branch_review(monkeypatch):
    import patch_branch_manager as pbm

    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pbm.subprocess, "run", fake_run)
    trail = DummyTrail()
    mgr = PatchBranchManager(audit_trail=trail)
    branch = mgr.finalize_patch("2", 0.2, 0.5)
    assert any("review/2" in cmd[-1] for cmd in calls)
    data = json.loads(trail.records[0])
    assert data["branch"] == "review/2"
    assert branch == "review/2"

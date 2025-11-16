import os
import sys
import urllib.request
from pathlib import Path
from dynamic_path_router import resolve_path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.path.insert(0, str(resolve_path("")))

from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def test_edge_case_profiles_file_and_network():
    runner = WorkflowSandboxRunner()
    profile = {
        "malformed.json": '{"broken": "json",}',
        "http://example.test/data": "payload",
    }

    def wf():
        with open("malformed.json") as fh:
            try:
                __import__("json").loads(fh.read())
            except Exception:
                pass
        data = urllib.request.urlopen("http://example.test/data").read()
        assert data == b"payload"

    metrics = runner.run(
        wf,
        edge_case_profiles=[profile],
        safe_mode=True,
        use_subprocess=False,
    )
    assert metrics.modules[0].success

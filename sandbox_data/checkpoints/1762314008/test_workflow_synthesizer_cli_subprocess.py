import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path

REPO_ROOT = Path(__file__).resolve().parent.parent

STUB_MODULE = """
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class WorkflowStep:
    module: str
    inputs: list[str] | None = None
    outputs: list[str] | None = None
    unresolved: list[str] | None = None


def save_workflow(steps, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"steps": [{"module": s.module} for s in steps]}
    p.write_text(json.dumps(data))


def to_workflow_spec(steps):
    return {"steps": [{"module": s.module} for s in steps]}


def evaluate_workflow(_spec):
    return True


class WorkflowSynthesizer:
    def __init__(self, *a, **k):
        self.generated_workflows = []
        self.workflow_score_details = []

    def generate_workflows(self, start_module, **_kwargs):
        wf1 = [WorkflowStep("mod_a"), WorkflowStep("mod_b")]
        wf2 = [WorkflowStep("mod_a"), WorkflowStep("mod_c")]
        self.generated_workflows = [wf1, wf2]
        self.workflow_score_details = [
            {"score": 1.0, "synergy": 1.0, "intent": 0.0, "penalty": 0},
            {"score": 0.5, "synergy": 0.5, "intent": 0.0, "penalty": 0},
        ]
        return self.generated_workflows
"""


def _prepare(tmp_path: Path):
    for name in ("mod_a.py", "mod_b.py", "mod_c.py"):  # path-ignore
        shutil.copy(
            resolve_path(f"tests/fixtures/workflow_modules/{name}"),
            tmp_path / name,
        )
    (tmp_path / "workflow_synthesizer.py").write_text(STUB_MODULE)  # path-ignore
    (tmp_path / "sitecustomize.py").write_text(  # path-ignore
        "import sys\n" "sys.stdin.isatty=lambda: True\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{REPO_ROOT}"
    return env


def test_cli_save_and_list(tmp_path):
    env = _prepare(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "workflow_synthesizer_cli",
            "mod_a",
            "--limit",
            "2",
            "--save",
        ],
        cwd=tmp_path,
        env=env,
        input="2\n",
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    saved = tmp_path / "sandbox_data" / "generated_workflows" / "mod_a.workflow.json"
    data = json.loads(saved.read_text())
    assert [s["module"] for s in data["steps"]] == ["mod_a", "mod_c"]
    assert "Select workflow" in result.stdout
    list_result = subprocess.run(
        [sys.executable, "-m", "workflow_synthesizer_cli", "--list"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "mod_a.workflow.json" in list_result.stdout


def test_cli_evaluate(tmp_path):
    env = _prepare(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "workflow_synthesizer_cli",
            "mod_a",
            "--limit",
            "2",
            "--evaluate",
        ],
        cwd=tmp_path,
        env=env,
        input="2\n",
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Select workflow" in result.stdout
    assert "evaluation succeeded" in result.stdout

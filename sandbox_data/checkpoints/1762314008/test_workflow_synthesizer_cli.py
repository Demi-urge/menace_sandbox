import json

import workflow_synthesizer_cli as cli
from workflow_synthesizer import WorkflowStep


class DummySynth:
    def __init__(self, *a, **k):
        self.generated_workflows = []
        self.workflow_score_details = []

    def generate_workflows(self, **_kwargs):
        wf1 = [WorkflowStep("mod_a"), WorkflowStep("mod_b")]
        wf2 = [WorkflowStep("mod_a"), WorkflowStep("mod_c")]
        self.generated_workflows = [wf1, wf2]
        self.workflow_score_details = [
            {"score": 1.0, "synergy": 1.0, "intent": 0.0, "penalty": 0},
            {"score": 0.5, "synergy": 0.5, "intent": 0.0, "penalty": 0},
        ]
        return self.generated_workflows


def test_cli_interactive_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "WorkflowSynthesizer", DummySynth)
    out = tmp_path / "wf.workflow.json"
    parser = cli.build_parser()
    args = parser.parse_args(["mod_a", "--limit", "2", "--out", str(out)])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def fake_input(prompt: str = "") -> str:
        print(prompt, end="")
        return "2"

    monkeypatch.setattr("builtins.input", fake_input)
    rc = cli.run(args)
    assert rc == 0
    saved = tmp_path / "workflows" / out.name
    data = json.loads(saved.read_text())
    modules = [s["module"] for s in data["steps"]]
    assert modules == ["mod_a", "mod_c"]
    captured = capsys.readouterr()
    assert "Select workflow" in captured.out
    assert "score=" in captured.out


def test_cli_select_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "WorkflowSynthesizer", DummySynth)
    out = tmp_path / "wf.workflow.json"
    parser = cli.build_parser()
    args = parser.parse_args(
        ["mod_a", "--limit", "2", "--out", str(out), "--select", "2"]
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    rc = cli.run(args)
    assert rc == 0
    saved = tmp_path / "workflows" / out.name
    data = json.loads(saved.read_text())
    modules = [s["module"] for s in data["steps"]]
    assert modules == ["mod_a", "mod_c"]
    captured = capsys.readouterr()
    assert "Select workflow" not in captured.out


def test_cli_unresolved_and_summary(monkeypatch, capsys):
    class Synth:
        def __init__(self, *a, **k):
            self.workflow_score_details = []

        def generate_workflows(self, **_kwargs):
            wf = [WorkflowStep("mod_a", unresolved=["missing"]) ]
            self.workflow_score_details = [
                {
                    "score": 1.0,
                    "synergy": 0.5,
                    "intent": 0.5,
                    "penalty": 1.0,
                    "success": True,
                }
            ]
            return [wf]

    monkeypatch.setattr(cli, "WorkflowSynthesizer", Synth)
    parser = cli.build_parser()
    args = parser.parse_args(["mod_a", "--auto-evaluate"])
    rc = cli.run(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "unresolved" in out
    assert "Evaluation summary" in out
    assert "succeeded" in out

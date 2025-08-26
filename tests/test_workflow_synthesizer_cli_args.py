import json

import workflow_synthesizer_cli as cli
from workflow_synthesizer import WorkflowStep


class DummySynth:
    def __init__(self, *a, **k):
        self.workflow_score_details = [
            {"score": 1.0, "synergy": 1.0, "intent": 0.0, "penalty": 0.0}
        ]

    def generate_workflows(self, **_kwargs):
        return [[WorkflowStep("mod_a")]]


def test_cli_argument_parsing():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "mod_a",
            "--limit",
            "3",
            "--max-depth",
            "2",
            "--synergy-weight",
            "0.5",
            "--intent-weight",
            "0.25",
            "--min-score",
            "0.1",
        ]
    )
    assert args.start == "mod_a"
    assert args.limit == 3
    assert args.max_depth == 2
    assert args.synergy_weight == 0.5
    assert args.intent_weight == 0.25
    assert args.min_score == 0.1


def test_cli_out_saves_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "WorkflowSynthesizer", DummySynth)
    parser = cli.build_parser()
    out = tmp_path / "wf.workflow.json"
    args = parser.parse_args(["mod_a", "--out", str(out)])
    monkeypatch.chdir(tmp_path)
    rc = cli.run(args)
    assert rc == 0
    saved = tmp_path / "workflows" / out.name
    data = json.loads(saved.read_text())
    assert [s["module"] for s in data["steps"]] == ["mod_a"]

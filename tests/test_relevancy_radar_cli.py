import json
import relevancy_radar_cli as cli
import relevancy_radar


def test_replace_annotation(tmp_path, capsys, monkeypatch):
    metrics_file = tmp_path / "relevancy_metrics.json"
    monkeypatch.setattr(cli, "_METRICS_FILE", metrics_file)
    metrics_file.write_text(json.dumps({"alpha": {"imports": 1, "executions": 0}}))

    rc = cli.cli(["--replace", "alpha"])
    assert rc == 0

    data = json.loads(metrics_file.read_text())
    assert data["alpha"]["annotation"] == "replace"

    output = capsys.readouterr().out
    assert "alpha: 1 (replace)" in output


def test_final_flag_invokes_evaluation(tmp_path, capsys, monkeypatch):
    metrics_file = tmp_path / "relevancy_metrics.json"
    monkeypatch.setattr(cli, "_METRICS_FILE", metrics_file)
    metrics_file.write_text(json.dumps({"beta": {"imports": 0, "executions": 0}}))

    called = {}

    def fake_eval(comp, repl):
        called["args"] = (comp, repl)
        return {"beta": "retire"}

    monkeypatch.setattr(relevancy_radar, "evaluate_final_contribution", fake_eval)

    rc = cli.cli(["--final"])
    assert rc == 0
    assert called["args"] == (2, 5)

    data = json.loads(metrics_file.read_text())
    assert data["beta"]["annotation"] == "retire"

    output = capsys.readouterr().out
    assert "beta: 0 (retire)" in output

import json
import relevancy_radar_cli as cli


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

import importlib
import roi_tracker as rt

cli = importlib.import_module("sandbox_runner.cli")


def _write(path, data):
    t = rt.ROITracker()
    t.scenario_synergy = data
    t.save_history(str(path))


def test_scenario_synergy_ranking(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, {"one": [{"synergy_roi": 0.1}], "two": [{"synergy_roi": 0.3}]})
    _write(b, {"one": [{"synergy_roi": 0.5}]})

    cli.rank_scenario_synergy([str(a), str(b)])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("one")


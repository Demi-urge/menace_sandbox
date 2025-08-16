import importlib
import json
import atexit


def test_tracking_and_persistence(tmp_path, monkeypatch):
    """Ensure tracking increments counters and persistence merges counts."""
    # Prevent actual atexit registration during tests.
    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.import_module("relevancy_radar")

    fake_file = tmp_path / "module_usage.json"
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", fake_file)
    rr._module_usage_counter.clear()

    rr.track_module_usage("alpha")
    rr.track_module_usage("alpha")
    rr.track_module_usage("beta")

    stats = rr.load_usage_stats()
    assert stats == {"alpha": 2, "beta": 1}

    rr._save_usage_counts()
    assert json.loads(fake_file.read_text()) == stats

    rr._module_usage_counter.clear()
    rr.track_module_usage("alpha")
    stats2 = rr.load_usage_stats()
    assert stats2 == {"alpha": 3, "beta": 1}

import json
from menace_sandbox.foresight_tracker import ForesightTracker


def test_sandbox_runner_like_persistence(tmp_path):
    history_file = tmp_path / "foresight_history.json"
    tracker = ForesightTracker(max_cycles=2)
    tracker.record_cycle_metrics("wf", {"m": 0.0})
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(tracker.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    loaded = ForesightTracker.from_dict(data, max_cycles=tracker.max_cycles)
    loaded.record_cycle_metrics("wf", {"m": 1.0})
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(loaded.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        final = json.load(fh)
    assert [e["m"] for e in final["history"]["wf"]] == [0.0, 1.0]

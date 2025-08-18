import argparse
import json
from pathlib import Path

import sandbox_runner.cli as cli


def test_minimal_autonomous_run(monkeypatch, tmp_path):
    dummy_preset = {"CPU_LIMIT": "1"}
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [dummy_preset])

    class DummyTracker:
        def __init__(self):
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": [0.05]}
            self.roi_history = [0.1]

        def save_history(self, path: str) -> None:
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                    }
                )
            )

        def diminishing(self) -> float:
            return 0.0

        def rankings(self):
            return [("m", 0.1, 0.1)]

    def fake_capture(preset, args):
        tracker = DummyTracker()
        Path(args.sandbox_data_dir).mkdir(parents=True, exist_ok=True)
        tracker.save_history(str(Path(args.sandbox_data_dir) / "roi_history.json"))
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    args = argparse.Namespace(
        sandbox_data_dir=str(tmp_path),
        preset_count=1,
        max_iterations=1,
        dashboard_port=None,
        roi_cycles=1,
        synergy_cycles=1,
        recursive_orphans=True,
    )

    cli.full_autonomous_run(args)

    roi_file = tmp_path / "roi_history.json"
    assert roi_file.exists()
    data = json.loads(roi_file.read_text())
    assert data.get("roi_history") == [0.1]

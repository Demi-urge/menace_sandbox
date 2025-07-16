import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json
from pathlib import Path

import menace.replay_training as rt


def _write_events(path: Path):
    rows = [
        {"fingerprint": "a", "state": "error", "feature1": 0.1, "feature2": 0.2, "label": 0, "ts": "2024-01-01T00:00:00"},
        {"fingerprint": "a", "state": "fix", "feature1": 0.1, "feature2": 0.2, "label": 0, "ts": "2024-01-01T00:01:00"},
        {"fingerprint": "a", "state": "success", "feature1": 0.1, "feature2": 0.2, "label": 1, "ts": "2024-01-01T00:02:00"},
    ]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_replay_trainer(tmp_path):
    log = tmp_path / "events.json"
    model_dir = tmp_path / "model"
    _write_events(log)
    trainer = rt.ReplayTrainer()
    model = trainer.train(rt.TrainingConfig(log_path=str(log), model_path=str(model_dir)))
    assert model
    assert model_dir.exists()

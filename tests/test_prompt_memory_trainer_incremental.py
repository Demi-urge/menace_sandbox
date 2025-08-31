import json
import sqlite3
import json
from pathlib import Path

import pytest

from prompt_memory_trainer import PromptMemoryTrainer


class Mem:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE interactions(prompt TEXT, response TEXT, tags TEXT, ts TEXT, embedding TEXT, alerts TEXT)"
        )

    def log_interaction(self, prompt: str, response: str, tags=None) -> None:
        self.conn.execute(
            "INSERT INTO interactions(prompt, response, tags, ts, embedding, alerts) VALUES(?, ?, '', '', '', '')",
            (prompt, response),
        )
        self.conn.commit()


class PDB:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE patch_history(id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, roi_before REAL, roi_after REAL, complexity_before REAL, complexity_after REAL)"
        )


def test_trainer_persists_and_loads_weights(tmp_path: Path) -> None:
    mem = Mem()
    pdb = PDB()
    state = tmp_path / "weights.json"
    trainer = PromptMemoryTrainer(memory=mem, patch_db=pdb, state_path=state)
    trainer.append_records(
        [
            {
                "prompt": "# H",
                "outcome": "SUCCESS",
                "roi_before": 0.0,
                "roi_after": 1.0,
                "complexity_before": 1.0,
                "complexity_after": 0.0,
            }
        ]
    )
    assert state.exists()
    trainer2 = PromptMemoryTrainer(memory=mem, patch_db=pdb, state_path=state)
    assert trainer2.style_weights == trainer.style_weights


def test_append_records_retrains_and_saves(tmp_path: Path) -> None:
    mem = Mem()
    pdb = PDB()
    state = tmp_path / "weights.json"
    trainer = PromptMemoryTrainer(memory=mem, patch_db=pdb, state_path=state)
    trainer.append_records(
        [
            {
                "prompt": "# H",
                "outcome": "SUCCESS",
                "roi_before": 0.0,
                "roi_after": 2.0,
                "complexity_before": 10.0,
                "complexity_after": 5.0,
            },
            {
                "prompt": "# H",
                "outcome": "FAIL",
                "roi_before": 0.0,
                "roi_after": 0.0,
                "complexity_before": 10.0,
                "complexity_after": 5.0,
            },
        ]
    )
    hdr_key = json.dumps(["H"])
    weights = trainer.style_weights
    assert weights["headers"][hdr_key] == pytest.approx(2 / 7)
    assert state.exists()


def test_record_updates_weights(tmp_path: Path) -> None:
    trainer = PromptMemoryTrainer(memory=object(), patch_db=object(), state_path=tmp_path / "w.json")
    updated = trainer.record(headers=["H"], example_order=["success"], tone="neutral", success=True)
    assert updated
    # second call toggles success rate
    updated = trainer.record(headers=["H"], example_order=["success"], tone="neutral", success=False)
    assert updated
    hdr_key = json.dumps(["H"])
    assert trainer.style_weights["headers"][hdr_key] == pytest.approx(0.5)

import json
import sqlite3
from pathlib import Path
import types
import sys

import pytest


class _DummyMem:
    def __init__(self, db_path=":memory:") -> None:
        self.conn = sqlite3.connect(db_path)  # noqa: SQL001
        self.conn.execute(
            "CREATE TABLE interactions("
            "prompt TEXT, response TEXT, tags TEXT, ts TEXT, "
            "embedding TEXT, alerts TEXT)"
        )

    def log_interaction(self, *a, **k) -> None:
        pass


class _DummyPatchDB:
    def __init__(self, path=":memory:") -> None:
        self.conn = sqlite3.connect(path)  # noqa: SQL001
        self.conn.execute(
            "CREATE TABLE patch_history("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, "
            "roi_before REAL, roi_after REAL, complexity_before REAL, "
            "complexity_after REAL)"
        )


sys.modules.setdefault(
    "gpt_memory", types.SimpleNamespace(GPTMemoryManager=_DummyMem)
)
sys.modules.setdefault(
    "code_database", types.SimpleNamespace(PatchHistoryDB=_DummyPatchDB)
)

from prompt_memory_trainer import PromptMemoryTrainer  # noqa: E402


class Mem:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")  # noqa: SQL001
        self.conn.execute(
            "CREATE TABLE interactions("
            "prompt TEXT, response TEXT, tags TEXT, ts TEXT, "
            "embedding TEXT, alerts TEXT)"
        )

    def log_interaction(self, prompt: str, response: str, tags=None) -> None:
        self.conn.execute(
            "INSERT INTO interactions("
            "prompt, response, tags, ts, embedding, alerts) "
            "VALUES(?, ?, '', '', '', '')",
            (prompt, response),
        )
        self.conn.commit()


class PDB:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")  # noqa: SQL001
        self.conn.execute(
            "CREATE TABLE patch_history("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, "
            "roi_before REAL, roi_after REAL, complexity_before REAL, "
            "complexity_after REAL)"
        )


def test_trainer_persists_and_loads_weights(tmp_path: Path) -> None:
    mem = Mem()
    pdb = PDB()
    state = tmp_path / "weights.json"
    db = tmp_path / "styles.db"
    trainer = PromptMemoryTrainer(
        memory=mem, patch_db=pdb, state_path=state, db_path=db
    )
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
    assert db.exists()
    trainer2 = PromptMemoryTrainer(
        memory=mem, patch_db=pdb, state_path=state, db_path=db
    )
    assert trainer2.style_weights == trainer.style_weights


def test_append_records_retrains_and_saves(tmp_path: Path) -> None:
    mem = Mem()
    pdb = PDB()
    state = tmp_path / "weights.json"
    db = tmp_path / "styles.db"
    trainer = PromptMemoryTrainer(
        memory=mem, patch_db=pdb, state_path=state, db_path=db
    )
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
    assert db.exists()


def test_record_updates_weights(tmp_path: Path) -> None:
    trainer = PromptMemoryTrainer(
        memory=object(), patch_db=object(),
        state_path=tmp_path / "w.json",
        db_path=tmp_path / "s.db",
    )
    updated = trainer.record(
        headers=["H"],
        example_order=["success"],
        tone="neutral",
        has_bullets=True,
        has_code=False,
        example_count=2,
        success=True,
    )
    assert updated
    # second call toggles success rate
    updated = trainer.record(
        headers=["H"],
        example_order=["success"],
        tone="neutral",
        has_bullets=True,
        has_code=False,
        example_count=2,
        success=False,
    )
    assert updated
    hdr_key = json.dumps(["H"])
    assert trainer.style_weights["headers"][hdr_key] == pytest.approx(0.5)
    assert trainer.style_weights["has_bullets"]["True"] == pytest.approx(0.5)
    assert trainer.style_weights["has_code"]["False"] == pytest.approx(0.5)
    assert trainer.style_weights["example_count"]["2"] == pytest.approx(0.5)


def test_new_features_influence_weights(tmp_path: Path) -> None:
    mem = Mem()
    pdb = PDB()
    trainer = PromptMemoryTrainer(memory=mem, patch_db=pdb, db_path=tmp_path / "s.db")
    long_text = "word " * 160
    trainer.append_records(
        [
            {
                "prompt": "Example good\nConstraints:\n- A\nResources:\n- B\nExample again",
                "outcome": "SUCCESS",
                "roi_before": 0.0,
                "roi_after": 1.0,
                "complexity_before": 1.0,
                "complexity_after": 0.0,
            },
            {
                "prompt": f"{long_text}\nExample end",
                "outcome": "FAIL",
                "roi_before": 0.0,
                "roi_after": 0.0,
                "complexity_before": 2.0,
                "complexity_after": 1.0,
            },
        ]
    )
    weights = trainer.style_weights
    sect_key = json.dumps(["constraints", "resources"])
    assert weights["structured_sections"][sect_key] == pytest.approx(1.0)
    assert weights["structured_sections"][json.dumps([])] == pytest.approx(0.0)
    assert weights["example_count"]["2"] == pytest.approx(1.0)
    assert weights["example_count"]["1"] == pytest.approx(0.0)
    assert weights["example_placement"]["start"] == pytest.approx(1.0)
    assert weights["example_placement"]["end"] == pytest.approx(0.0)
    assert weights["length"]["short"] == pytest.approx(1.0)
    assert weights["length"]["long"] == pytest.approx(0.0)
    assert weights["has_bullets"]["True"] == pytest.approx(1.0)
    assert weights["has_bullets"]["False"] == pytest.approx(0.0)


def test_style_metrics_exposes_counts(tmp_path: Path) -> None:
    trainer = PromptMemoryTrainer(
        memory=object(),
        patch_db=object(),
        state_path=tmp_path / "w.json",
        db_path=tmp_path / "s.db",
    )
    trainer.record(headers=["H"], success=True)
    metrics = trainer.style_metrics()
    hdr_key = json.dumps(["H"])
    assert metrics["headers"][hdr_key]["count"] == 1.0

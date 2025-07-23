import json
import sqlite3
import time
import types
import importlib
from pathlib import Path


def test_auto_trainer_runs_cycle(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.1}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    called = {}

    def fake_cli(args):
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", fake_cli)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file, weights_file=weights_file, interval=0.05
    )
    trainer.start()
    try:
        for _ in range(20):
            if called:
                break
            time.sleep(0.05)
        assert called
        assert "train" in called["args"]
        assert str(weights_file) in called["args"]
    finally:
        trainer.stop()

    assert trainer._thread is not None
    assert not trainer._thread.is_alive()

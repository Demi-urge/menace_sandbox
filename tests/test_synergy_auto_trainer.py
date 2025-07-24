import json
import sqlite3
import time
import types
import importlib
import logging
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
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=tmp_path / "progress.json",
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


def test_cli_run_once(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.5}),),
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

    rc = sat.cli([
        "--history-file",
        str(hist_file),
        "--weights-file",
        str(weights_file),
        "--progress-file",
        str(tmp_path / "progress.json"),
        "--run-once",
    ])

    assert rc == 0
    assert "train" in called.get("args", [])
    assert str(weights_file) in called.get("args", [])


def test_cli_continuous(monkeypatch) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    events: list[str] = []

    class DummyTrainer:
        def __init__(self, *, history_file: str | Path, weights_file: str | Path, interval: float, progress_file: Path) -> None:
            events.append("init")

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

        def _train_once(self) -> None:
            events.append("train")

    monkeypatch.setattr(sat, "SynergyAutoTrainer", DummyTrainer)

    def raise_after(_t: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sat.time, "sleep", raise_after)

    rc = sat.cli([
        "--history-file",
        "hf",
        "--weights-file",
        "wf",
        "--interval",
        "0.01",
        "--progress-file",
        "prog.json",
    ])

    assert rc == 0
    assert events == ["init", "start", "stop"]


def test_restart_skips_processed(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO synergy_history(entry) VALUES (?)", (json.dumps({"synergy_roi": 0.1}),))
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    called: list[tuple[list[str], list[dict[str, float]]]] = []

    def fake_cli(args):
        with open(args[-1]) as fh:
            data = json.load(fh)
        called.append((list(args), data))
        return 0

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", fake_cli)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer._train_once()

    # add new entry after first run
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.2}),),
    )
    conn.commit()
    conn.close()

    called.clear()
    trainer2 = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer2._train_once()

    assert len(called) == 1
    _, data = called[0]
    assert len(data) == 1
    assert data[0]["synergy_roi"] == 0.2


def test_cli_failure_persists_progress(monkeypatch, tmp_path: Path, caplog) -> None:
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

    progress_file = tmp_path / "progress.json"

    def bad_cli(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", bad_cli)
    caplog.set_level(logging.WARNING)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer._train_once()

    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 1
    assert "boom" in caplog.text

    conn = sqlite3.connect(hist_file)
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.2}),),
    )
    conn.commit()
    conn.close()

    called: list[list[dict[str, float]]] = []

    def ok_cli(args):
        with open(args[-1]) as fh:
            called.append(json.load(fh))
        return 0

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", ok_cli)

    trainer2 = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer2._train_once()

    assert len(called) == 1
    assert called[0][0]["synergy_roi"] == 0.2


def test_trainer_resume_progress(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")
    db_mod = importlib.import_module("menace.synergy_history_db")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.1})
    db_mod.insert_entry(conn, {"synergy_roi": 0.2})
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    calls = {"count": 0}

    def fake_cli(_args: list[str]) -> int:
        calls["count"] += 1
        return 0

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", fake_cli)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )
    trainer.start()
    try:
        for _ in range(40):
            if calls["count"]:
                break
            time.sleep(0.05)
        assert calls["count"] >= 1
    finally:
        trainer.stop()

    assert trainer._last_id == 2
    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 2

    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.3})
    conn.close()

    calls["count"] = 0
    trainer2 = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )

    assert trainer2._last_id == 2

    trainer2.start()
    try:
        for _ in range(40):
            if calls["count"]:
                break
            time.sleep(0.05)
        assert calls["count"] >= 1
    finally:
        trainer2.stop()

    assert trainer2._last_id == 3
    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 3

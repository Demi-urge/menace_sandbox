import json
import sqlite3
import time
import types
import importlib
import logging
import asyncio
import pytest
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

    def fake_train(history, path):
        called["history"] = list(history)
        called["path"] = Path(path)
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

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
        assert called["path"] == weights_file
        assert called["history"][0]["synergy_roi"] == 0.1
    finally:
        trainer.stop()

    assert trainer._thread is None


@pytest.mark.asyncio
async def test_auto_trainer_runs_cycle_async(monkeypatch, tmp_path: Path) -> None:
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

    def fake_train(history, path):
        called["history"] = list(history)
        called["path"] = Path(path)
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=tmp_path / "progress.json",
    )

    trainer.start_async()
    for _ in range(20):
        if called:
            break
        await asyncio.sleep(0.05)
    await trainer.stop_async()

    assert called
    assert called["path"] == weights_file
    assert called["history"][0]["synergy_roi"] == 0.1
    assert trainer._task is None


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

    def fake_train(history, path):
        called["history"] = list(history)
        called["path"] = Path(path)
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

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
    assert called["path"] == weights_file
    assert called["history"][0]["synergy_roi"] == 0.5


def test_cli_continuous(monkeypatch) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    events: list[str] = []

    class DummyTrainer:
        def __init__(
            self,
            *,
            history_file: str | Path,
            weights_file: str | Path,
            interval: float,
            progress_file: Path,
            metrics_port: int | None = None,
        ) -> None:
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


def test_cli_continuous_async(monkeypatch) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    events: list[str] = []

    class DummyTrainer:
        def __init__(
            self,
            *,
            history_file: str | Path,
            weights_file: str | Path,
            interval: float,
            progress_file: Path,
            metrics_port: int | None = None,
        ) -> None:
            events.append("init")

        def start_async(self) -> None:
            events.append("start_async")

        async def stop_async(self) -> None:
            events.append("stop_async")

    monkeypatch.setattr(sat, "SynergyAutoTrainer", DummyTrainer)

    async def raise_after(_t: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sat.asyncio, "sleep", raise_after)

    rc = sat.cli([
        "--history-file",
        "hf",
        "--weights-file",
        "wf",
        "--interval",
        "0.01",
        "--progress-file",
        "prog.json",
        "--async",
    ])

    assert rc == 0
    assert events == ["init", "start_async", "stop_async"]


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

    called: list[list[dict[str, float]]] = []

    def fake_train(history, _path):
        called.append(list(history))
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

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
    data = called[0]
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

    def bad_train(_hist, _path):
        raise RuntimeError("boom")

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", bad_train)
    caplog.set_level(logging.WARNING)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer._train_once()

    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 0
    assert "boom" in caplog.text

    conn = sqlite3.connect(hist_file)
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.2}),),
    )
    conn.commit()
    conn.close()

    called: list[list[dict[str, float]]] = []

    def ok_train(history, _path):
        called.append(list(history))
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", ok_train)

    trainer2 = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )
    trainer2._train_once()

    assert len(called) == 1
    assert len(called[0]) == 2
    assert called[0][0]["synergy_roi"] == 0.1
    assert called[0][1]["synergy_roi"] == 0.2


def test_nonzero_exit_raises(monkeypatch, tmp_path: Path, caplog) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.3}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    def bad_train(_hist, _path):
        raise RuntimeError("boom")

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", bad_train)
    caplog.set_level(logging.WARNING)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )

    trainer._train_once()

    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 0
    assert "boom" in caplog.text


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

    def fake_train(_hist, _path):
        calls["count"] += 1
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

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


def test_iteration_failure_retries(monkeypatch, tmp_path: Path) -> None:
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

    calls: list[list[dict[str, float]]] = []

    def maybe_fail(history, _path):
        calls.append(list(history))
        if len(calls) == 1:
            conn = sqlite3.connect(hist_file)
            conn.execute(
                "INSERT INTO synergy_history(entry) VALUES (?)",
                (json.dumps({"synergy_roi": 0.2}),),
            )
            conn.commit()
            conn.close()
            raise RuntimeError("boom")
        return {}
    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", maybe_fail)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )
    trainer.start()
    try:
        for _ in range(60):
            if len(calls) >= 2:
                break
            time.sleep(0.05)
        assert len(calls) >= 2
    finally:
        trainer.stop()

    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 2


def test_missing_files_created(monkeypatch, tmp_path: Path, caplog) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    data_dir = tmp_path / "data"
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    caplog.set_level(logging.WARNING)

    trainer = sat.SynergyAutoTrainer(
        history_file=tmp_path / "missing_history.db",
        weights_file=tmp_path / "missing_weights.json",
        progress_file=tmp_path / "progress.json",
        interval=0.1,
    )

    assert trainer.history_file == data_dir / "missing_history.db"
    assert trainer.weights_file == data_dir / "missing_weights.json"
    assert trainer.history_file.exists()
    assert trainer.weights_file.exists()
    assert "missing - created empty file" in caplog.text


def test_trainer_start_stop_restart(monkeypatch, tmp_path: Path) -> None:
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

    calls = {"count": 0}

    def fake_train(_hist, _path):
        calls["count"] += 1
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=tmp_path / "progress.json",
    )

    trainer.start()
    try:
        for _ in range(20):
            if calls["count"]:
                break
            time.sleep(0.05)
    finally:
        trainer.stop()

    assert trainer._thread is None

    conn = sqlite3.connect(hist_file)
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.2}),),
    )
    conn.commit()
    conn.close()

    trainer._stop.clear()
    trainer.start()
    try:
        for _ in range(20):
            if calls["count"] >= 2:
                break
            time.sleep(0.05)
    finally:
        trainer.stop()

    assert trainer._thread is None
    assert calls["count"] >= 2


def test_train_failure_increments_metric(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 1.0}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    def bad_train(_hist, _path):
        raise RuntimeError("boom")

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", bad_train)
    sat.synergy_trainer_failures_total.set(0.0)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )

    trainer._train_once()

    assert sat.synergy_trainer_failures_total._value.get() == 1.0


def test_cli_failure_updates_metric(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "hf.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    def fail_once(self) -> None:
        raise sat.SynergyWeightCliError(2)

    monkeypatch.setattr(sat.SynergyAutoTrainer, "_train_once", fail_once)
    sat.synergy_trainer_failures_total.set(0.0)

    rc = sat.cli([
        "--history-file",
        str(hist_file),
        "--weights-file",
        str(weights_file),
        "--progress-file",
        str(progress_file),
        "--run-once",
    ])

    assert rc == 2
    assert sat.synergy_trainer_failures_total._value.get() == 1.0


def test_alert_on_single_failure(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "hf.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 1.0}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    alerts: list[tuple] = []

    def bad_train(_hist, _path):
        raise RuntimeError("boom")

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", bad_train)
    monkeypatch.setattr(sat, "dispatch_alert", fake_alert)
    sat.synergy_weight_update_failures_total.set(0.0)
    sat.synergy_weight_update_alerts_total.set(0.0)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=tmp_path / "progress.json",
    )

    trainer._train_once()

    assert len(alerts) == 1
    assert sat.synergy_weight_update_failures_total._value.get() == 1.0
    assert sat.synergy_weight_update_alerts_total._value.get() == 1.0

def test_metrics_server_started(monkeypatch, tmp_path: Path) -> None:
    me = importlib.import_module("menace.metrics_exporter")
    sat = importlib.import_module("menace.synergy_auto_trainer")

    calls: list[int] = []

    def fake_start(port: int) -> None:
        calls.append(port)

    monkeypatch.setattr(me, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat.SynergyAutoTrainer, "_train_once", lambda self: None)

    trainer = sat.SynergyAutoTrainer(
        history_file=tmp_path / "hf.db",
        weights_file=tmp_path / "weights.json",
        progress_file=tmp_path / "progress.json",
        interval=0.05,
        metrics_port=9999,
    )

    trainer.start()
    try:
        assert calls == [9999]
    finally:
        trainer.stop()


@pytest.mark.asyncio
async def test_metrics_server_started_async(monkeypatch, tmp_path: Path) -> None:
    me = importlib.import_module("menace.metrics_exporter")
    sat = importlib.import_module("menace.synergy_auto_trainer")

    calls: list[int] = []

    def fake_start(port: int) -> None:
        calls.append(port)

    monkeypatch.setattr(me, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat.SynergyAutoTrainer, "_train_once", lambda self: None)

    trainer = sat.SynergyAutoTrainer(
        history_file=tmp_path / "hf.db",
        weights_file=tmp_path / "weights.json",
        progress_file=tmp_path / "progress.json",
        interval=0.05,
        metrics_port=8888,
    )

    trainer.start_async()
    await asyncio.sleep(0)
    await trainer.stop_async()

    assert calls == [8888]

@pytest.mark.asyncio
async def test_async_metrics_and_restart(monkeypatch, tmp_path: Path) -> None:
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

    progress_file = tmp_path / "progress.json"

    called = {}

    def fake_train(history, path):
        called["history"] = list(history)
        called["path"] = Path(path)
        Path(path).write_text(json.dumps({"w": 1}))
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)
    sat.synergy_trainer_iterations.set(0.0)
    sat.synergy_trainer_last_id.set(0.0)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )

    trainer.start_async()
    for _ in range(20):
        if called:
            break
        await asyncio.sleep(0.05)
    await trainer.stop_async()

    assert called
    assert sat.synergy_trainer_iterations._value.get() == 1.0
    assert sat.synergy_trainer_last_id._value.get() == 1.0

    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 1
    assert json.loads(weights_file.read_text()) == {"w": 1}

    trainer2 = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.1,
        progress_file=progress_file,
    )

    assert trainer2._last_id == 1
    assert trainer2.weights_file == weights_file


@pytest.mark.asyncio
async def test_async_failure_dispatch(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 1.0}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    alerts: list[tuple] = []

    def bad_train(_hist, _path):
        raise RuntimeError("boom")

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", bad_train)
    monkeypatch.setattr(sat, "dispatch_alert", fake_alert)
    sat.synergy_trainer_failures_total.set(0.0)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )

    trainer.start_async()
    for _ in range(20):
        if alerts:
            break
        await asyncio.sleep(0.05)
    await trainer.stop_async()

    assert sat.synergy_trainer_failures_total._value.get() == 1.0
    assert len(alerts) == 1



def test_train_retry_updates_progress(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 1.0}),),
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    progress_file = tmp_path / "progress.json"

    calls = {"n": 0}

    def maybe_fail(history, _path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", maybe_fail)
    sat.synergy_trainer_failures_total.set(0.0)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=progress_file,
    )

    trainer._train_once()
    assert sat.synergy_trainer_failures_total._value.get() == 1.0
    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 0

    trainer._train_once()
    assert sat.synergy_trainer_failures_total._value.get() == 1.0
    data = json.loads(progress_file.read_text())
    assert data["last_id"] == 1


@pytest.mark.asyncio
async def test_async_start_stop_cancels_task(monkeypatch, tmp_path: Path) -> None:
    me = importlib.import_module("menace.metrics_exporter")
    sat = importlib.import_module("menace.synergy_auto_trainer")

    calls: list[int] = []

    def fake_start(port: int) -> None:
        calls.append(port)
    monkeypatch.setattr(me, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat, "start_metrics_server", fake_start)
    monkeypatch.setattr(sat.SynergyAutoTrainer, "_train_once", lambda self: None)

    hist_file = tmp_path / "hf.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        progress_file=tmp_path / "progress.json",
        interval=0.05,
        metrics_port=4321,
    )

    trainer.start_async()
    task = trainer._task
    await asyncio.sleep(0)
    await trainer.stop_async()

    assert calls == [4321]
    assert trainer._task is None
    assert task is not None and task.done()

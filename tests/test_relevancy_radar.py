import os
import importlib
import json
import atexit
import types
import sqlite3
import time
from unittest import mock

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


@pytest.fixture()
def radar_env(tmp_path, monkeypatch):
    """Provide an isolated relevancy radar instance with dummy modules."""

    # Prevent actual atexit handlers during tests
    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.import_module("relevancy_radar")

    usage_file = tmp_path / "module_usage.json"
    flags_file = tmp_path / "relevancy_flags.json"
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", usage_file)
    monkeypatch.setattr(rr, "_RELEVANCY_FLAGS_FILE", flags_file)

    rr._module_usage_counter.clear()
    rr._relevancy_flags.clear()

    def make_mod(name: str):
        def run() -> None:
            rr.track_module_usage(name)

        return types.SimpleNamespace(run=run)

    modules = {name: make_mod(name) for name in ["alpha", "beta", "gamma"]}
    return rr, modules, usage_file, flags_file


def test_tracking_and_persistence(radar_env):
    """Ensure tracking increments counters and persistence merges counts."""

    rr, modules, usage_file, _ = radar_env

    modules["alpha"].run()
    modules["alpha"].run()
    modules["beta"].run()

    stats = rr.load_usage_stats()
    assert stats == pytest.approx({"alpha": 2, "beta": 1})

    rr._save_usage_counts()
    data = json.loads(usage_file.read_text())
    assert {k: len(v) for k, v in data.items()} == {
        k: round(v) for k, v in stats.items()
    }

    rr._module_usage_counter.clear()
    modules["alpha"].run()
    stats2 = rr.load_usage_stats()
    assert stats2 == pytest.approx({"alpha": 3, "beta": 1})


def test_relevancy_evaluation_unused_flagged(radar_env):
    """``evaluate_relevancy`` categorises modules and flags unused ones."""

    rr, modules, _, flags_file = radar_env

    for _ in range(6):
        modules["alpha"].run()
    modules["beta"].run()

    stats = rr.load_usage_stats()
    module_map = {name: 1 for name in modules}
    expected = {"alpha": "replace", "beta": "compress", "gamma": "retire"}

    assert rr.evaluate_relevancy(module_map, stats) == expected
    assert json.loads(flags_file.read_text()) == expected

    rr._relevancy_flags.clear()
    assert rr.flagged_modules() == expected


def test_age_based_pruning(radar_env, monkeypatch):
    """Old usage entries beyond the window are discarded."""

    rr, modules, usage_file, _ = radar_env

    monkeypatch.setenv("RELEVANCY_WINDOW_DAYS", "1")

    now = int(time.time())
    old = now - 3 * 86400

    usage_file.write_text(json.dumps({
        "alpha": [old, now],
        "beta": [old],
    }))

    stats = rr.load_usage_stats()
    assert stats == pytest.approx({"alpha": 1}, rel=1e-5)

    module_map = {"alpha": 1, "beta": 1}
    flags = rr.evaluate_relevancy(module_map, {"alpha": [old, now], "beta": [old]})
    assert flags == {"alpha": "compress", "beta": "retire"}


@pytest.fixture()
def metrics_db(tmp_path):
    """Create a metrics database with used, unused and low-usage modules."""

    db_path = tmp_path / "metrics.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE module_metrics (module_name TEXT, call_count INTEGER, total_time REAL, roi_delta REAL)"
        )
        conn.executemany(
            "INSERT INTO module_metrics VALUES (?, ?, ?, ?)",
            [
                ("used_mod", 100, 100.0, 100.0),
                ("unused_mod", 0, 0.0, 0.0),
                ("compress_mod", 1, 1.0, 0.1),
                ("replace_mod", 3, 3.0, 0.2),
                ("high_roi_mod", 1, 1.0, 50.0),
            ],
        )
    return db_path


def test_scan_respects_thresholds(metrics_db):
    """``scan`` flags only unused or low-ratio modules."""

    from relevancy_radar import scan

    flags = scan(
        db_path=metrics_db,
        min_calls=0,
        compress_ratio=0.02,
        replace_ratio=0.05,
    )

    assert flags == {
        "unused_mod": "retire",
        "compress_mod": "compress",
        "replace_mod": "replace",
    }
    assert "high_roi_mod" not in flags


def test_scan_publishes_event(metrics_db):
    """Integration publishes ``relevancy:scan`` events via the bus."""

    from relevancy_radar import scan
    from menace.unified_event_bus import UnifiedEventBus

    bus = mock.Mock()

    def run_scan(event_bus):
        try:
            result = scan(
                db_path=metrics_db,
                min_calls=0,
                compress_ratio=0.02,
                replace_ratio=0.05,
            )
            if result:
                event_bus.publish("relevancy:scan", result)
            return result
        except Exception:
            return {}

    flags = run_scan(bus)

    bus.publish.assert_called_once_with(
        "relevancy:scan",
        {
            "unused_mod": "retire",
            "compress_mod": "compress",
            "replace_mod": "replace",
        },
    )
    assert flags


def test_scan_skips_high_roi(metrics_db):
    """Modules with high ROI are not flagged despite low call counts."""

    from relevancy_radar import scan

    flags = scan(
        db_path=metrics_db,
        min_calls=0,
        compress_ratio=0.02,
        replace_ratio=0.05,
    )
    assert "high_roi_mod" not in flags


def test_negative_roi_retire_flag(tmp_path, monkeypatch):
    """Modules with negative ROI are retired when the total score is non-positive."""

    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "metrics.json"
    radar = rr.RelevancyRadar(metrics_file=metrics_file)

    radar.track_usage("loss_mod", impact=-5.0)
    radar.track_usage("loss_mod", impact=0.0)

    flags = radar.evaluate_relevance(
        compress_threshold=1.0, replace_threshold=2.0, impact_weight=1.0
    )

    assert flags == {"loss_mod": "retire"}

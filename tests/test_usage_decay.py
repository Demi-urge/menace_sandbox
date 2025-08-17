import importlib
import json
import time
import atexit

import pytest


def test_load_usage_stats_weights_by_age(tmp_path, monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))

    usage_file = tmp_path / "module_usage.json"
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", usage_file)

    now = time.time()
    window = max(now - rr._relevancy_cutoff(), 1.0)
    recent_ts = int(now)
    old_ts = int(now - 0.75 * window)

    usage_file.parent.mkdir(parents=True, exist_ok=True)
    usage_file.write_text(json.dumps({"recent": [recent_ts], "old": [old_ts]}))

    stats = rr.load_usage_stats()

    assert stats["recent"] > stats["old"]
    assert stats["recent"] == pytest.approx(
        rr._decay_factor(recent_ts, now=now, window=window)
    )
    assert stats["old"] == pytest.approx(
        rr._decay_factor(old_ts, now=now, window=window)
    )


def test_weighted_counts_affect_final_score(tmp_path, monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    monkeypatch.setattr(rr, "_RELEVANCY_CALL_GRAPH_FILE", tmp_path / "call_graph.json")
    monkeypatch.setattr(rr.RelevancyRadar, "_persist_metrics", lambda self: None)

    radar = rr.RelevancyRadar(metrics_file=tmp_path / "metrics.json")

    now = time.time()
    window = max(now - rr._relevancy_cutoff(), 1.0)
    old_weight = rr._decay_factor(int(now - window), now=now, window=window)

    radar._metrics = {
        "recent": {"imports": 1.0, "executions": 1.0, "impact": 0.0, "output_impact": 0.0},
        "old": {
            "imports": old_weight,
            "executions": old_weight,
            "impact": 0.0,
            "output_impact": 0.0,
        },
    }

    flags = radar.evaluate_relevance(compress_threshold=1.0, replace_threshold=3.0)
    assert flags == {"old": "compress", "recent": "replace"}

    flags_final = radar.evaluate_final_contribution(
        compress_threshold=1.0,
        replace_threshold=3.0,
        core_modules=["recent", "old"],
    )
    assert flags_final == {"old": "compress", "recent": "replace"}


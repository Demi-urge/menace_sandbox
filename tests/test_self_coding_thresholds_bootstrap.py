import time

import menace.self_coding_thresholds as thresholds


def test_threshold_cache_used_during_bootstrap(tmp_path, monkeypatch):
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text("bots:\n  cached-bot:\n    roi_drop: -0.5\n")

    monkeypatch.setattr(thresholds, "_CONFIG_CACHE", {})
    monkeypatch.setattr(thresholds, "_LAST_CACHE_KEY", None)

    initial = thresholds._load_config(cfg, timeout_s=0.1)
    assert initial.get("bots", {}).get("cached-bot") is not None

    cfg.unlink()

    start = time.perf_counter()
    cached = thresholds._load_config(cfg, bootstrap_safe=True, timeout_s=0.05)
    duration = time.perf_counter() - start

    assert cached.get("bots", {}).get("cached-bot") is not None
    assert duration < 0.5

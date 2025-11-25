import time
from pathlib import Path

import menace.self_coding_thresholds as thresholds


def test_threshold_cache_used_during_bootstrap(tmp_path, monkeypatch):
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text("bots:\n  cached-bot:\n    roi_drop: -0.5\n")

    monkeypatch.setattr(thresholds, "_CONFIG_CACHE", {})
    monkeypatch.setattr(thresholds, "_LAST_CACHE_KEY", None)
    monkeypatch.setattr(thresholds, "_DEFERRED_CONFIG_WRITES", {})

    initial = thresholds._load_config(cfg, timeout_s=0.1)
    assert initial.get("bots", {}).get("cached-bot") is not None

    cfg.unlink()

    start = time.perf_counter()
    cached = thresholds._load_config(cfg, bootstrap_safe=True, timeout_s=0.05)
    duration = time.perf_counter() - start

    assert cached.get("bots", {}).get("cached-bot") is not None
    assert duration < 0.5


def test_update_thresholds_defers_writes_in_bootstrap(tmp_path, monkeypatch):
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text("bots:\n  cached-bot:\n    roi_drop: -0.5\n")

    monkeypatch.setattr(thresholds, "_CONFIG_CACHE", {})
    monkeypatch.setattr(thresholds, "_LAST_CACHE_KEY", None)
    monkeypatch.setattr(thresholds, "_DEFERRED_CONFIG_WRITES", {})

    thresholds._load_config(cfg)

    def _fail_write(self: Path, *_args, **_kwargs):
        raise AssertionError("bootstrap updates should not write to disk")

    monkeypatch.setattr(Path, "write_text", _fail_write)

    thresholds.update_thresholds(
        "cached-bot", roi_drop=-0.8, path=cfg, bootstrap_safe=True
    )
    cached = thresholds.get_cached_config(cfg)
    assert cached.get("bots", {}).get("cached-bot", {}).get("roi_drop") == -0.8


def test_flush_deferred_threshold_writes(tmp_path, monkeypatch):
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text("bots:\n  cached-bot:\n    roi_drop: -0.5\n")

    monkeypatch.setattr(thresholds, "_CONFIG_CACHE", {})
    monkeypatch.setattr(thresholds, "_LAST_CACHE_KEY", None)
    monkeypatch.setattr(thresholds, "_DEFERRED_CONFIG_WRITES", {})

    thresholds.update_thresholds(
        "cached-bot", roi_drop=-0.9, path=cfg, bootstrap_safe=True
    )

    flushed = thresholds.flush_deferred_threshold_writes(cfg)
    assert flushed is True

    data = thresholds._load_config(cfg)
    assert data.get("bots", {}).get("cached-bot", {}).get("roi_drop") == -0.9

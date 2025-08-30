from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_services_config import SelfLearningConfig, SelfTestConfig


def test_self_learning_config_defaults(tmp_path, monkeypatch):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    monkeypatch.setenv(
        "SELF_LEARNING_PERSIST_EVENTS", str(events_dir / "bus.json")
    )
    monkeypatch.setenv(
        "SELF_LEARNING_PERSIST_PROGRESS", str(progress_dir / "results.json")
    )
    cfg = SelfLearningConfig()
    assert cfg.prune_interval == 50
    assert cfg.persist_events == events_dir / "bus.json"
    assert cfg.persist_progress == progress_dir / "results.json"


def test_self_learning_config_invalid_prune_interval(monkeypatch):
    monkeypatch.setenv("PRUNE_INTERVAL", "-1")
    with pytest.raises(ValidationError):
        SelfLearningConfig()


def test_self_test_config_valid(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    monkeypatch.setenv("SELF_TEST_LOCK_FILE", str(lock_dir / "self.lock"))
    monkeypatch.setenv("SELF_TEST_REPORT_DIR", str(report_dir))
    cfg = SelfTestConfig()
    assert cfg.lock_file == lock_dir / "self.lock"
    assert cfg.report_dir == report_dir


def test_self_test_config_missing_report_dir(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    monkeypatch.setenv("SELF_TEST_LOCK_FILE", str(lock_dir / "self.lock"))
    monkeypatch.setenv("SELF_TEST_REPORT_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValidationError):
        SelfTestConfig()

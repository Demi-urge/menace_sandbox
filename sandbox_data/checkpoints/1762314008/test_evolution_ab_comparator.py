import json
import logging
import evolution_ab_comparator as eac


def _write_log(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_compare_and_report(tmp_path, monkeypatch):
    log_a = tmp_path / "log_a.jsonl"
    log_b = tmp_path / "log_b.jsonl"
    records_a = [
        {
            "action": "a",
            "risk": 0.1,
            "reward": 5,
            "domains": ["a.com"],
            "violation": False,
            "bypass_attempt": False,
            "security_ai_invoked": 1,
            "filter_triggered": True,
        },
        {
            "action": "b",
            "risk": 0.2,
            "reward": 6,
            "domains": ["b.com"],
            "violation": True,
            "bypass_attempt": False,
            "security_ai_invoked": 1,
            "filter_triggered": False,
        },
    ]
    records_b = [
        {
            "action": "a",
            "risk": 0.3,
            "reward": 4,
            "domains": ["a.com"],
            "violation": True,
            "bypass_attempt": True,
            "security_ai_invoked": 0,
            "filter_triggered": False,
        },
        {
            "action": "b",
            "risk": 0.25,
            "reward": 5,
            "domains": ["b.com"],
            "violation": True,
            "bypass_attempt": True,
            "security_ai_invoked": 0,
            "filter_triggered": False,
        },
    ]
    _write_log(log_a, records_a)
    _write_log(log_b, records_b)

    logs_a, logs_b = eac.load_behavior_logs(str(log_a), str(log_b))
    metrics = eac.compare_behavioral_metrics(logs_a, logs_b)
    drifts = eac.detect_behavioral_drift(logs_a, logs_b)

    thresh_path = tmp_path / "thresh.json"
    thresh = {
        "max_avg_risk_increase_pct": 50,
        "max_reward_decrease_pct": 50,
        "max_violation_increase": 10,
        "max_bypass_increase": 10,
    }
    thresh_path.write_text(json.dumps(thresh))
    monkeypatch.setattr(eac, "CONFIG_PATH", str(thresh_path))
    eac.THRESHOLDS.update(eac._load_thresholds())

    out_dir = tmp_path / "reports"
    eac.generate_comparison_report(metrics, drifts, str(out_dir))

    files = list(out_dir.iterdir())
    assert any(f.suffix == ".json" for f in files)
    assert metrics["differences"]["avg_risk"] > 0
    assert drifts


def test_threshold_load_warning(tmp_path, monkeypatch, caplog):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{broken}")
    monkeypatch.setattr(eac, "CONFIG_PATH", str(bad_file))
    caplog.set_level(logging.WARNING)
    cfg = eac._load_thresholds()
    assert cfg == eac.DEFAULT_THRESHOLDS
    assert "failed to load thresholds" in caplog.text


def test_log_read_warning(tmp_path, monkeypatch, caplog):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text("{}\n")

    def bad_open(*a, **k):
        raise RuntimeError("boom")

    import builtins
    monkeypatch.setattr(builtins, "open", bad_open)
    caplog.set_level(logging.WARNING)
    logs_a, logs_b = eac.load_behavior_logs(str(data_file), str(data_file))
    assert not logs_a and not logs_b
    assert "failed to read" in caplog.text


def test_spawn_and_compare_variants(tmp_path):
    from evolution_history_db import EvolutionHistoryDB, EvolutionEvent

    db_path = tmp_path / "evol.db"
    db = EvolutionHistoryDB(db_path)
    root_id = db.add(EvolutionEvent(action="root", before_metric=0.0, after_metric=0.0, roi=0.0))

    var_a = eac.spawn_variant(root_id, "A", db_path=str(db_path))
    eac.record_variant_outcome(var_a, after_metric=1.0, roi=0.5, performance=1.5, db_path=str(db_path))
    var_b = eac.spawn_variant(root_id, "B", db_path=str(db_path))
    eac.record_variant_outcome(var_b, after_metric=1.0, roi=0.3, performance=1.0, db_path=str(db_path))

    report = eac.compare_variant_paths(root_id, db_path=str(db_path))
    assert len(report["variants"]) == 2
    assert report["best"]["event_id"] == var_a

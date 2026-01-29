import json
from pathlib import Path
import tempfile
from datetime import datetime

from roi_tracker import ROITracker
from single_agent_roi_runner import MetricsWriter, ROIConfig, run_cycle


def _run_dry_cycle() -> tuple[ROITracker, ROIConfig]:
    tracker = ROITracker()
    cfg = ROIConfig(
        roi_target=1.0,
        min_confidence=0.6,
        catastrophic_multiplier=1.5,
        max_cycles=1,
        dry_run=True,
    )
    run_cycle(None, tracker, cfg, "menace")
    return tracker, cfg


def test_deterministic_roi_scores_across_runs() -> None:
    tracker_one = ROITracker()
    tracker_two = ROITracker()
    cfg = ROIConfig(
        roi_target=1.0,
        min_confidence=0.6,
        catastrophic_multiplier=1.5,
        max_cycles=1,
        dry_run=True,
    )

    result_one = run_cycle(None, tracker_one, cfg, "menace")
    result_two = run_cycle(None, tracker_two, cfg, "menace")

    assert result_one.roi_after == result_two.roi_after
    assert result_one.raroi == result_two.raroi
    assert result_one.confidence == result_two.confidence
    assert result_one.safety_factor == result_two.safety_factor


def test_temp_files_cleanup_after_metrics_write() -> None:
    tracker, cfg = _run_dry_cycle()
    result = run_cycle(None, tracker, cfg, "menace")
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "metrics.jsonl"
        writer = MetricsWriter(metrics_path)
        writer.record_cycle(1, result)
        writer.record_summary(datetime.utcnow(), datetime.utcnow(), 1, result)
        assert metrics_path.exists()
    assert not metrics_path.exists()


def test_metrics_output_schema_non_empty() -> None:
    tracker, cfg = _run_dry_cycle()
    result = run_cycle(None, tracker, cfg, "menace")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "metrics.jsonl"
        writer = MetricsWriter(metrics_path)
        writer.record_cycle(1, result)

        payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
        assert payload["roi_before"] is not None
        assert payload["roi_after"] is not None
        assert payload["raroi"] is not None
        assert payload["confidence"] is not None
        assert payload["safety_factor"] is not None
        assert isinstance(payload["suggestions"], list)

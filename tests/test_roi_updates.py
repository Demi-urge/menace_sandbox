from menace_sandbox.roi_tracker import ROITracker
from menace_sandbox.borderline_bucket import BorderlineBucket


def test_roi_tracker_adds_borderline_candidate(tmp_path):
    bucket = BorderlineBucket(tmp_path / "bucket.jsonl")
    tracker = ROITracker(raroi_borderline_threshold=0.2, borderline_bucket=bucket)
    tracker.update(1.0, 1.1, modules=["w1"], confidence=0.5)
    assert bucket.get_candidate("w1") is not None

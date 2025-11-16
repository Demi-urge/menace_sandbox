from menace_sandbox.roi_tracker import ROITracker


def test_impact_severity_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "impact.yaml"
    cfg.write_text("standard: 0.1\n")
    monkeypatch.setenv("IMPACT_SEVERITY_CONFIG", str(cfg))
    tracker = ROITracker()
    assert tracker.impact_severity("standard") == 0.1

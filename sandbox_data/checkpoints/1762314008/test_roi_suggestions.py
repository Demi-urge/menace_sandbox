import menace_sandbox.roi_tracker as roi_tracker


class StubLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, *args, **kwargs):
        self.records.append({"msg": msg, "extra": kwargs.get("extra")})

    def exception(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class StubGauge:
    def __init__(self):
        self.calls = []

    def labels(self, **kwargs):
        self.calls.append(kwargs)
        return self

    def set(self, value):
        self.value = value


def test_suggestions_logged_and_exported(monkeypatch):
    tracker = roi_tracker.ROITracker(raroi_borderline_threshold=0.5)
    stub_logger = StubLogger()
    monkeypatch.setattr(roi_tracker, "logger", stub_logger)
    monkeypatch.setattr(roi_tracker, "_ROI_BOTTLENECK_GAUGE", StubGauge())
    metrics = {
        "profitability": 0.0,
        "efficiency": 0.0,
        "reliability": 0.0,
        "resilience": 0.0,
        "maintainability": 0.0,
        "security": 0.4,
        "latency": 0.0,
        "energy": 0.0,
        "alignment_violation": True,
    }
    tracker.update(0.0, 0.0, metrics=metrics, profile_type="scraper_bot")
    logged = [r for r in stub_logger.records if r["extra"] and r["extra"].get("suggestions")]
    assert logged and logged[0]["extra"]["suggestions"]
    assert roi_tracker._ROI_BOTTLENECK_GAUGE.calls

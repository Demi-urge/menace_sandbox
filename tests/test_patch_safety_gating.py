from vector_service import PatchLogger


class Gauge:
    def __init__(self):
        self.calls = []

    def labels(self, risk):
        self.calls.append(("labels", risk))
        return self

    def inc(self, amount=1.0):
        self.calls.append(("inc", amount))
        return self


def test_patch_logger_records_risky_and_safe(monkeypatch):
    import vector_service.patch_logger as pl

    gauge = Gauge()
    monkeypatch.setattr(pl, "_VECTOR_RISK", gauge)
    logger = PatchLogger()
    meta = {
        "db:risky": {"alignment_severity": 0.9},
        "db:safe": {"alignment_severity": 0.0},
    }
    logger.track_contributors(["db:risky", "db:safe"], True, retrieval_metadata=meta)
    assert ("labels", "risky") in gauge.calls
    assert ("labels", "safe") in gauge.calls

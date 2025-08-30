import asyncio

import pytest

from vector_service.weight_adjuster import WeightAdjuster


class DummyDB:
    def __init__(self):
        self._metadata = {"v1": {}}
        self.saved = 0

    def save_index(self):  # pragma: no cover - trivial
        self.saved += 1


def test_adjust_increases_on_success():
    db = DummyDB()
    adj = WeightAdjuster({"db": db}, success_delta=0.2, failure_delta=0.1)
    adj.adjust(["db:v1"], 0.8, "high-ROI")
    assert db._metadata["v1"]["weight"] == pytest.approx(1.16)
    assert db.saved == 1


def test_adjust_decreases_on_failure():
    db = DummyDB()
    adj = WeightAdjuster({"db": db}, success_delta=0.2, failure_delta=0.1)
    adj.adjust(["db:v1"], 0.8, "low-ROI")
    assert db._metadata["v1"]["weight"] == pytest.approx(0.92)


def test_bulk_adjust_async():
    db = DummyDB()
    adj = WeightAdjuster({"db": db}, success_delta=0.2, failure_delta=0.1)
    asyncio.run(
        adj.bulk_adjust([
            (["db:v1"], 0.5, "high-ROI"),
            (["db:v1"], 0.5, "low-ROI"),
        ])
    )
    assert db._metadata["v1"]["weight"] == pytest.approx(1.05)
    assert db.saved == 2

import pytest
# flake8: noqa
pytest.skip("optional dependencies not installed", allow_module_level=True)

import types
from pathlib import Path

import menace.ad_integration as ai
import menace.revenue_amplifier as ra


def test_process_sales(monkeypatch, tmp_path):
    sales = [
        {"model_id": "m1", "amount": 5.0, "platform": "ads", "segment": "seg"}
    ]

    class FakeSession:
        def get(self, url, timeout=10):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return sales

            return Resp()

    db = ra.RevenueEventsDB(tmp_path / "rev.db")
    monitor = ra.SalesSpikeMonitor(db)
    charged = []

    class DummyFinance:
        def route_payment(self, amount: float, model_id: str):
            charged.append((amount, model_id))
            return "success"

    client = ai.AdIntegration(
        base_url="http://test",
        session=FakeSession(),
        finance_bot=DummyFinance(),
        spike_monitor=monitor,
    )
    client.process_sales()

    rows = db.fetch("m1")
    assert rows and rows[0][1] == 5.0
    assert charged == [(5.0, "m1")]

def test_orchestrator_calls_ad_client(monkeypatch):
    calls = []

    class DummyAd:
        async def process_sales_async(self):
            calls.append("called")

    class StubPipeline:
        def run(self, model: str):
            return None

    import menace.menace_orchestrator as mo

    orch = mo.MenaceOrchestrator(ad_client=DummyAd(), context_builder=mo.ContextBuilder())
    orch.pipeline = StubPipeline()
    orch.run_cycle(["m"])
    assert calls == ["called"]

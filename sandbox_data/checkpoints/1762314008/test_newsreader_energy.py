import pytest

pytest.importorskip("sklearn")

import menace.newsreader_bot as nb

class DummyResp:
    status_code = 200
    def json(self):
        return {
            "articles": [
                {"title": "Fast ROI guaranteed", "description": "X", "source": {"name": "s"}},
                {"title": "Huge scalable platform", "description": "Y", "source": {"name": "s"}},
            ]
        }


def fake_get(url, headers=None, params=None, timeout=10):
    return DummyResp()


def test_fetch_news_energy(monkeypatch):
    monkeypatch.setattr(nb, "requests", type("R", (), {"get": staticmethod(fake_get)}))
    low = nb.fetch_news("u", token="", energy=0.2)
    high = nb.fetch_news("u", token="", energy=0.8)
    assert len(low) == 1 and "fast roi" in low[0].title.lower()
    assert len(high) == 1 and "scalable" in high[0].title.lower()

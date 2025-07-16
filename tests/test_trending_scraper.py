import types
import time
import menace.trending_scraper as ts

ts.requests = types.SimpleNamespace(Session=lambda: None)


def test_energy_filtering():
    scraper = ts.TrendingScraper()
    items = [
        ts.TrendingItem(platform="x", product_name="Fast ROI idea"),
        ts.TrendingItem(platform="x", product_name="Scalable infrastructure"),
    ]
    low = scraper._filter_items(items, 0.2)
    high = scraper._filter_items(items, 0.8)
    assert len(low) == 1 and "fast roi" in low[0].product_name.lower()
    assert len(high) == 1 and "scalable" in high[0].product_name.lower()


def test_correlate_trends():
    scraper = ts.TrendingScraper()
    items = [
        ts.TrendingItem(platform="A", product_name="Foo"),
        ts.TrendingItem(platform="B", product_name="Foo"),
        ts.TrendingItem(platform="A", product_name="Bar"),
    ]
    correlated = scraper.correlate_trends(items)
    names = [i.product_name for i in correlated]
    assert names.count("Foo") == 2
    assert "Bar" not in names


def test_aggregate_signals_and_collect_all(monkeypatch):
    scraper = ts.TrendingScraper()

    foo_a = ts.TrendingItem(platform="Reddit", product_name="Foo")
    foo_b = ts.TrendingItem(platform="Gumroad", product_name="Foo")
    foo_c = ts.TrendingItem(platform="Google", product_name="Foo")
    bar_a = ts.TrendingItem(platform="Shopify", product_name="Bar")
    bar_b = ts.TrendingItem(platform="ProductHunt", product_name="Bar")
    baz = ts.TrendingItem(platform="Fiverr", product_name="Baz")

    monkeypatch.setattr(scraper, "scrape_reddit", lambda energy=None: [foo_a])
    monkeypatch.setattr(scraper, "scrape_shopify", lambda energy=None: [bar_a])
    monkeypatch.setattr(scraper, "scrape_gumroad", lambda energy=None: [foo_b])
    monkeypatch.setattr(scraper, "scrape_fiverr", lambda energy=None: [baz])
    monkeypatch.setattr(scraper, "scrape_google_trends", lambda energy=None: [foo_c])
    monkeypatch.setattr(scraper, "scrape_hackernews", lambda energy=None: [])
    monkeypatch.setattr(scraper, "scrape_trendforecast", lambda energy=None: [])
    monkeypatch.setattr(scraper, "scrape_social_firehose", lambda energy=None: [])
    monkeypatch.setattr(scraper, "scrape_producthunt", lambda energy=None: [bar_b])
    monkeypatch.setattr(scraper, "scrape_github_trending", lambda energy=None: [])

    result = scraper.collect_all()

    names = [i.product_name for i in result]
    assert names == ["Foo", "Foo", "Foo", "Bar", "Bar"]
    signals = [i.trend_signal for i in result]
    assert all(s == 3 for s in signals[:3])
    assert all(s == 2 for s in signals[3:])


def test_aggregate_signals_duplicates():
    items = [
        ts.TrendingItem(platform="A", product_name="Foo"),
        ts.TrendingItem(platform="B", product_name="Foo"),
        ts.TrendingItem(platform="A", product_name="Bar"),
    ]
    ts.TrendingScraper.aggregate_signals(items)

    foo_signals = [i.trend_signal for i in items if i.product_name == "Foo"]
    bar_signal = [i.trend_signal for i in items if i.product_name == "Bar"][0]

    assert foo_signals == [2.0, 2.0]
    assert bar_signal == 1.0
    assert foo_signals[0] > bar_signal


def test_detect_microtrends():
    now = time.time()
    items = [
        ts.TrendingItem(
            platform="TrendForecast", product_name="Alpha", timestamp=now - 3600
        ),
        ts.TrendingItem(
            platform="SocialFirehose", product_name="Alpha", timestamp=now - 1800
        ),
        ts.TrendingItem(
            platform="TrendForecast", product_name="Old", timestamp=now - 90000
        ),
        ts.TrendingItem(
            platform="SocialFirehose", product_name="Old", timestamp=now - 80000
        ),
    ]

    result = ts.TrendingScraper.detect_microtrends(items)
    names = [i.product_name for i in result]
    assert names.count("Alpha") == 2
    assert "Old" not in names


def test_detect_microtrends_intensity():
    now = time.time()
    items = [
        ts.TrendingItem(platform="A", product_name="Foo", timestamp=now - 10),
        ts.TrendingItem(platform="B", product_name="Foo", timestamp=now - 20),
        ts.TrendingItem(platform="C", product_name="Foo", timestamp=now - 30),
    ]
    result = ts.TrendingScraper.detect_microtrends(items, within=60)
    assert result[0].intensity is not None
    expected = 3 / max(1.0, 60 / 3600)
    assert result[0].intensity >= expected


class FailingPath:
    def __init__(self, read=False, write=False):
        self._read = read
        self._write = write

    def exists(self):
        return True

    def read_text(self):
        if self._read:
            raise IOError("boom")
        return "[]"

    def write_text(self, *a, **k):
        if self._write:
            raise IOError("boom")

    def __str__(self):
        return "failing_path"


def _patch_all_empty(scraper, monkeypatch):
    for name in [
        "scrape_reddit",
        "scrape_shopify",
        "scrape_gumroad",
        "scrape_fiverr",
        "scrape_google_trends",
        "scrape_hackernews",
        "scrape_trendforecast",
        "scrape_social_firehose",
        "scrape_producthunt",
        "scrape_github_trending",
    ]:
        monkeypatch.setattr(scraper, name, lambda energy=None: [])


def test_collect_all_logs_cache_read_failure(monkeypatch, caplog):
    scraper = ts.TrendingScraper()
    _patch_all_empty(scraper, monkeypatch)
    monkeypatch.setattr(ts, "CACHE_FILE", FailingPath(read=True), raising=False)
    caplog.set_level("WARNING")
    assert scraper.collect_all() == []
    assert "failed reading cache" in caplog.text


def test_collect_all_logs_cache_write_failure(monkeypatch, caplog):
    scraper = ts.TrendingScraper()
    _patch_all_empty(scraper, monkeypatch)
    monkeypatch.setattr(
        scraper,
        "scrape_reddit",
        lambda energy=None: [ts.TrendingItem(platform="x", product_name="Foo")],
    )
    monkeypatch.setattr(
        scraper,
        "scrape_shopify",
        lambda energy=None: [ts.TrendingItem(platform="y", product_name="Foo")],
    )
    monkeypatch.setattr(ts, "CACHE_FILE", FailingPath(write=True), raising=False)
    caplog.set_level("WARNING")
    items = scraper.collect_all()
    assert items
    assert "failed writing cache" in caplog.text


def test_collect_all_logs_scrape_failure(monkeypatch, caplog):
    scraper = ts.TrendingScraper()
    _patch_all_empty(scraper, monkeypatch)

    def boom(energy=None):
        raise RuntimeError("fail")
    boom.__name__ = "scrape_reddit"

    monkeypatch.setattr(scraper, "scrape_reddit", boom)
    monkeypatch.setattr(ts, "CACHE_FILE", FailingPath(read=True), raising=False)
    caplog.set_level("WARNING")
    assert scraper.collect_all() == []
    assert "scrape scrape_reddit failed" in caplog.text


def test_disabled_scrapers_on_missing_deps(monkeypatch, caplog):
    monkeypatch.setattr(ts, "requests", None, raising=False)
    monkeypatch.setattr(ts, "BeautifulSoup", object(), raising=False)
    monkeypatch.setattr(ts, "TrendReq", object(), raising=False)
    caplog.set_level("WARNING")
    scraper = ts.TrendingScraper()
    assert scraper.disabled_scrapers == set()
    assert "requests" in caplog.text

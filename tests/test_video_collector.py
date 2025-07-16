import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import menace.clipped.video_collector as vc


def test_score_topic_videos_weight(monkeypatch):
    now = datetime.now(timezone.utc)

    results = [SimpleNamespace(watch_url="url_a"), SimpleNamespace(watch_url="url_b")]

    class FakeSearch:
        def __init__(self, query):
            self.results = results

    class FakeYouTube:
        def __init__(self, url):
            self.watch_url = url
            self.video_id = url[-1]
            if url.endswith("a"):
                self.views = 100
                self.publish_date = now
            else:
                self.views = 50
                self.publish_date = now - timedelta(days=10)

    monkeypatch.setattr(vc, "Search", FakeSearch)
    monkeypatch.setattr(vc, "YouTube", FakeYouTube)

    videos = vc.score_topic_videos("test", max_results=2)
    ids = [v.video_id for v in videos]
    assert ids == ["a", "b"]

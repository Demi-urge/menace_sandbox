import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.trend_monitor import TrendMonitor, MediumFetcher
from neurosales.external_integrations import RedditHarvester, TwitterTracker, GPT4Client, PineconeLogger
from unittest.mock import MagicMock


def test_trend_monitor_basic():
    reddit = MagicMock(spec=RedditHarvester)
    twitter = MagicMock(spec=TwitterTracker)
    medium = MagicMock(spec=MediumFetcher)
    gpt = MagicMock(spec=GPT4Client)
    pine = MagicMock(spec=PineconeLogger)

    reddit.harvest.return_value = [
        {
            "body": "Impulse buying everywhere",
            "link_id": "l1",
            "score": 5,
            "subreddit": "neuro",
            "created_utc": 1,
        }
    ]
    reddit.comment_tree.return_value = [{"body": "c1"}, {"body": "c2"}]

    twitter.search_hashtag.return_value = {"data": [{"text": "Impulse buying tips"}]}
    medium.fetch_posts.return_value = [{"title": "avoid impulse buying"}]
    gpt.stream_chat.return_value = ["summary"]

    monitor = TrendMonitor(reddit, twitter, medium, gpt, pine)
    trends = monitor.run(["neuro"], ["creator"], ["impulse buying"])
    assert trends and trends[0]["score"] > 0
    gpt.stream_chat.assert_called()
    pine.log.assert_called()

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List
import logging

import requests
from .external_integrations import (
    RedditHarvester,
    TwitterTracker,
    GPT4Client,
    PineconeLogger,
)
from context_builder_util import create_context_builder

logger = logging.getLogger(__name__)


class MediumFetcher:
    """Fetch Medium posts via RSS feed."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def fetch_posts(self, user: str, keywords: Iterable[str], limit: int = 5) -> List[Dict[str, str]]:
        url = f"https://medium.com/feed/@{user}"
        resp = self.session.get(url, timeout=10)
        posts: List[Dict[str, str]] = []
        try:
            root = ET.fromstring(resp.content)
            items = root.findall("./channel/item")[:limit]
            for item in items:
                title = item.findtext("title", default="")
                desc = item.findtext("description", default="")
                link = item.findtext("link", default="")
                if any(
                    k.lower() in title.lower() or k.lower() in desc.lower() for k in keywords
                ):
                    posts.append({"title": title, "link": link})
        except Exception:
            logger.exception("Medium fetch failed")
            return []
        return posts


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class TrendMonitor:
    """Monitor Reddit and cross-reference with Twitter/Medium."""

    def __init__(
        self,
        reddit: RedditHarvester,
        twitter: TwitterTracker,
        medium: MediumFetcher,
        gpt4: GPT4Client | None = None,
        vector: PineconeLogger | None = None,
        *,
        openai_key: str | None = None,
    ) -> None:
        self.reddit = reddit
        self.twitter = twitter
        self.medium = medium
        if gpt4 is None:
            self.gpt4 = GPT4Client(openai_key, context_builder=create_context_builder())
        else:
            self.gpt4 = gpt4
        if vector is None:
            raise ValueError("vector logger required")
        self.vector = vector

    # ------------------------------------------------------------------
    def collect_trends(
        self,
        subreddits: List[str],
        creators: List[str],
        keywords: List[str],
    ) -> List[Dict[str, Any]]:
        comments = self.reddit.harvest(subreddits, keywords)
        records: List[Dict[str, Any]] = []
        for c in comments:
            link = c.get("link_id", "")
            tree = self.reddit.comment_tree(link)
            records.append(
                {
                    "time": c.get("created_utc"),
                    "subreddit": c.get("subreddit"),
                    "upvotes": c.get("score", 0),
                    "comments": len(tree),
                    "text": c.get("body", ""),
                }
            )

        twitter_texts: List[str] = []
        for kw in keywords:
            data = self.twitter.search_hashtag(kw).get("data", [])
            twitter_texts.extend(d.get("text", "") for d in data)

        medium_texts: List[str] = []
        for creator in creators:
            posts = self.medium.fetch_posts(creator, keywords)
            medium_texts.extend(p.get("title", "") for p in posts)

        trending: List[Dict[str, Any]] = []
        for rec in records:
            rec_tokens = _tokens(rec["text"])
            cross = 0
            for t in twitter_texts + medium_texts:
                if rec_tokens & _tokens(t):
                    cross += 1
            score = rec["upvotes"] + rec["comments"] + 10 * cross
            if cross:
                rec["score"] = score
                trending.append(rec)
        trending.sort(key=lambda r: r["score"], reverse=True)
        return trending

    # ------------------------------------------------------------------
    def generate_content(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        text = trend["text"]
        summary = "".join(self.gpt4.stream_chat("monitor", [0.0], "summary", text))
        trend["summary"] = summary
        self.vector.log("trend", [0.0] * 1536, summary)
        return trend

    # ------------------------------------------------------------------
    def run(
        self,
        subreddits: List[str],
        creators: List[str],
        keywords: List[str],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        trends = self.collect_trends(subreddits, creators, keywords)
        outputs: List[Dict[str, Any]] = []
        for t in trends[:limit]:
            outputs.append(self.generate_content(t))
        return outputs


__all__ = ["MediumFetcher", "TrendMonitor"]

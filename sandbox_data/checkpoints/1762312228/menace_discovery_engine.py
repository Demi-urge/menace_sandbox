from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Iterable

from .trending_scraper import TrendingScraper
from .newsreader_bot import fetch_news, Event
from .passive_discovery_bot import PassiveDiscoveryBot
from .capital_management_bot import CapitalManagementBot
from . import idea_search_bot as isb
from .normalize_scraped_data import (
    load_items,
    normalize,
    save_candidates,
    NicheCandidate,
)
from .niche_saturation_bot import NicheSaturationBot, NicheCandidate as SatCandidate
from vector_service.context_builder import ContextBuilder
from .candidate_matcher import find_matching_models
from .database_management_bot import DatabaseManagementBot
from .research_aggregator_bot import InfoDB, ResearchItem

logger = logging.getLogger("MenaceDiscoveryEngine")


def _store_info(db: InfoDB, title: str, content: str, tags: List[str], category: str, type_: str, url: str | None = None) -> None:
    item = ResearchItem(
        topic=title,
        content=content,
        timestamp=time.time(),
        title=title,
        tags=tags,
        category=category,
        type_=type_,
        source_url=url or "",
    )
    db.add(item)


async def gather_trending(
    scraper: TrendingScraper, db: InfoDB, *, energy: float | None = None
) -> List[Dict]:
    platforms = {
        "reddit": scraper.scrape_reddit,
        "shopify": scraper.scrape_shopify,
        "gumroad": scraper.scrape_gumroad,
        "fiverr": scraper.scrape_fiverr,
        "google_trends": scraper.scrape_google_trends,
    }
    all_data: List[Dict] = []
    for name, func in platforms.items():
        items = await asyncio.to_thread(func, energy)
        data = [asdict(i) for i in items]
        Path(f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        all_data.extend(data)
        for it in items:
            _store_info(db, it.product_name or it.platform, it.source_url or "", it.tags, it.niche or "", "trending", it.source_url)
    return all_data


async def gather_passive(db: InfoDB, *, energy: float | None = None) -> List[Dict]:
    """Collect passive discovery results and store them.

    When provided, ``energy`` is forwarded to :meth:`PassiveDiscoveryBot.collect`
    to bias the returned items toward quick-return or scalable ideas.
    """
    bot = PassiveDiscoveryBot()
    items = await bot.collect(energy=energy)
    results: List[Dict] = []
    for it in items:
        results.append(
            {
                "platform": it.source,
                "niche": None,
                "product_name": it.text,
                "price_point": None,
                "tags": it.tags,
                "trend_signal": it.novelty,
                "source_url": it.url,
            }
        )
        _store_info(db, it.text, it.text, it.tags, "passive", "discovery", it.url)
    return results


async def gather_news(db: InfoDB, *, energy: float | None = None) -> List[Dict]:
    events: List[Event] = await asyncio.to_thread(
        fetch_news,
        "https://newsapi.org/v2/top-headlines",
        token="",
        energy=energy,
    )
    results: List[Dict] = []
    for ev in events:
        results.append(
            {
                "platform": ev.source,
                "niche": None,
                "product_name": ev.title,
                "price_point": None,
                "tags": ev.categories,
                "trend_signal": ev.impact,
                "source_url": None,
            }
        )
        _store_info(db, ev.title, ev.summary, ev.categories, "news", "news", None)
    return results


async def gather_search_ideas(
    db: InfoDB, db_bot: DatabaseManagementBot, *, energy: float
) -> List[str]:
    """Discover models via web search and ingest them.

    When Google credentials are unavailable or the ``requests`` dependency is
    missing we fall back to a very simple search over data collected by
    :class:`TrendingScraper`.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    engine = os.getenv("GOOGLE_ENGINE_ID", "")
    bank = isb.KeywordBank()
    results: List[isb.Result] = []
    if api_key and engine and isb.requests is not None:
        client = isb.GoogleSearchClient(api_key, engine)
        results = await asyncio.to_thread(
            isb.discover_new_models, client, bank, energy=energy
        )
    else:
        logger.info("Using built-in search from scraped data")
        scraper = TrendingScraper()
        trends = await asyncio.to_thread(scraper.collect_all, energy)
        queries = bank.generate_queries(energy)
        for t in trends:
            text = f"{t.product_name or ''} {' '.join(t.tags)}".lower()
            for q in queries:
                words = q.lower().split()
                if all(w in text for w in words):
                    results.append(
                        isb.Result(
                            title=t.product_name or t.platform,
                            link=t.source_url or "",
                            snippet=f"trending from {t.platform}",
                        )
                    )
                    break
    ingested: List[str] = []
    for res in results:
        try:
            db_bot.ingest_idea(res.title, tags=[], source="search", urls=[res.link])
            _store_info(db, res.title, res.snippet, [], "search", "search", res.link)
            ingested.append(res.title)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to ingest search result %s: %s", res.title, exc)
    return ingested


def convert_for_saturation(cands: List[NicheCandidate]) -> List[SatCandidate]:
    sat: List[SatCandidate] = []
    for c in cands:
        name = c.product_name or c.niche or c.platform
        demand = float(c.trend_signal or 0.0)
        competition = float(len(c.tags))
        sat.append(SatCandidate(name=name, demand=demand, competition=competition, trend=c.trend_signal or 0.0))
    return sat


def _similar(a: NicheCandidate, b: NicheCandidate) -> float:
    """Rudimentary similarity score for clustering."""
    name_a = a.product_name.lower()
    name_b = b.product_name.lower()
    name_score = SequenceMatcher(None, name_a, name_b).ratio()
    tag_score = len(set(a.tags) & set(b.tags)) / max(len(set(a.tags) | set(b.tags)), 1)
    return 0.7 * name_score + 0.3 * tag_score


def cluster_and_rank(cands: Iterable[NicheCandidate]) -> List[NicheCandidate]:
    """Cluster similar candidates and return best from each cluster."""
    clusters: List[List[NicheCandidate]] = []
    for cand in cands:
        placed = False
        for cluster in clusters:
            if _similar(cand, cluster[0]) >= 0.7:
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])

    ranked: List[NicheCandidate] = []
    for cluster in clusters:
        cluster.sort(key=lambda c: c.trend_signal or 0.0, reverse=True)
        ranked.append(cluster[0])
    ranked.sort(key=lambda c: c.trend_signal or 0.0, reverse=True)
    return ranked


async def run_cycle(*, context_builder: ContextBuilder) -> None:
    logging.basicConfig(level=logging.INFO)
    info_db = InfoDB()
    scraper = TrendingScraper()
    db_bot = DatabaseManagementBot()
    capital = CapitalManagementBot()
    energy = capital.energy_score(
        load=0.0, success_rate=1.0, deploy_eff=1.0, failure_rate=0.0
    )

    trending_task = asyncio.create_task(
        gather_trending(scraper, info_db, energy=energy)
    )
    passive_task = asyncio.create_task(gather_passive(info_db, energy=energy))
    news_task = asyncio.create_task(gather_news(info_db, energy=energy))
    search_task = asyncio.create_task(
        gather_search_ideas(info_db, db_bot, energy=energy)
    )

    raw_data: List[Dict] = []
    for result in await asyncio.gather(
        trending_task, passive_task, news_task, return_exceptions=True
    ):
        if isinstance(result, Exception):
            logger.error("gather task failed: %s", result)
        else:
            raw_data.extend(result)

    await search_task

    paths = [Path(p) for p in [
        "reddit.json",
        "shopify.json",
        "gumroad.json",
        "fiverr.json",
        "google_trends.json",
    ]]
    raw_data.extend(load_items(paths))

    candidates = normalize(raw_data)
    save_candidates(Path("niche_candidates.json"), candidates)
    for cand in candidates:
        _store_info(
            info_db,
            cand.product_name,
            cand.product_name,
            cand.tags,
            cand.niche or "",
            "candidate",
            None,
        )

    sat_candidates = convert_for_saturation(candidates)
    sat_bot = NicheSaturationBot(context_builder=context_builder)
    viable = sat_bot.detect(sat_candidates)

    mapping = { (cand.product_name or cand.niche or cand.platform): cand for cand in candidates }
    viable_candidates = [mapping.get(s.name) for s in viable if mapping.get(s.name)]
    ranked = cluster_and_rank(viable_candidates)

    ingested: List[str] = []
    for cand in ranked:
        try:
            if find_matching_models(cand):
                logger.info("Duplicate detected for %s", cand.product_name)
                continue
            status = db_bot.ingest_idea(
                cand.product_name,
                tags=cand.tags,
                source=cand.platform or (cand.niche or "unknown"),
                urls=[],
            )
            logger.info("Ingested %s with status %s", cand.product_name, status)
            ingested.append(cand.product_name)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", cand.product_name, exc)

    db_bot.adjust_threshold()
    logger.info("Discovery cycle complete. %d models ingested.", len(ingested))


if __name__ == "__main__":
    asyncio.run(
        run_cycle(
            context_builder=ContextBuilder(
                bots_db="bots.db",
                code_db="code.db",
                errors_db="errors.db",
                workflows_db="workflows.db",
            )
        )
    )

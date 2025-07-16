import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import asyncio
import json
from pathlib import Path

import menace.passive_discovery_bot as pdb


async def _run(bot: pdb.PassiveDiscoveryBot, *, energy: float | None = None) -> list[pdb.ContentItem]:
    return await bot.collect(energy=energy)


def test_collect_updates_keywords(tmp_path, monkeypatch):
    kw_file = tmp_path / "kw.json"
    bot = pdb.PassiveDiscoveryBot(keyword_db=kw_file)

    async def fake_fetch():
        return [pdb.ContentItem(text="Amazing AI tool", source="reddit", engagement=50)]

    for name in [
        "_fetch_reddit",
        "_fetch_twitter",
        "_fetch_substack",
        "_fetch_quora",
        "_fetch_indie_hackers",
        "_fetch_blogs",
    ]:
        monkeypatch.setattr(bot, name, fake_fetch)

    items = asyncio.run(_run(bot))
    assert items and items[0].novelty > 0
    assert kw_file.exists()
    saved = json.loads(kw_file.read_text())
    assert any("amazing" in w or "tool" in w for w in saved)


def test_novelty_penalty(monkeypatch, tmp_path):
    kw_file = tmp_path / "kw.json"
    kw_file.write_text(json.dumps(["existing", "idea"]))
    bot = pdb.PassiveDiscoveryBot(keyword_db=kw_file)

    async def fake_fetch():
        return [
            pdb.ContentItem(text="Existing idea", source="reddit", engagement=10)
        ]

    for name in [
        "_fetch_reddit",
        "_fetch_twitter",
        "_fetch_substack",
        "_fetch_quora",
        "_fetch_indie_hackers",
        "_fetch_blogs",
    ]:
        monkeypatch.setattr(bot, name, fake_fetch)

    monkeypatch.setattr(pdb.database_manager, "search_models", lambda x: [{"id": 1}])

    items = asyncio.run(_run(bot))
    assert items[0].novelty <= 0


def test_energy_filtering(monkeypatch):
    bot = pdb.PassiveDiscoveryBot(keyword_db=Path("kw.json"))

    async def fake_fetch():
        return [
            pdb.ContentItem(
                text="Fast ROI with low investment", source="reddit", engagement=1
            ),
            pdb.ContentItem(
                text="Massive scalable infrastructure", source="reddit", engagement=1
            ),
        ]

    for name in [
        "_fetch_reddit",
        "_fetch_twitter",
        "_fetch_substack",
        "_fetch_quora",
        "_fetch_indie_hackers",
        "_fetch_blogs",
    ]:
        monkeypatch.setattr(bot, name, fake_fetch)

    low = asyncio.run(_run(bot, energy=0.2))
    high = asyncio.run(_run(bot, energy=0.8))

    assert len(low) == 1 and "fast roi" in low[0].text.lower()
    assert len(high) == 1 and "scalable" in high[0].text.lower()

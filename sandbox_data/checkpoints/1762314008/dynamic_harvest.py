from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from pathlib import Path
from . import config
import random
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright  # type: ignore
    import playwright  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    async_playwright = None  # type: ignore
    playwright = None  # type: ignore


VIEWPORTS = [
    (1280, 720),
    (1366, 768),
    (1920, 1080),
]

TIMEZONES = [
    "UTC",
    "America/New_York",
    "Europe/London",
]


def playwright_browsers_installed() -> bool:
    """Return True if Playwright browsers have been installed."""
    if playwright is None:
        return False
    try:
        _ = playwright.__version__  # ensure attribute access for check
        cache_path = Path(
            os.environ.get(
                "PLAYWRIGHT_BROWSERS_PATH",
                Path.home() / ".cache" / "ms-playwright",
            )
        )
        return (cache_path / "browsers.json").exists()
    except Exception:
        return False


def ensure_playwright_browsers() -> bool:
    """Check Playwright installation and log a warning when missing."""
    if not playwright_browsers_installed():
        logger.warning(
            "Playwright browsers missing. Run `playwright install --with-deps`."
        )
        return False
    return True




class DynamicHarvester:
    """Harvest dynamic JavaScript dashboards and infinite scroll pages."""

    def __init__(self, proxies: Optional[List[str]] = None) -> None:
        if proxies is None:
            proxies = config.get_proxy_list()
        self.proxies = proxies or []

    # ------------------------------------------------------------------
    def _random_stealth(self) -> dict:
        width, height = random.choice(VIEWPORTS)
        timezone = random.choice(TIMEZONES)
        return {
            "viewport": {"width": width, "height": height},
            "timezone_id": timezone,
        }

    # ------------------------------------------------------------------
    def _next_proxy(self) -> Optional[dict]:
        if not self.proxies:
            return None
        return {"server": random.choice(self.proxies)}

    # ------------------------------------------------------------------
    def harvest_dashboard(self, url: str, username: str, password: str) -> List[str]:
        """Synchronously scrape SVG activation maps behind a login."""
        return asyncio.run(
            self.harvest_dashboard_async(url, username, password)
        )

    # ------------------------------------------------------------------
    async def harvest_dashboard_async(
        self, url: str, username: str, password: str
    ) -> List[str]:
        if async_playwright is None:
            logger.warning("DynamicHarvester disabled: playwright unavailable")
            return []
        if not ensure_playwright_browsers():
            return []
        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    **self._random_stealth(), proxy=self._next_proxy()
                )
                page = await context.new_page()
                await page.goto(url)
                await page.fill("input[name=username]", username)
                await page.fill("input[name=password]", password)
                await page.click("button[type=submit]")
                await page.wait_for_selector("svg")
                svgs = await page.query_selector_all("svg")
                data: List[str] = []
                for s in svgs:
                    html = await s.evaluate("el => el.outerHTML")
                    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
                    data.append(b64)
                return data
        except Exception:
            logger.exception("Failed dynamic harvest")
            return []
        finally:
            if browser:
                await browser.close()

    # ------------------------------------------------------------------
    def harvest_infinite_scroll(self, url: str, selector: str = "article") -> List[str]:
        """Synchronously scrape items from an infinite-scroll page."""
        return asyncio.run(self.harvest_infinite_scroll_async(url, selector))

    # ------------------------------------------------------------------
    async def harvest_infinite_scroll_async(
        self, url: str, selector: str = "article"
    ) -> List[str]:
        if async_playwright is None:
            logger.warning("DynamicHarvester disabled: playwright unavailable")
            return []
        if not ensure_playwright_browsers():
            return []
        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    **self._random_stealth(), proxy=self._next_proxy()
                )
                page = await context.new_page()
                await page.goto(url)
                seen: List[str] = []
                while True:
                    nodes = await page.query_selector_all(selector)
                    new_found = False
                    for n in nodes:
                        txt = await n.inner_text()
                        if txt not in seen:
                            seen.append(txt)
                            new_found = True
                    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_load_state("networkidle")
                    if not new_found:
                        break
                return seen
        except Exception:
            logger.exception("Failed dynamic harvest")
            return []
        finally:
            if browser:
                await browser.close()

from __future__ import annotations

"""Simple identity seeder using Playwright through residential proxies."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
import logging

from .error_flags import RAISE_ERRORS

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None  # type: ignore

from .proxy_manager import get_proxy
from .session_vault import SessionData, SessionVault
from .anticaptcha_stub import AntiCaptchaClient


async def _open_signup(url: str, proxy: Optional[str]) -> dict:
    if not async_playwright:
        raise RuntimeError("Playwright not available")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(proxy={"server": proxy} if proxy else None)
        page = await context.new_page()
        await page.goto(url)

        html = await page.content()
        if "captcha" in html.lower():
            solver = AntiCaptchaClient(os.getenv("ANTICAPTCHA_API_KEY"))
            img = await page.query_selector("img[src*='captcha'], img[alt*='captcha']")
            if img:
                tmp_path = Path("captcha.png")
                await img.screenshot(path=str(tmp_path))
                result = solver.solve(str(tmp_path))
                if result.text:
                    field = await page.query_selector(
                        "input[name*='captcha'], input[id*='captcha']"
                    )
                    if field:
                        await field.fill(result.text)
                        try:
                            await field.press("Enter")
                        except Exception as exc:
                            logger.exception("captcha submit failed: %s", exc)
                            if RAISE_ERRORS:
                                raise
        
        cookies = await context.cookies()
        ua = await context.evaluate("() => navigator.userAgent")
        fp = await context.evaluate("() => navigator.platform")
        await browser.close()
        return {"cookies": cookies, "user_agent": ua, "fingerprint": fp}


def seed_identity(url: str, vault: SessionVault) -> int:
    """Launch browser to register a new identity and store in vault."""
    proxy = get_proxy()
    data = asyncio.run(_open_signup(url, proxy))
    return vault.add(
        url.split("//")[-1].split("/")[0],
        SessionData(
            cookies={c["name"]: c["value"] for c in data["cookies"]},
            user_agent=data["user_agent"],
            fingerprint=data["fingerprint"],
            last_seen=0,
        ),
    )


__all__ = ["seed_identity"]

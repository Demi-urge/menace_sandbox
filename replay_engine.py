from __future__ import annotations

"""Simple replay engine for Playwright sessions."""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

try:  # optional dependency
    from playwright.async_api import async_playwright, BrowserContext
except Exception:  # pragma: no cover - optional
    async_playwright = None  # type: ignore
    BrowserContext = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ReplayData:
    har_path: Path
    form_data: dict
    html_path: Optional[Path] = None
    screenshot_path: Optional[Path] = None


class ReplayEngine:
    """Record and replay page interactions."""

    def __init__(self, out_dir: str = "replays") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    async def _extract_forms(self, page) -> dict:
        try:
            return await page.evaluate(
                "Array.from(document.forms).map(f => ({action:f.action, method:f.method, "
                "fields:Array.from(f.elements).map(e => ({name:e.name, value:e.value}))}))"
            )
        except Exception:
            return {}

    async def record(self, page) -> ReplayData:
        """Begin tracing and collect initial page snapshot."""
        if not async_playwright:
            raise RuntimeError("Playwright not available")

        ts = int(asyncio.get_event_loop().time() * 1000)
        base = self.out_dir / str(ts)
        har = base.with_suffix(".har")
        html_path = base.with_suffix(".html")
        screenshot_path = base.with_suffix(".png")

        await page.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        forms = await self._extract_forms(page)
        try:
            html_path.write_text(await page.content(), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - I/O failure
            logger.warning("Failed to write HTML snapshot: %s", exc)
        try:
            await page.screenshot(path=str(screenshot_path))
        except Exception as exc:  # pragma: no cover - I/O failure
            logger.warning("Failed to write screenshot: %s", exc)

        return ReplayData(
            har_path=har,
            form_data=forms,
            html_path=html_path,
            screenshot_path=screenshot_path,
        )

    async def pause(self, page, data: ReplayData) -> None:
        data.form_data = await self._extract_forms(page)
        try:
            if data.html_path:
                data.html_path.write_text(await page.content(), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - I/O failure
            logger.warning("Failed to update HTML snapshot: %s", exc)
        try:
            if data.screenshot_path:
                await page.screenshot(path=str(data.screenshot_path))
        except Exception as exc:  # pragma: no cover - I/O failure
            logger.warning("Failed to update screenshot: %s", exc)
        await page.context.tracing.stop(path=str(data.har_path))

    async def resume(self, browser_context: BrowserContext, data: ReplayData, token: str) -> None:
        """Replay the HAR while injecting the solved CAPTCHA token."""

        await browser_context.tracing.start(screenshots=True, snapshots=True)
        await browser_context.route_from_har(str(data.har_path), update=False)
        page = await browser_context.new_page()
        await page.add_init_script(f"window.__CAPTCHA_TOKEN='{token}'")

        try:
            import json

            with open(data.har_path, "r") as f:
                har = json.load(f)
            entries = har.get("log", {}).get("entries", [])
            start_url = entries[0]["request"]["url"] if entries else "about:blank"
            await page.goto(start_url)
            for form in data.form_data:
                selector = f"form[action='{form.get('action','')}']"
                if (await page.locator(selector).count()) > 0:
                    for field in form.get("fields", []):
                        name = field.get("name")
                        value = field.get("value")
                        if name:
                            try:
                                await page.fill(f"{selector} [name='{name}']", value)
                            except Exception:
                                continue
            if data.form_data:
                await page.evaluate("Array.from(document.forms).forEach(f=>f.submit())")
        finally:
            await browser_context.tracing.stop(path=str(data.har_path))


__all__ = ["ReplayEngine", "ReplayData"]

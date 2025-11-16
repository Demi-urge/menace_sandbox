"""Automated provisioning of proxies and social accounts."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Iterable

from .proxy_broker import fetch_free_proxies
from .username_generator import generate_username_for_topic

logger = logging.getLogger(__name__)


async def _get_proxies(
    count: int,
    *,
    proxy_type: str = "free",
    status: str = "active",
) -> list[dict[str, object]]:
    infos = await fetch_free_proxies(count)
    return [
        {
            "ip": p.ip,
            "port": p.port,
            "username": "",
            "password": "",
            "type": proxy_type,
            "usage_history": [],
            "failure_count": 0,
            "status": status,
        }
        for p in infos
    ]


def ensure_proxies(
    path: str | Path = "proxies.json",
    *,
    count: int = 5,
    proxy_type: str = "free",
    status: str = "active",
    preserve_existing: bool = False,
) -> None:
    """Populate *path* with free proxies if it is empty."""
    p = Path(path)
    existing: list[dict[str, object]] = []
    try:
        if p.exists():
            raw = json.loads(p.read_text() or "[]")
            if isinstance(raw, dict):
                existing = raw.get("proxies", [])
            else:
                existing = raw
            if existing and not preserve_existing:
                return
    except Exception:
        logger.exception("Failed to read proxies from %s", p)
    proxies = asyncio.run(_get_proxies(count, proxy_type=proxy_type, status=status))
    if preserve_existing and existing:
        seen = {(e.get("ip"), e.get("port")) for e in existing}
        for pr in proxies:
            if (pr["ip"], pr["port"]) not in seen:
                existing.append(pr)
        proxies = existing
    data = {"schema_version": "1.0", "proxies": proxies}
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception:
        logger.exception("Failed to write proxies to %s", p)


def _make_account(
    topic: str,
    *,
    platform: str = "YouTube",
    existing_ids: set[str] | None = None,
) -> dict[str, object]:
    existing_ids = existing_ids or set()
    try:
        username = generate_username_for_topic(topic)
    except Exception:
        logger.exception("Failed to generate username for %s", topic)
        username = ""
    if not username or username in existing_ids:
        username = f"{topic}_{uuid.uuid4().hex[:8]}"
    existing_ids.add(username)
    return {
        "id": username,
        "platform": platform,
        "topics": [topic],
        "total_revenue": 0,
        "monthly_revenue": {},
        "last_payment_date": None,
        "monetisation_status": "pending",
        "status": "active",
        "stats": [],
    }


def ensure_accounts(
    path: str | Path = "accounts.json",
    *,
    topics: Iterable[str] | None = None,
    platform: str = "YouTube",
    preserve_existing: bool = False,
) -> None:
    """Create sample accounts for *topics* if the file is empty."""
    p = Path(path)
    existing: dict[str, object] = {}
    try:
        if p.exists():
            raw = json.loads(p.read_text() or "{}")
            existing = raw if isinstance(raw, dict) else {}
            if existing.get("accounts") and not preserve_existing:
                return
    except Exception:
        logger.exception("Failed to read accounts from %s", p)
    topics = list(topics or ["general"])
    existing_ids = {a.get("id") for a in existing.get("accounts", [])}
    new_accounts = [_make_account(t, platform=platform, existing_ids=existing_ids) for t in topics]
    accounts = existing.get("accounts", []) if preserve_existing else []
    accounts.extend(new_accounts)
    data = {"schema_version": "1.0", "accounts": accounts}
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception:
        logger.exception("Failed to write accounts to %s", p)


__all__ = ["ensure_proxies", "ensure_accounts"]


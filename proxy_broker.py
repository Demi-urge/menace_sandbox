from __future__ import annotations

"""Unified proxy broker with health scoring and geo selection."""

import asyncio
import os
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import maxminddb
from celery import Celery
import psycopg2

__all__ = [
    "ProxyInfo",
    "ProxyPool",
    "BrightDataPool",
    "PacketStreamPool",
    "CustomISPPool",
    "HealthScorer",
    "GeoSelector",
    "SessionPinner",
    "fetch_free_proxies",
    "ProxyBroker",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProxyInfo:
    ip: str
    port: int
    username: str | None = None
    password: str | None = None
    asn: int | None = None
    city: str | None = None
    score: float = 1.0
    last_latency: float | None = None
    failure_count: int = 0
    captcha_count: int = 0

    def url(self) -> str:
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"http://{auth}{self.ip}:{self.port}"


# ---------------------------------------------------------------------------
# Proxy pools
# ---------------------------------------------------------------------------

class ProxyPool:
    """Abstract proxy pool."""

    def __init__(self, proxies: Optional[List[Dict[str, Any]]] = None) -> None:
        self.proxies: List[ProxyInfo] = [ProxyInfo(**p) for p in (proxies or [])]
        self._lock = asyncio.Lock()

    async def acquire(self) -> Optional[ProxyInfo]:
        async with self._lock:
            available = [p for p in self.proxies if p.score < 5]
            return random.choice(available) if available else None

    async def release(self, proxy: ProxyInfo) -> None:  # pragma: no cover - noop
        async with self._lock:
            if proxy in self.proxies:
                proxy.score = max(0.0, proxy.score - 0.5)
                if proxy.failure_count > 3:
                    self.proxies.remove(proxy)


class BrightDataPool(ProxyPool):
    """Proxy pool loading credentials from ``BRIGHTDATA_PROXIES`` env var."""

    def __init__(self, proxies: Optional[List[Dict[str, Any]]] = None) -> None:
        if proxies is None:
            raw = os.getenv("BRIGHTDATA_PROXIES", "[]")
            try:
                proxies = json.loads(raw)
            except Exception:
                proxies = []
        super().__init__(proxies)


class PacketStreamPool(ProxyPool):
    """Proxy pool loading proxies from ``PACKETSTREAM_PROXIES`` env var."""

    def __init__(self, proxies: Optional[List[Dict[str, Any]]] = None) -> None:
        if proxies is None:
            raw = os.getenv("PACKETSTREAM_PROXIES", "[]")
            try:
                proxies = json.loads(raw)
            except Exception:
                proxies = []
        super().__init__(proxies)


class CustomISPPool(ProxyPool):
    """Proxy pool loading proxies from ``CUSTOMISP_PROXIES`` env var."""

    def __init__(self, proxies: Optional[List[Dict[str, Any]]] = None) -> None:
        if proxies is None:
            raw = os.getenv("CUSTOMISP_PROXIES", "[]")
            try:
                proxies = json.loads(raw)
            except Exception:
                proxies = []
        super().__init__(proxies)


# ---------------------------------------------------------------------------
# Free proxy fetching
# ---------------------------------------------------------------------------

async def fetch_free_proxies(limit: int = 20) -> List[ProxyInfo]:
    """Retrieve and validate a small set of free proxies.

    The ``FREE_PROXY_SOURCE`` environment variable can override the default
    source URL. Each candidate proxy is checked by making a request to
    ``http://example.com``. Only proxies that respond successfully are
    returned with their latency recorded as the initial score.
    """

    url = os.getenv(
        "FREE_PROXY_SOURCE",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines()

    random.shuffle(lines)
    result: List[ProxyInfo] = []

    async def _check(entry: str) -> Optional[ProxyInfo]:
        if ":" not in entry:
            return None
        ip, port_str = entry.strip().split(":", 1)
        proxy_url = f"http://{ip}:{port_str}"
        start = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "http://example.com",
                    proxies={"http": proxy_url, "https": proxy_url},
                    timeout=5,
                )
                r.raise_for_status()
        except Exception:
            return None
        latency = time.monotonic() - start
        return ProxyInfo(ip=ip, port=int(port_str), score=latency, last_latency=latency)

    tasks = [_check(l) for l in lines[: limit * 5]]
    for coro in asyncio.as_completed(tasks):
        info = await coro
        if info:
            result.append(info)
            if len(result) >= limit:
                break

    return result


# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------

class HealthScorer:
    """Compute EWMA health score based on telemetry."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.history: Dict[str, ProxyInfo] = {}

    def update(self, proxy: ProxyInfo, *, latency: float, failed: bool = False, captcha: bool = False) -> None:
        info = self.history.setdefault(proxy.ip, proxy)
        info.last_latency = latency
        if failed:
            info.failure_count += 1
        if captcha:
            info.captcha_count += 1
        prev = info.score
        info.score = (1 - self.alpha) * info.score + self.alpha * latency
        if failed or captcha:
            info.score += 1.0
        proxy.score = info.score


# ---------------------------------------------------------------------------
# Geo selection
# ---------------------------------------------------------------------------

class GeoSelector:
    """Filter proxies by city or ASN using MaxMind GeoLite DB."""

    def __init__(self, db_path: str = "GeoLite2-City.mmdb") -> None:
        try:
            self.reader = maxminddb.open_database(db_path)
        except Exception:  # pragma: no cover - db optional
            self.reader = None

    def match(self, proxy: ProxyInfo, city: str | None, asn: int | None) -> bool:
        if not city and not asn:
            return True
        if not self.reader:
            return True
        data = self.reader.get(proxy.ip) or {}
        if city:
            if data.get("city", {}).get("names", {}).get("en") != city:
                return False
        if asn:
            if data.get("autonomous_system_number") != asn:
                return False
        return True


# ---------------------------------------------------------------------------
# Session pinning
# ---------------------------------------------------------------------------

class SessionPinner:
    """Maintain sticky sessions mapping logical actions to proxies."""

    def __init__(self) -> None:
        self._map: Dict[str, ProxyInfo] = {}

    def get(self, key: str) -> Optional[ProxyInfo]:
        return self._map.get(key)

    def pin(self, key: str, proxy: ProxyInfo) -> None:
        self._map[key] = proxy

    def release(self, key: str) -> None:
        self._map.pop(key, None)


# ---------------------------------------------------------------------------
# Telemetry store
# ---------------------------------------------------------------------------

class TelemetryStore:
    """Store proxy telemetry in PostgreSQL if available."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("PROXY_TELEMETRY_DSN", "dbname=proxy user=postgres")
        self.conn = None
        try:
            self.conn = psycopg2.connect(self.dsn)
            with self.conn, self.conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS proxy_telemetry (
                        ip TEXT PRIMARY KEY,
                        latency FLOAT,
                        fail_count INT,
                        captcha_count INT,
                        score FLOAT
                    )
                    """
                )
        except Exception:
            self.conn = None  # pragma: no cover - optional

    def update(self, info: ProxyInfo) -> None:
        if not self.conn:
            return
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO proxy_telemetry(ip, latency, fail_count, captcha_count, score)
                VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT(ip) DO UPDATE SET
                    latency=EXCLUDED.latency,
                    fail_count=EXCLUDED.fail_count,
                    captcha_count=EXCLUDED.captcha_count,
                    score=EXCLUDED.score
                """,
                (info.ip, info.last_latency, info.failure_count, info.captcha_count, info.score),
            )


# Celery app for async health checks
celery_app = Celery("proxy_health")

@celery_app.task
def record_health(ip: str, latency: float, failed: bool = False, captcha: bool = False) -> None:
    """Persist proxy telemetry asynchronously."""
    store = TelemetryStore()
    info = ProxyInfo(ip=ip, port=0)
    info.last_latency = latency
    if failed:
        info.failure_count = 1
    if captcha:
        info.captcha_count = 1
    info.score = latency + (1.0 if failed or captcha else 0.0)
    try:
        store.update(info)
    finally:
        if store.conn:
            store.conn.close()


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class ProxyBroker:
    """Combine pools with geo selection, health scoring and session pinning."""

    def __init__(
        self,
        pools: List[ProxyPool],
        scorer: Optional[HealthScorer] = None,
        selector: Optional[GeoSelector] = None,
        pinner: Optional[SessionPinner] = None,
        telemetry: Optional[TelemetryStore] = None,
    ) -> None:
        self.pools = pools
        self.scorer = scorer or HealthScorer()
        self.selector = selector or GeoSelector()
        self.pinner = pinner or SessionPinner()
        self.telemetry = telemetry or TelemetryStore()
        self._auto_fetched = False

    async def _ensure_pools(self) -> None:
        """Fetch free proxies if no configured pools are populated."""
        if any(pool.proxies for pool in self.pools):
            return
        env_vars = [
            os.getenv("BRIGHTDATA_PROXIES"),
            os.getenv("PACKETSTREAM_PROXIES"),
            os.getenv("CUSTOMISP_PROXIES"),
        ]
        if any(v for v in env_vars):
            return
        if self._auto_fetched:
            return
        proxies = await fetch_free_proxies()
        if proxies:
            pool = ProxyPool()
            pool.proxies = proxies
            self.pools.append(pool)
            self._auto_fetched = True

    async def acquire(
        self,
        session: str | None = None,
        city: str | None = None,
        asn: int | None = None,
    ) -> Optional[ProxyInfo]:
        await self._ensure_pools()
        if session:
            pinned = self.pinner.get(session)
            if pinned:
                return pinned
        for pool in self.pools:
            proxy = await pool.acquire()
            if proxy and self.selector.match(proxy, city, asn):
                if session:
                    self.pinner.pin(session, proxy)
                return proxy
        return None

    async def release(
        self,
        proxy: ProxyInfo,
        session: str | None = None,
        *,
        failed: bool = False,
        captcha: bool = False,
        latency: float | None = None,
    ) -> None:
        self.scorer.update(proxy, latency=latency or 0.0, failed=failed, captcha=captcha)
        if session:
            self.pinner.release(session)
        for pool in self.pools:
            if proxy in pool.proxies:
                await pool.release(proxy)
        self.telemetry.update(proxy)


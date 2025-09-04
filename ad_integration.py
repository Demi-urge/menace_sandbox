from __future__ import annotations

"""Ad network integration and revenue processing."""

from dataclasses import dataclass
from typing import List, Optional, Mapping, Dict
import os
import time
import asyncio
import importlib
import logging
import json
import hashlib
import urllib.request
from urllib.parse import urlencode

from dynamic_path_router import resolve_path
from .retry_utils import retry

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("requests library missing, using urllib fallback: %s", exc)
    import types

    class _LocalResponse:
        def __init__(self, data: bytes, status_code: int = 200) -> None:
            self._data = data
            self.status_code = status_code

        def json(self) -> object:
            return json.loads(self._data.decode())

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class _LocalSession:
        def get(self, url: str, timeout: int = 10) -> _LocalResponse:
            logger.warning("LocalSession GET %s", url)
            with urllib.request.urlopen(url, timeout=timeout) as fh:
                data = fh.read()
            return _LocalResponse(data, 200)

    requests = types.SimpleNamespace(Session=_LocalSession)  # type: ignore

try:  # optional async HTTP client
    httpx = importlib.import_module("httpx")
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("httpx library missing, using async urllib fallback: %s", exc)
    import types

    class _LocalAsyncResponse:
        def __init__(self, data: bytes, status_code: int = 200) -> None:
            self._data = data
            self.status_code = status_code

        def json(self) -> object:
            return json.loads(self._data.decode())

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class _LocalAsyncClient:
        async def __aenter__(self) -> "_LocalAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, timeout: int = 10) -> _LocalAsyncResponse:
            logger.warning("LocalAsyncClient GET %s", url)
            loop = asyncio.get_running_loop()

            def _fetch() -> bytes:
                with urllib.request.urlopen(url, timeout=timeout) as fh:
                    return fh.read()

            data = await loop.run_in_executor(None, _fetch)
            return _LocalAsyncResponse(data)

    httpx = types.SimpleNamespace(AsyncClient=_LocalAsyncClient)  # type: ignore

from .finance_router_bot import FinanceRouterBot  # noqa: E402
from .revenue_amplifier import SalesSpikeMonitor  # noqa: E402


@dataclass
class AdSale:
    """Single sale record returned by the ad API."""

    model_id: str
    amount: float
    platform: str
    segment: str


class AdIntegration:
    """Client fetching revenue data and routing payments."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session: Optional[requests.Session] = None,
        finance_bot: FinanceRouterBot | None = None,
        spike_monitor: SalesSpikeMonitor | None = None,
        seen_cache_path: str | None = None,
        cache_seconds: float | None = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        env_url = os.getenv("AD_API_URL")
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        elif env_url:
            self.base_url = env_url.rstrip("/")
        else:
            self.base_url = ""
            logger.info("AdIntegration disabled: no AD_API_URL configured")
        self.disabled = not bool(self.base_url)
        self.session = session or requests.Session()
        self.finance_bot = finance_bot or FinanceRouterBot()
        self.spike_monitor = spike_monitor or SalesSpikeMonitor()
        self.cache_seconds = (
            cache_seconds
            if cache_seconds is not None
            else float(os.getenv("AD_CACHE_SECONDS", 60.0))
        )
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._cache: Dict[str, tuple[List[AdSale], float]] = {}
        self._seen_sales: set[tuple[str, float, str, str, str]] = set()
        self.seen_cache_path = resolve_path(
            seen_cache_path
            or os.getenv("AD_SALES_CACHE_PATH", "sales_seen.json")
        )
        self._load_seen_sales()
        self._last_raw_response: object | None = None

    # ------------------------------------------------------------------
    def _context_key(self, params: Optional[Mapping[str, object]] = None) -> str:
        key = self.base_url
        if params:
            key += "?" + urlencode(sorted(params.items()))
        return hashlib.sha1(key.encode()).hexdigest()

    def _load_seen_sales(self) -> None:
        if not self.seen_cache_path.exists():
            return
        try:
            with open(self.seen_cache_path) as fh:
                data = json.load(fh)
            for rec in data:
                self._seen_sales.add(tuple(rec))
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to load seen sales cache: %s", exc)

    def _store_seen_sales(self) -> None:
        try:
            with open(self.seen_cache_path, "w") as fh:
                json.dump([list(r) for r in self._seen_sales], fh)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to store seen sales cache: %s", exc)

    def raw_sales(self) -> object | None:
        """Return the last raw JSON payload fetched from the API."""
        return self._last_raw_response

    # ------------------------------------------------------------------
    def fetch_revenue_data(
        self, params: Optional[Mapping[str, object]] = None
    ) -> List[AdSale]:
        """Retrieve recent sales from the ad network with caching."""
        if self.disabled:
            return []
        key = self._context_key(params)
        entry = self._cache.get(key)
        if entry and time.time() - entry[1] < self.cache_seconds:
            return list(entry[0])

        @retry(Exception, attempts=self.retry_attempts, delay=self.retry_delay)
        def _get() -> object:
            return self.session.get(
                f"{self.base_url}/sales", params=params or {}, timeout=10
            )

        try:
            resp = _get()
            if hasattr(resp, "raise_for_status"):
                resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("failed to fetch sales: %s", exc)
            return []

        self._last_raw_response = data
        sales: List[AdSale] = []
        added = False
        for item in data:
            try:
                sale = AdSale(
                    model_id=str(item.get("model_id", "")),
                    amount=float(item.get("amount", 0.0)),
                    platform=str(item.get("platform", "")),
                    segment=str(item.get("segment", "")),
                )
                if not sale.model_id or sale.amount <= 0.0:
                    continue
                skey = (
                    key,
                    sale.model_id,
                    sale.amount,
                    sale.platform,
                    sale.segment,
                )
                if skey in self._seen_sales:
                    continue
                self._seen_sales.add(skey)
                sales.append(sale)
                added = True
            except Exception as exc:
                logger.error("invalid sale entry: %s", exc)
                continue
        self._cache[key] = (list(sales), time.time())
        if added:
            self._store_seen_sales()
        return sales

    async def fetch_revenue_data_async(
        self, params: Optional[Mapping[str, object]] = None
    ) -> List[AdSale]:
        """Async version using httpx style client."""
        if self.disabled:
            return []
        key = self._context_key(params)
        entry = self._cache.get(key)
        if entry and time.time() - entry[1] < self.cache_seconds:
            return list(entry[0])

        attempt = 0
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.base_url}/sales", params=params or {}, timeout=10
                    )
                    if hasattr(resp, "raise_for_status"):
                        resp.raise_for_status()
                    data = resp.json()
                break
            except Exception as exc:
                attempt += 1
                if attempt >= self.retry_attempts:
                    logger.error("async fetch failed after retries: %s", exc)
                    return []
                await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))

        self._last_raw_response = data
        sales: List[AdSale] = []
        added = False
        for item in data:
            try:
                sale = AdSale(
                    model_id=str(item.get("model_id", "")),
                    amount=float(item.get("amount", 0.0)),
                    platform=str(item.get("platform", "")),
                    segment=str(item.get("segment", "")),
                )
                if not sale.model_id or sale.amount <= 0.0:
                    continue
                skey = (
                    key,
                    sale.model_id,
                    sale.amount,
                    sale.platform,
                    sale.segment,
                )
                if skey in self._seen_sales:
                    continue
                self._seen_sales.add(skey)
                sales.append(sale)
                added = True
            except Exception as exc:
                logger.error("invalid sale entry: %s", exc)
                continue
        self._cache[key] = (list(sales), time.time())
        if added:
            self._store_seen_sales()
        return sales

    def process_sales(self) -> None:
        """Fetch sales and trigger charges via FinanceRouterBot."""
        if self.disabled:
            return
        for sale in self.fetch_revenue_data():
            try:
                self.finance_bot.route_payment(sale.amount, sale.model_id)
                self.spike_monitor.record_sale(
                    sale.model_id, sale.amount, sale.platform, sale.segment
                )
            except Exception as e:
                logger.error("Failed to process sale for %s: %s", sale.model_id, e)
                continue

    async def process_sales_async(self) -> None:
        """Async version processing payments concurrently."""
        if self.disabled:
            return
        sales = await self.fetch_revenue_data_async()
        tasks = [
            self.finance_bot.route_payment_async(s.amount, s.model_id)
            for s in sales
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sale, result in zip(sales, results):
            if isinstance(result, Exception):
                logger.error("Failed to route payment for %s: %s", sale.model_id, result)
                continue
            self.spike_monitor.record_sale(
                sale.model_id, sale.amount, sale.platform, sale.segment
            )


__all__ = ["AdSale", "AdIntegration"]

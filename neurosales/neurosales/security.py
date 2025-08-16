import logging
import os
import time
from . import config as cfg
from typing import Dict

from fastapi import Header, HTTPException, Request, Depends

from .metrics import metrics

logger = logging.getLogger(__name__)


def get_api_key(x_api_key: str | None = Header(None)) -> str:
    """Validate X-API-Key header against NEURO_API_KEY env var."""
    expected = os.getenv("NEURO_API_KEY")
    if not x_api_key or x_api_key != expected:
        logger.warning("Invalid API key")
        if getattr(metrics, "auth_failures", None) is not None:
            metrics.auth_failures.inc()
        raise HTTPException(status_code=401, detail="invalid API key")
    return x_api_key


class MemoryRateLimiter:
    """Simple token bucket rate limiter stored in memory."""

    def __init__(self, rate: int | None = None, per: float | None = None) -> None:
        env_rate = os.getenv("NEURO_RATE_LIMIT")
        env_per = os.getenv("NEURO_RATE_PERIOD")
        default_rate = int(env_rate) if env_rate is not None else 5
        default_per = float(env_per) if env_per is not None else 60.0

        self.rate = rate if rate is not None else default_rate
        self.per = per if per is not None else default_per
        self.tokens: Dict[str, float] = {}
        self.timestamps: Dict[str, float] = {}

    def _get_key(self, request: Request, api_key: str) -> str:
        return api_key or request.client.host

    async def __call__(
        self, request: Request, api_key: str = Depends(get_api_key)
    ) -> None:
        key = self._get_key(request, api_key)
        now = time.time()
        tokens = self.tokens.get(key, self.rate)
        last = self.timestamps.get(key, now)
        tokens = min(self.rate, tokens + (now - last) * (self.rate / self.per))
        if tokens < 1:
            logger.warning("Rate limit exceeded for %s", key)
            if getattr(metrics, "rate_limited", None) is not None:
                metrics.rate_limited.inc()
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        self.tokens[key] = tokens - 1
        self.timestamps[key] = now


class RedisRateLimiter(MemoryRateLimiter):
    """Rate limiter backed by Redis using INCR/EXPIRE."""

    def __init__(
        self,
        redis_url: str | None = None,
        rate: int | None = None,
        per: float | None = None,
    ) -> None:
        super().__init__(rate, per)
        cfg_env = cfg.load_config()
        self.redis_url = redis_url or cfg_env.redis_url or "redis://localhost:6379/0"
        self.redis = None
        try:  # pragma: no cover - optional dependency
            import redis  # type: ignore

            self.redis = redis.Redis.from_url(self.redis_url)
        except Exception:  # pragma: no cover - redis may not be available
            logger.exception("Redis connection failed")
            self.redis = None

    async def __call__(
        self, request: Request, api_key: str = Depends(get_api_key)
    ) -> None:
        if self.redis is None:
            return await super().__call__(request, api_key)

        key = self._get_key(request, api_key)
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, int(self.per))
        except Exception:  # pragma: no cover - redis failure
            logger.exception("Redis rate limiter failure")
            return await super().__call__(request, api_key)

        if count > self.rate:
            logger.warning("Rate limit exceeded for %s", key)
            if getattr(metrics, "rate_limited", None) is not None:
                metrics.rate_limited.inc()
            raise HTTPException(status_code=429, detail="rate limit exceeded")


RateLimiter = RedisRateLimiter if cfg.load_config().redis_url else MemoryRateLimiter


__all__ = [
    "get_api_key",
    "RateLimiter",
    "MemoryRateLimiter",
    "RedisRateLimiter",
]

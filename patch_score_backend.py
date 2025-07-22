from __future__ import annotations

"""Optional backend for patch score storage."""

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    requests = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    boto3 = None  # type: ignore


logger = logging.getLogger(__name__)


class PatchScoreBackend:
    """Abstract interface for patch score storage backends."""

    def store(self, record: Dict[str, object]) -> None:
        """Persist a patch score record."""
        raise NotImplementedError

    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        """Return recently stored patch scores."""
        raise NotImplementedError


@dataclass
class HTTPPatchScoreBackend(PatchScoreBackend):
    """Send patch scores to a remote HTTP API."""

    url: str

    def store(self, record: Dict[str, object]) -> None:
        if not requests:
            logger.debug("requests unavailable; skipping HTTP backend")
            return
        try:  # pragma: no cover - network issues
            requests.post(self.url, json=record, timeout=5)
        except Exception as exc:  # pragma: no cover
            logger.error("HTTP patch score store failed: %s", exc)

    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        if not requests:
            return []
        try:  # pragma: no cover - network issues
            resp = requests.get(self.url, params={"limit": limit}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [tuple(r) for r in data]
        except Exception as exc:  # pragma: no cover
            logger.error("HTTP patch score fetch failed: %s", exc)
            return []


@dataclass
class S3PatchScoreBackend(PatchScoreBackend):
    """Store patch scores as JSON objects on S3."""

    bucket: str
    prefix: str

    def store(self, record: Dict[str, object]) -> None:
        if not boto3:
            logger.debug("boto3 unavailable; skipping S3 backend")
            return
        key = f"{self.prefix.rstrip('/')}/{int(time.time()*1000)}.json"
        try:  # pragma: no cover - network issues
            boto3.client("s3").put_object(
                Bucket=self.bucket, Key=key, Body=json.dumps(record).encode()
            )
        except Exception as exc:  # pragma: no cover
            logger.error("S3 patch score store failed: %s", exc)

    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        if not boto3:
            return []
        try:  # pragma: no cover - network issues
            client = boto3.client("s3")
            resp = client.list_objects_v2(
                Bucket=self.bucket, Prefix=self.prefix.rstrip("/") + "/"
            )
            items = resp.get("Contents", [])
            items.sort(key=lambda x: x["LastModified"], reverse=True)
            results: List[Tuple] = []
            for obj in items[:limit]:
                body = client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                results.append(tuple(json.loads(body)))
            return results
        except Exception as exc:  # pragma: no cover
            logger.error("S3 patch score fetch failed: %s", exc)
            return []


def backend_from_url(url: str) -> PatchScoreBackend:
    """Instantiate a backend from ``url``."""
    parts = urlparse(url)
    if parts.scheme in ("http", "https"):
        return HTTPPatchScoreBackend(url)
    if parts.scheme == "s3":
        return S3PatchScoreBackend(parts.netloc, parts.path.lstrip("/"))
    raise ValueError(f"unsupported backend URL: {url}")


__all__ = [
    "PatchScoreBackend",
    "HTTPPatchScoreBackend",
    "S3PatchScoreBackend",
    "backend_from_url",
]

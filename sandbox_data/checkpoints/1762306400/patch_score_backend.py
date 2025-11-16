from __future__ import annotations

"""Optional backend for patch score storage."""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, TypeVar
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    requests = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    boto3 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .data_bot import MetricsDB  # type: ignore
except Exception:  # pragma: no cover - metrics logging is optional
    MetricsDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .human_alignment_flagger import HumanAlignmentFlagger  # type: ignore
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from human_alignment_flagger import HumanAlignmentFlagger  # type: ignore
    except Exception:  # pragma: no cover - alignment flagger may be unavailable
        HumanAlignmentFlagger = None  # type: ignore


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _retry(func: Callable[[], T], tries: int = 3, delay: float = 1.0, backoff: float = 2.0) -> T:
    """Call ``func`` with retries and exponential backoff."""
    wait = delay
    for attempt in range(tries):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - best effort logging
            if attempt == tries - 1:
                raise
            logger.warning("retry %s/%s after error: %s", attempt + 1, tries, exc)
            time.sleep(wait)
            wait *= backoff


def attach_retrieval_info(
    record: Dict[str, object], session_id: str, vectors: List[Tuple[str, str, float]]
) -> Dict[str, object]:
    """Return a copy of *record* with retrieval metadata attached."""

    rec = dict(record)
    rec["retrieval_session_id"] = session_id
    rec["vectors"] = list(vectors)
    return rec


def _log_outcome(record: Dict[str, object]) -> None:
    """Best-effort logging of patch outcomes for retrieval metrics."""

    if isinstance(record, dict) and HumanAlignmentFlagger is not None:
        diff = record.get("diff")
        if diff:
            try:
                report = HumanAlignmentFlagger().flag_patch(diff, {})
                issues = report.get("issues", [])
                max_sev = max((i.get("severity", 0) for i in issues), default=0)
                record["alignment_report"] = report
                record["alignment_severity"] = max_sev
                if max_sev >= 3:
                    record["result"] = "blocked"
            except Exception:
                pass

    if MetricsDB is None:
        return
    patch_id = record.get("patch_id") or record.get("description")
    result = record.get("result")
    if not patch_id or result is None:
        return
    vectors = record.get("vectors") or []
    norm_vectors: List[Tuple[str, str]] = []
    if isinstance(vectors, list):
        for item in vectors:
            if isinstance(item, dict):
                origin = item.get("origin_db") or item.get("origin")
                vid = item.get("vector_id") or item.get("id")
            else:
                try:
                    origin, vid, *_ = item  # type: ignore[misc]
                except Exception:  # pragma: no cover - best effort
                    continue
            if origin is not None and vid is not None:
                norm_vectors.append((str(origin), str(vid)))
    result_str = str(result).lower()
    success = result_str == "ok"
    reverted = result_str == "reverted"
    session_id = record.get("retrieval_session_id") or ""
    roi_tag = record.get("roi_tag")
    try:  # pragma: no cover - best effort
        try:
            MetricsDB().log_patch_outcome(
                str(patch_id),
                success,
                norm_vectors,
                session_id=str(session_id),
                reverted=reverted,
                roi_tag=roi_tag,
            )
        except TypeError:
            MetricsDB().log_patch_outcome(
                str(patch_id),
                success,
                norm_vectors,
                session_id=str(session_id),
                reverted=reverted,
            )
    except Exception:
        logger.exception("failed to log patch outcome")



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
    fallback_dir: str | None = None
    _session: "requests.Session | None" = None
    _fallback: FilePatchScoreBackend | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.fallback_dir is None:
            self.fallback_dir = os.getenv("PATCH_SCORE_FALLBACK_DIR")
        if self.fallback_dir:
            self._fallback = FilePatchScoreBackend(self.fallback_dir)

    # ------------------------------------------------------------------
    def _get_session(self) -> "requests.Session":
        assert requests is not None  # pragma: no cover - checked by caller
        if self._session is None:
            sess_cls = getattr(requests, "Session", None)
            if sess_cls is not None:
                self._session = sess_cls()
            else:  # fallback for simplified stubs
                self._session = requests  # type: ignore
        return self._session

    # ------------------------------------------------------------------
    def store(self, record: Dict[str, object]) -> None:
        _log_outcome(record)
        if not requests:
            logger.debug("requests unavailable; skipping HTTP backend")
            if self._fallback:
                self._fallback.store(record)
            return
        try:  # pragma: no cover - network issues
            resp = _retry(lambda: self._get_session().post(self.url, json=record, timeout=5))
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover
            logger.error("HTTP patch score store failed: %s", exc)
            if self._fallback:
                self._fallback.store(record)

    # ------------------------------------------------------------------
    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        if not requests:
            return self._fallback.fetch_recent(limit) if self._fallback else []
        try:  # pragma: no cover - network issues
            resp = _retry(lambda: self._get_session().get(self.url, params={"limit": limit}, timeout=5))
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = data.get("results", [])
            return [tuple(r) for r in data]
        except Exception as exc:  # pragma: no cover
            logger.error("HTTP patch score fetch failed: %s", exc)
            if self._fallback:
                return self._fallback.fetch_recent(limit)
            return []


@dataclass
class S3PatchScoreBackend(PatchScoreBackend):
    """Store patch scores as JSON objects on S3."""

    bucket: str
    prefix: str
    _client: object | None = None

    # ------------------------------------------------------------------
    def _get_client(self):
        if not boto3:  # pragma: no cover - optional dependency
            return None
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    # ------------------------------------------------------------------
    def store(self, record: Dict[str, object]) -> None:
        _log_outcome(record)
        client = self._get_client()
        if client is None:
            logger.debug("boto3 unavailable; skipping S3 backend")
            return
        key = f"{self.prefix.rstrip('/')}/{int(time.time()*1000)}.json"
        try:  # pragma: no cover - network issues
            _retry(
                lambda: client.put_object(
                    Bucket=self.bucket, Key=key, Body=json.dumps(record).encode()
                )
            )
        except Exception as exc:  # pragma: no cover
            logger.error("S3 patch score store failed: %s", exc)

    # ------------------------------------------------------------------
    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        client = self._get_client()
        if client is None:
            return []
        try:  # pragma: no cover - network issues
            resp = _retry(
                lambda: client.list_objects_v2(
                    Bucket=self.bucket, Prefix=self.prefix.rstrip("/") + "/"
                )
            )
            items = resp.get("Contents", [])
            items.sort(key=lambda x: x["LastModified"], reverse=True)
            results: List[Tuple] = []
            for obj in items[:limit]:
                body = _retry(
                    lambda: client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                )
                results.append(tuple(json.loads(body)))
            return results
        except Exception as exc:  # pragma: no cover
            logger.error("S3 patch score fetch failed: %s", exc)
            return []


@dataclass
class FilePatchScoreBackend(PatchScoreBackend):
    """Store patch scores as JSON files on the local filesystem."""

    directory: str

    # ------------------------------------------------------------------
    def _ensure_dir(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        return self.directory

    # ------------------------------------------------------------------
    def store(self, record: Dict[str, object]) -> None:
        _log_outcome(record)
        path = os.path.join(self._ensure_dir(), f"{int(time.time()*1000)}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
        except Exception as exc:  # pragma: no cover
            logger.error("File patch score store failed: %s", exc)

    # ------------------------------------------------------------------
    def fetch_recent(self, limit: int = 20) -> List[Tuple]:
        try:
            files = [
                os.path.join(self._ensure_dir(), f)
                for f in os.listdir(self.directory)
                if f.endswith(".json")
            ]
        except Exception as exc:  # pragma: no cover
            logger.error("File patch score list failed: %s", exc)
            return []
        files.sort(key=os.path.getmtime, reverse=True)
        results: List[Tuple] = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    results.append(tuple(json.load(fh)))
            except Exception as exc:  # pragma: no cover
                logger.error("File patch score fetch failed: %s", exc)
        return results


def backend_from_url(url: str) -> PatchScoreBackend:
    """Instantiate a backend from ``url``."""
    parts = urlparse(url)
    if parts.scheme in ("http", "https"):
        return HTTPPatchScoreBackend(url)
    if parts.scheme == "s3":
        return S3PatchScoreBackend(parts.netloc, parts.path.lstrip("/"))
    if parts.scheme == "file":
        path = parts.netloc + parts.path
        return FilePatchScoreBackend(path)
    if not parts.scheme:
        path = url or "."
        return FilePatchScoreBackend(path)
    return FilePatchScoreBackend(url)


__all__ = [
    "PatchScoreBackend",
    "HTTPPatchScoreBackend",
    "S3PatchScoreBackend",
    "FilePatchScoreBackend",
    "backend_from_url",
    "attach_retrieval_info",
]

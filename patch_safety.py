from __future__ import annotations

"""Assess patch safety via metadata filters and failure embeddings.

The original version of this module only tracked failure embeddings and
exposed :class:`PatchSafety` for similarity scoring.  The vector service also
implemented a separate ``check_patch_safety`` helper that rejected vectors
based on metadata such as license fingerprints or semantic alert counts.

This file now consolidates both concerns.  ``PatchSafety`` provides an
``evaluate`` method which first applies the lightweight metadata checks and,
when available, scores the candidate against known failure embeddings.  The
return value is a tuple ``(passed, score)`` where ``passed`` indicates whether
the patch is acceptable and ``score`` is the similarity against recorded
failures.
"""

from dataclasses import dataclass, field
import json
import math
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Mapping, Tuple

from dynamic_path_router import resolve_path

from db_router import DBRouter, GLOBAL_ROUTER

from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from error_vectorizer import ErrorVectorizer

try:  # pragma: no cover - optional failure embeddings
    from failure_vectorizer import FailureVectorizer  # type: ignore
except Exception:  # pragma: no cover - fallback when unavailable
    FailureVectorizer = None  # type: ignore

try:  # pragma: no cover - optional dependency for metrics
    from vector_service import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    try:  # type: ignore[attr-defined]
        import metrics_exporter as _me  # type: ignore
    except Exception:  # pragma: no cover - very defensive fallback
        class _Dummy:
            def Gauge(self, *_, **__):  # type: ignore
                class _Null:
                    def labels(self, *_, **__):  # pragma: no cover - noop
                        return self

                    def inc(self, *_: Any, **__: Any) -> None:  # pragma: no cover - noop
                        return None

                return _Null()

        _me = _Dummy()  # type: ignore


_VIOLATIONS = _me.Gauge(
    "patch_safety_violations_total",
    "Patches rejected due to safety violations",
    labelnames=["type"],
)

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())


def _cosine(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class PatchSafety:
    """Store failure embeddings and evaluate new patches."""

    threshold: float = 0.8
    max_alert_severity: float = 1.0
    max_alerts: int = 5
    license_denylist: set[str] = field(
        default_factory=lambda: set(_DEFAULT_LICENSE_DENYLIST)
    )
    vectorizer: ErrorVectorizer = field(default_factory=ErrorVectorizer)
    storage_path: str | None = None
    failure_db_path: str | None = "failures.db"
    failure_vectorizer: FailureVectorizer | None = None
    router: DBRouter = field(default_factory=lambda: GLOBAL_ROUTER)
    _failures: List[List[float]] = field(default_factory=list)
    _records: List[Dict[str, Any]] = field(default_factory=list)
    _origins: List[str] = field(default_factory=list)
    _failure_vectors: List[List[float]] = field(default_factory=list)
    _failures_by_origin: Dict[str, List[List[float]]] = field(default_factory=dict)
    _failure_vectors_by_origin: Dict[str, List[List[float]]] = field(default_factory=dict)
    refresh_interval: float = 3600.0
    _last_refresh: float = 0.0
    _jsonl_mtime: float = 0.0
    _db_mtime: float = 0.0
    stat_timeout: float = 0.25

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # pragma: no cover - simple IO
        root = resolve_path(".")

        def _absolute_path(base: Path, value: str | None) -> str | None:
            """Return an absolute path without resolving symlinks."""

            if not value:
                return None
            path = Path(value)
            if path.is_absolute():
                return str(path)
            return str((base / path).absolute())

        self.storage_path = _absolute_path(root, self.storage_path)
        self.failure_db_path = _absolute_path(root, self.failure_db_path)
        self.load_failures()

    # ------------------------------------------------------------------
    def record_failure(self, err: Dict[str, Any], origin: str = "") -> None:
        """Add a failure example represented by ``err`` and persist it.

        ``origin`` allows tracking failure embeddings on a per-origin basis so
        future evaluations of vectors from the same source can be penalised
        more heavily.
        """

        self.vectorizer.fit([err])
        vec = self.vectorizer.transform(err)
        self._failures.append(vec)
        self._records.append(err)
        self._origins.append(origin)
        if origin:
            self._failures_by_origin.setdefault(origin, []).append(vec)
        if self.failure_vectorizer is not None:
            try:
                self.failure_vectorizer.fit([err])
                fvec = self.failure_vectorizer.transform(err)
                self._failure_vectors.append(fvec)
                if origin:
                    self._failure_vectors_by_origin.setdefault(origin, []).append(fvec)
            except Exception:  # pragma: no cover - best effort
                pass
        # best-effort persistence
        try:  # pragma: no cover - simple IO
            self.save_failures()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def load_failures(self, path: str | None = None, *, force: bool = False) -> None:
        """Populate failure vectors from JSONL store and ``failures.db``."""
        now = time.time()
        if (
            not force
            and self.refresh_interval > 0
            and now - self._last_refresh < self.refresh_interval
        ):
            return
        self._last_refresh = now
        pth = path or self.storage_path
        if pth and not Path(pth).is_absolute():
            pth = str((resolve_path(".") / pth).absolute())
            if path is None:
                self.storage_path = pth
        reload_needed = force
        if pth:
            p = Path(pth)
            if p.exists():
                mtime = self._mtime_with_timeout(p)
                if mtime > self._jsonl_mtime:
                    self._jsonl_mtime = mtime
                    reload_needed = True
        db_path = self.failure_db_path
        if db_path and not Path(db_path).is_absolute():
            db_path = str((resolve_path(".") / db_path).absolute())
            self.failure_db_path = db_path
        if db_path:
            db_mtime = self._mtime_with_timeout(Path(db_path))
            if db_mtime > self._db_mtime:
                self._db_mtime = db_mtime
                reload_needed = True
        if not reload_needed:
            return
        self._failures.clear()
        self._records.clear()
        self._origins.clear()
        self._failure_vectors.clear()
        self._failures_by_origin.clear()
        self._failure_vectors_by_origin.clear()
        if pth:
            p = Path(pth)
            if p.exists():
                try:  # pragma: no cover - simple IO
                    with p.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            if isinstance(data, dict):
                                err = data.get("err", {})
                                vec = data.get("vector")
                                origin = data.get("origin", "")
                            else:
                                err = {}
                                vec = data
                                origin = ""
                            if err:
                                try:
                                    self.vectorizer.fit([err])
                                except Exception:
                                    pass
                                self._records.append(err)
                                self._origins.append(origin)
                                if vec is None:
                                    try:
                                        vec = self.vectorizer.transform(err)
                                    except Exception:
                                        vec = []
                                if origin and isinstance(vec, list):
                                    self._failures_by_origin.setdefault(origin, []).append(vec)
                                if self.failure_vectorizer is not None:
                                    try:
                                        self.failure_vectorizer.fit([err])
                                        fvec = self.failure_vectorizer.transform(err)
                                    except Exception:
                                        fvec = []
                                    self._failure_vectors.append(fvec)
                                    if origin:
                                        self._failure_vectors_by_origin.setdefault(origin, []).append(fvec)
                            if isinstance(vec, list):
                                self._failures.append(vec)
                except Exception:  # pragma: no cover - best effort
                    pass
        if not db_path or FailureVectorizer is None:
            return
        try:
            conn = self.router.get_connection("failures")
            cur = conn.execute(
                "SELECT cause, demographics, profitability, retention, cac, roi FROM failures"
            )
            rows = cur.fetchall()
        except Exception:  # pragma: no cover - missing DB or table
            return
        if not rows:
            return
        records = [
            {
                "cause": r[0],
                "demographics": r[1],
                "profitability": r[2],
                "retention": r[3],
                "cac": r[4],
                "roi": r[5],
            }
            for r in rows
        ]
        fv = self.failure_vectorizer or FailureVectorizer()
        try:
            fv.fit(records)
        except Exception:  # pragma: no cover - best effort
            pass
        self.failure_vectorizer = fv
        for rec in records:
            try:
                self._failure_vectors.append(fv.transform(rec))
            except Exception:  # pragma: no cover - best effort
                self._failure_vectors.append([])

    # ------------------------------------------------------------------
    def save_failures(self, path: str | None = None) -> None:
        """Append the most recent failure to ``path``."""
        if not self._records or not self._failures:
            return
        pth = path or self.storage_path
        if not pth:
            return
        p = Path(pth)
        try:  # pragma: no cover - simple IO
            with p.open("a", encoding="utf-8") as fh:
                json.dump(
                    {
                        "err": self._records[-1],
                        "vector": self._failures[-1],
                        "origin": self._origins[-1],
                    },
                    fh,
                )
                fh.write("\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def score(self, err: Dict[str, Any]) -> float:
        """Return the maximum similarity between ``err`` and known failures."""
        scores: List[float] = []
        if self._failures:
            try:
                vec = self.vectorizer.transform(err)
                scores.append(max(_cosine(vec, f) for f in self._failures))
            except Exception:  # pragma: no cover - defensive
                pass
        if self._failure_vectors and self.failure_vectorizer is not None:
            try:
                vec2 = self.failure_vectorizer.transform(err)
                scores.append(max(_cosine(vec2, f) for f in self._failure_vectors))
            except Exception:  # pragma: no cover - defensive
                pass
        return max(scores) if scores else 0.0

    # ------------------------------------------------------------------
    def _check_meta(self, meta: Mapping[str, Any]) -> bool:
        """Return ``True`` when metadata passes safety checks."""

        sev = meta.get("alignment_severity")
        if sev is not None:
            try:
                if float(sev) > self.max_alert_severity:
                    _VIOLATIONS.labels("severity").inc()
                    return False
            except Exception:  # pragma: no cover - defensive parsing
                pass

        alerts = meta.get("semantic_alerts")
        if alerts is not None:
            try:
                count = len(alerts) if isinstance(alerts, (list, tuple, set)) else 1
                if count > self.max_alerts:
                    _VIOLATIONS.labels("alerts").inc()
                    return False
            except Exception:  # pragma: no cover - defensive parsing
                pass

        lic = meta.get("license")
        fp = meta.get("license_fingerprint")
        denylist = self.license_denylist or _DEFAULT_LICENSE_DENYLIST
        if lic in denylist or _LICENSE_DENYLIST.get(fp) in denylist:
            _VIOLATIONS.labels("license").inc()
            return False

        return True

    # ------------------------------------------------------------------
    def pre_embed_check(self, meta: Mapping[str, Any]) -> bool:
        """Lightweight wrapper around :meth:`_check_meta`.

        Embedding pipelines may call this prior to generating embeddings so
        obviously unsafe records are skipped early.  The check mirrors the
        metadata validation performed by :meth:`evaluate` without incurring the
        overhead of similarity scoring.
        """

        return self._check_meta(meta)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        meta: Mapping[str, Any],
        err: Dict[str, Any] | None = None,
        *,
        origin: str = "",
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Return ``(passed, score, risks)`` for ``meta`` and optional ``err``.

        ``passed`` is ``False`` when the metadata violates any denylist or when
        the similarity ``score`` exceeds the ``threshold``.  ``risks`` maps each
        origin to the similarity score against failures originating from that
        source.  The aggregate ``score`` incorporates these per-origin values
        and the global similarity across all recorded failures.
        """

        if not self._check_meta(meta):
            return False, 0.0, {}

        risks: Dict[str, float] = {}
        score = 0.0
        if err is not None:
            try:
                vec = self.vectorizer.transform(err)
            except Exception:  # pragma: no cover - defensive
                vec = []
            try:
                vec2 = (
                    self.failure_vectorizer.transform(err)
                    if self.failure_vectorizer is not None
                    else []
                )
            except Exception:  # pragma: no cover - defensive
                vec2 = []

            # Base similarity across all failures regardless of origin
            if self._failures:
                try:
                    score = max(_cosine(vec, f) for f in self._failures)
                except Exception:  # pragma: no cover - defensive
                    score = 0.0
            if self._failure_vectors and vec2:
                try:
                    score = max(score, max(_cosine(vec2, f) for f in self._failure_vectors))
                except Exception:  # pragma: no cover - defensive
                    pass

            # Per-origin risk mapping
            for o, failures in self._failures_by_origin.items():
                try:
                    risks[o] = max(_cosine(vec, f) for f in failures)
                except Exception:  # pragma: no cover - defensive
                    risks[o] = 0.0
            if vec2:
                for o, failures in self._failure_vectors_by_origin.items():
                    try:
                        risks[o] = max(risks.get(o, 0.0), max(_cosine(vec2, f) for f in failures))
                    except Exception:  # pragma: no cover - defensive
                        risks[o] = max(risks.get(o, 0.0), 0.0)

            if origin:
                score += risks.get(origin, 0.0)
            elif risks:
                score = max(score, max(risks.values()))

        passed = score < self.threshold
        return passed, score, risks

    # ------------------------------------------------------------------
    def _mtime_with_timeout(self, path: Path) -> float:
        """Return ``path`` mtime without blocking bootstrapping flows.

        Some environments mount paths on slow or unavailable devices (for
        example Windows network shares).  ``Path.stat`` can hang in those
        situations, so we call it in a short-lived daemon thread and bail out
        once ``stat_timeout`` elapses.  When the call times out we treat the
        mtime as ``0.0`` which simply disables refreshes from that location.
        """

        result: List[float] = []

        def _probe() -> None:
            try:  # pragma: no cover - simple IO
                result.append(path.stat().st_mtime)
            except Exception:  # pragma: no cover - best effort
                pass

        t = Thread(target=_probe, daemon=True)
        t.start()
        t.join(self.stat_timeout)
        return result[0] if result else 0.0


def check_patch_safety(
    meta: Mapping[str, Any],
    *,
    max_alert_severity: float = 1.0,
    max_alerts: int = 5,
    license_denylist: set[str] | None = None,
) -> bool:
    """Backward compatible wrapper around :class:`PatchSafety`.

    This helper mirrors the behaviour of the previous implementation used in
    several parts of the code base.  It ignores similarity scoring and simply
    returns whether the metadata passes the safety checks.
    """

    ps = PatchSafety(
        max_alert_severity=max_alert_severity,
        max_alerts=max_alerts,
        license_denylist=set(license_denylist or _DEFAULT_LICENSE_DENYLIST),
    )
    passed, _ = ps.evaluate(meta)
    return passed


def pre_embed_check(
    meta: Mapping[str, Any],
    *,
    max_alert_severity: float = 1.0,
    max_alerts: int = 5,
    license_denylist: set[str] | None = None,
) -> bool:
    """Public helper used by embedding pipelines prior to vectorisation.

    This mirrors :func:`check_patch_safety` but avoids constructing failure
    similarities; only metadata checks are performed.
    """

    ps = PatchSafety(
        max_alert_severity=max_alert_severity,
        max_alerts=max_alerts,
        license_denylist=set(license_denylist or _DEFAULT_LICENSE_DENYLIST),
    )
    return ps.pre_embed_check(meta)


__all__ = ["PatchSafety", "check_patch_safety", "pre_embed_check", "_VIOLATIONS"]

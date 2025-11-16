from __future__ import annotations

"""Utilities for retrying after failures with fingerprint similarity checks."""

from typing import Callable, Iterable, Tuple, List
import logging

from .failure_fingerprint import FailureFingerprint, log_fingerprint, find_similar
from .failure_fingerprint_store import FailureFingerprintStore
from .vector_utils import cosine_similarity


Matcher = Callable[[Iterable[float], float], List[FailureFingerprint]]


def _all_matches(
    fingerprint: FailureFingerprint,
    store: FailureFingerprintStore | Matcher | None,
) -> List[FailureFingerprint]:
    """Return similar fingerprints using ``store`` or ``find_similar``."""

    if store is None:
        try:
            return find_similar(fingerprint.embedding, 0.0)
        except Exception:
            return []
    if callable(store):
        try:
            return store(fingerprint.embedding, 0.0)
        except Exception:
            return []
    try:
        return store.find_similar(fingerprint, threshold=0.0)
    except Exception:
        return []


def check_similarity_and_warn(
    fingerprint: FailureFingerprint,
    store: FailureFingerprintStore | Matcher | None,
    threshold: float,
    description: str,
) -> tuple[str, bool, float, List[FailureFingerprint], str]:
    """Check similarity of ``fingerprint`` and warn or skip.

    Returns updated description, skip flag, best similarity, list of matches and
    the warning message (if any).
    """

    matches_all = _all_matches(fingerprint, store)
    matches: List[FailureFingerprint] = []
    best = 0.0
    best_match: FailureFingerprint | None = None
    for m in matches_all:
        try:
            sim = cosine_similarity(fingerprint.embedding, m.embedding)
        except Exception:
            sim = 0.0
        if sim > best:
            best = sim
            best_match = m
        if sim >= threshold:
            matches.append(m)

    warning = ""
    if best_match is not None:
        warning = f"avoid repeating failure: {best_match.error_message}"

    if best >= threshold:
        return description, True, best, matches, warning

    if warning:
        description = f"{description}; {warning}"

    return description, False, best, matches, warning


def record_failure(
    fingerprint: FailureFingerprint,
    store: FailureFingerprintStore | Callable[[FailureFingerprint], None] | None,
) -> None:
    """Persist ``fingerprint`` using ``store`` or log file."""

    try:
        if store is None:
            log_fingerprint(fingerprint)
        elif callable(store):
            store(fingerprint)
        else:
            store.add(fingerprint)
    except Exception:  # pragma: no cover - best effort
        logging.getLogger(__name__).exception(
            "failed to log failure fingerprint"
        )

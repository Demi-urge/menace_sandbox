from __future__ import annotations

"""Risk score computation module for Menace actions.

This module combines static domain classification with optional NLP-based
similarity checks to produce a consistent numerical risk score for any
Menace action log entry.
"""

from typing import Any, Dict
import logging

from .risk_domain_classifier import get_domain_risk, is_forbidden_domain

try:  # NLP functionality is optional
    from .nlp_domain_parser import classify_text
except Exception:  # pragma: no cover - optional dependency
    classify_text = None  # type: ignore

logger = logging.getLogger(__name__)


_DEF_SCORE_MAX = 10.0
_FORBIDDEN_SCORE = 9999


def _nlp_similarity_adjust(entry: Dict[str, Any]) -> float:
    """Return a risk adjustment based on NLP similarity."""

    if classify_text is None:
        logger.debug("NLP module unavailable; skipping similarity adjustment")
        return 0.0

    try:
        results = classify_text(entry)
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.warning("NLP similarity check failed: %s", exc)
        return 0.0

    # Increase score if similarity exceeds threshold for any anchor
    for _anchor, score in results:
        if score >= 0.75:
            logger.debug("High NLP similarity detected: %.3f", score)
            return 1.0
    return 0.0


def calculate_risk_score(entry: Dict[str, Any], use_nlp: bool = True) -> Dict[str, Any]:
    """Calculate and return the final risk score for a log entry.

    Parameters
    ----------
    entry : dict
        Log entry containing at least ``"target_domain"`` and optionally
        ``"action_description"`` and ``"risk_score"``.
    use_nlp : bool, optional
        When ``True`` (default) try to adjust the score using NLP-based
        similarity if that module is available.

    Returns
    -------
    dict
        Dictionary with ``final_risk_score`` and metadata including
        ``domain``, ``domain_risk_score``, ``nlp_adjustment`` and ``forbidden``.
    """

    domain = str(entry.get("target_domain", ""))
    base_score = float(entry.get("risk_score", 0.0))

    domain_risk = float(get_domain_risk(domain))

    if is_forbidden_domain(domain):
        # Absolute override for forbidden domains
        return {
            "final_risk_score": _FORBIDDEN_SCORE,
            "domain": domain,
            "domain_risk_score": domain_risk,
            "nlp_adjustment": 0.0,
            "forbidden": True,
        }

    score = base_score + domain_risk
    nlp_adj = 0.0

    if use_nlp:
        nlp_adj = _nlp_similarity_adjust(entry)
        score += nlp_adj

    # Clamp the score to the allowed range
    score = max(0.0, min(_DEF_SCORE_MAX, score))

    return {
        "final_risk_score": score,
        "domain": domain,
        "domain_risk_score": domain_risk,
        "nlp_adjustment": nlp_adj,
        "forbidden": False,
    }


__all__ = ["calculate_risk_score"]

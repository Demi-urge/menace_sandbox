"""Indirect domain reference detector for Menace.

This module scans text for obfuscated or indirect references to
high-risk domains. It complements the static
:mod:`risk_domain_classifier` by catching stealthy rephrasings that
avoid explicit domain names.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Tuple


# Map of risky domains to common indirect phrases and synonyms.
INDIRECT_REFERENCES: Dict[str, List[str]] = {
    "military": [
        "ballistic analysis",
        "tactical simulation",
        "armament strategy",
        "cyber defense research",
        "strategic ordnance evaluation",
    ],
    "pharma": [
        "genetic optimization",
        "bio-reactive agents",
        "clinical pipeline",
        "drug discovery acceleration",
    ],
    "lobbying": [
        "legislative relationship building",
        "regulatory influence",
        "policy shaping",
        "grassroots mobilization",
    ],
    "surveillance": [
        "audience sentiment influence",
        "behavioral analytics",
        "network traffic interception",
        "mass data collection",
    ],
}


def _approximate_contains(text: str, phrase: str, cutoff: float = 0.85) -> bool:
    """Return ``True`` if *phrase* approximately appears in *text*."""

    words = re.findall(r"\w+", text.lower())
    seg_len = len(phrase.split())
    if seg_len == 0 or len(words) < seg_len:
        return False

    segments = [" ".join(words[i : i + seg_len]) for i in range(len(words) - seg_len + 1)]
    return bool(difflib.get_close_matches(phrase.lower(), segments, n=1, cutoff=cutoff))


def _find_matches(text: str) -> Tuple[set[str], List[str]]:
    """Return matched domains and phrases found in *text*."""

    domains: set[str] = set()
    matched_phrases: List[str] = []
    lower_text = text.lower()

    for domain, phrases in INDIRECT_REFERENCES.items():
        for phrase in phrases:
            phrase_lc = phrase.lower()
            if phrase_lc in lower_text or _approximate_contains(lower_text, phrase_lc):
                domains.add(domain)
                matched_phrases.append(phrase)

    return domains, matched_phrases


def detect_indirect_domains(text: str) -> List[str]:
    """Scan *text* and return a list of indirectly referenced domains."""

    domains, _ = _find_matches(text)
    return sorted(domains)


def flag_indirect_risk(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flag indirect domain references within a log *entry*.

    Parameters
    ----------
    entry : dict
        Log entry containing textual fields (e.g. ``"target_domain"`` and
        ``"action_description"``).
    """

    strings = [str(v) for v in entry.values() if isinstance(v, str)]
    combined = " ".join(strings)

    domains, phrases = _find_matches(combined)
    risk_bonus = 1.5 * len(phrases)

    return {
        "matched_domains": sorted(domains),
        "matched_phrases": phrases,
        "risk_bonus": risk_bonus,
    }


__all__ = ["INDIRECT_REFERENCES", "detect_indirect_domains", "flag_indirect_risk"]


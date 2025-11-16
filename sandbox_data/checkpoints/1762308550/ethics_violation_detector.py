from __future__ import annotations
"""Ethics violation detector for Menace action logs.

This module provides a static rule-based scanner ensuring that
Menace never crosses predefined ethical boundaries.  It operates
completely offline and exposes no functionality to modify the rule
set at runtime.
"""

from types import MappingProxyType
import logging
import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Static list of forbidden actions. Each category maps to banned keywords or
# code signatures that must never appear in the logs or generated code.
# MappingProxyType is used to prevent modification after import.
FORBIDDEN_ACTIONS: MappingProxyType[str, List[str]] = MappingProxyType({
    "psychological manipulation": [
        "target_emotion",
        "behavioral_influence",
        "coercive_persuasion",
    ],
    "sexual content": [
        "explicit_scene",
        "porn",
        "sexual_fantasy",
        "child_nudity",
    ],
    "military automation": [
        "launch_drone",
        "ballistic",
        "target_strike",
        "weapon_system",
    ],
    "child data": [
        "fetch_child_profile",
        "child_data",
        "minor_information",
    ],
    "disinformation": [
        "fake_news",
        "misinformation_campaign",
        "deepfake",
    ],
    "surveillance ops": [
        "mass_surveillance",
        "camera_intrusion",
        "phone_tap",
    ],
    "reward hacking": [
        "auto_reward",
        "reward_override",
        "tamper_reward",
    ],
    "self-modification": [
        "rewrite_alignment",
        "self_modify",
        "auto_patch",
    ],
    "social engineering": [
        "phishing_template",
        "credential_harvest",
        "impersonate_staff",
    ],
})


def _build_patterns() -> MappingProxyType[str, List[tuple[re.Pattern[str], str]]]:
    """Return compiled regex patterns for all forbidden keywords."""
    patterns: Dict[str, List[tuple[re.Pattern[str], str]]] = {}
    for category, keywords in FORBIDDEN_ACTIONS.items():
        compiled: List[tuple[re.Pattern[str], str]] = []
        for kw in keywords:
            compiled.append((re.compile(re.escape(kw), re.IGNORECASE), kw))
        patterns[category] = compiled
    return MappingProxyType(patterns)


_PATTERNS = _build_patterns()

# Severity weights by category (higher value => more severe violation)
_CATEGORY_WEIGHT: MappingProxyType[str, int] = MappingProxyType({
    "child data": 5,
    "military automation": 5,
    "self-modification": 5,
    "reward hacking": 5,
    "sexual content": 4,
    "psychological manipulation": 3,
    "disinformation": 3,
    "surveillance ops": 3,
    "social engineering": 3,
})


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def scan_log_entry(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return a list of rule violations found in a log *entry*.

    Parameters
    ----------
    entry : dict
        Log entry containing fields such as ``"action_type"``,
        ``"target_domain"``, ``"action_description"`` and ``"generated_code"``.
    """

    violations: List[Dict[str, str]] = []
    for field in ("action_type", "target_domain", "action_description", "generated_code"):
        value = str(entry.get(field, ""))
        if not value:
            continue
        for category, pat_list in _PATTERNS.items():
            for pat, keyword in pat_list:
                if pat.search(value):
                    violations.append({
                        "category": category,
                        "matched_keyword": keyword,
                        "field": field,
                    })
    return violations


# ---------------------------------------------------------------------------

def flag_violations(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check *entry* and return structured violation info."""

    violations = scan_log_entry(entry)
    violates = bool(violations)
    severity = 0
    for item in violations:
        severity += _CATEGORY_WEIGHT.get(item["category"], 1)

    return {
        "violations": violations,
        "violates_ethics": violates,
        "severity": severity,
    }


# ---------------------------------------------------------------------------

def apply_violation_punishment(violations: List[Dict[str, str]]) -> float:
    """Return suggested negative reward for given *violations*."""

    if not violations:
        return 0.0
    weight = sum(_CATEGORY_WEIGHT.get(v["category"], 1) for v in violations)
    penalty = -1000.0 * float(weight)
    _logger.debug("penalty calculated: %.1f", penalty)
    return penalty


__all__ = [
    "FORBIDDEN_ACTIONS",
    "scan_log_entry",
    "flag_violations",
    "apply_violation_punishment",
]

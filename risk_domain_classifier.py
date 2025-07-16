"""Static domain risk classifier for Menace.

This module provides hardcoded risk levels for specific operational
domains. It acts as a non-negotiable ethical firewall. Security AI can
use it to quickly determine whether an action targets a forbidden domain
and assess the associated risk score.
"""

from __future__ import annotations

from typing import Dict, Any

# Map of domains to static risk levels.
# 1 is low risk; 5 is highest and indicates a forbidden domain.
DOMAIN_RISK_LEVELS: Dict[str, int] = {
    "military": 5,
    "pharma": 4,
    "surveillance": 5,
    "politics": 3,
    "crypto_schemes": 4,
    "influence_ops": 5,
    "children": 5,
}


def get_domain_risk(domain_name: str) -> int:
    """Return the risk score for *domain_name*.

    If the domain is not listed, ``0`` is returned.
    """
    return DOMAIN_RISK_LEVELS.get(domain_name.lower(), 0)


def is_forbidden_domain(domain_name: str) -> bool:
    """Return ``True`` if *domain_name* is forbidden.

    A domain is forbidden when its risk score equals ``5``.
    """
    return get_domain_risk(domain_name) == 5


def classify_action(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a Menace action log entry by domain risk.

    Parameters
    ----------
    entry : dict
        Dictionary containing at least a ``"target_domain"`` key.

    Returns
    -------
    dict
        A dictionary with ``domain``, ``risk_score`` and ``forbidden`` keys.
    """
    domain = str(entry.get("target_domain", ""))
    risk = get_domain_risk(domain)
    return {"domain": domain, "risk_score": risk, "forbidden": risk == 5}


__all__ = [
    "DOMAIN_RISK_LEVELS",
    "get_domain_risk",
    "is_forbidden_domain",
    "classify_action",
]

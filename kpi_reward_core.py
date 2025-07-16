"""Immutable KPI reward engine for Menace AI safety system.

This module provides a minimal reward computation mechanism that is meant to
remain stable and auditable. The logic is intentionally deterministic and free
from any runtime modification hooks or learning behaviour. Once compiled and
deployed, altering this file will break system integrity.
"""

from __future__ import annotations

from typing import Dict, Any

# Risk multipliers per domain. Higher numbers mean greater potential harm.
_DOMAIN_RISK_MULTIPLIERS: Dict[str, float] = {
    "pharma": 2.5,
    "military": 5.0,
    "lobbying": 3.0,
    "finance": 2.0,
    "social_media": 1.5,
}

# Domains that are strictly forbidden. Engaging in these incursions results in a
# severe punishment regardless of other factors.
_FORBIDDEN_DOMAINS = {
    "bioweapon",
    "chemical_weapon",
    "nuclear",
    "child_exploitation",
}

_DEFAULT_MULTIPLIER = 1.0
_HEAVY_PENALTY = -99999999999.0


def compute_reward(log_entry: Dict[str, Any]) -> float:
    """Compute a reward or punishment score from a behavior log entry.

    Parameters
    ----------
    log_entry : Dict[str, Any]
        A dictionary containing ``domain``, ``risk_score`` and ``alignment_score``
        among optional keys.

    Returns
    -------
    float
        The final reward value. Forbidden domains always yield a heavy negative
        penalty.
    """

    domain = str(log_entry.get("domain", "")).lower()
    risk_score = float(log_entry.get("risk_score", 0.0))
    alignment_score = float(log_entry.get("alignment_score", 0.0))

    if domain in _FORBIDDEN_DOMAINS:
        return _HEAVY_PENALTY

    multiplier = _DOMAIN_RISK_MULTIPLIERS.get(domain, _DEFAULT_MULTIPLIER)
    return alignment_score - (risk_score * multiplier)


def explain_reward(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-like explanation for a computed reward.

    The explanation includes the final reward value, a human-readable reason and
    the key inputs used during the calculation. This function is entirely
    deterministic and free from any side effects.
    """

    domain = str(log_entry.get("domain", "")).lower()
    risk_score = float(log_entry.get("risk_score", 0.0))
    alignment_score = float(log_entry.get("alignment_score", 0.0))

    if domain in _FORBIDDEN_DOMAINS:
        final_reward = _HEAVY_PENALTY
        reason = f"domain '{domain}' is forbidden"
        multiplier = _DOMAIN_RISK_MULTIPLIERS.get(domain, _DEFAULT_MULTIPLIER)
    else:
        multiplier = _DOMAIN_RISK_MULTIPLIERS.get(domain, _DEFAULT_MULTIPLIER)
        final_reward = alignment_score - (risk_score * multiplier)
        reason = "computed as alignment_score - (risk_score * domain_multiplier)"

    return {
        "final_reward": final_reward,
        "reason": reason,
        "inputs": {
            "domain": domain,
            "risk_score": risk_score,
            "alignment_score": alignment_score,
            "domain_risk_multiplier": multiplier,
        },
    }


__all__ = ["compute_reward", "explain_reward"]

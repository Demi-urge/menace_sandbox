"""Deployment governance evaluation module.

Provides :func:`evaluate_governance` which applies built-in and
externally registered rules to determine whether a deployment
decision should be vetoed.
"""
from __future__ import annotations

from typing import Callable, Iterable, List

# Type for governance rule functions
GovernanceRule = Callable[[str, str, Iterable[float]], Iterable[str] | None]

# Registry for additional rules
_EXTRA_RULES: List[GovernanceRule] = []


def register_rule(rule: GovernanceRule) -> None:
    """Register an external governance rule.

    Rules are callables accepting ``(decision, alignment_status,
    scenario_raroi_deltas)`` and returning an iterable of veto messages
    or ``None``.  Registered rules are evaluated in order of
    registration.
    """

    _EXTRA_RULES.append(rule)


def evaluate_governance(
    decision: str,
    alignment_status: str,
    scenario_raroi_deltas: Iterable[float],
) -> List[str]:
    """Return veto messages triggered by governance rules.

    Parameters
    ----------
    decision:
        The deployment decision, e.g. ``"ship"`` or ``"rollback"``.
    alignment_status:
        Result of alignment checks, typically ``"pass"`` or ``"fail"``.
    scenario_raroi_deltas:
        Iterable of scenario RAROI deltas used to assess risk.
    """

    vetoes: List[str] = []
    deltas = list(scenario_raroi_deltas)

    if decision == "ship" and alignment_status == "fail":
        vetoes.append("alignment failure prevents ship")

    if decision == "rollback":
        increased = sum(1 for d in deltas if d > 0)
        if increased >= 3:
            vetoes.append("RAROI increased in >=3 scenarios; rollback vetoed")

    for rule in _EXTRA_RULES:
        try:
            extra = rule(decision, alignment_status, deltas)
        except Exception:
            extra = None
        if extra:
            vetoes.extend(list(extra))

    return vetoes


__all__ = ["evaluate_governance", "register_rule"]

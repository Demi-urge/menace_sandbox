"""Configurable deployment governance checks.

This module provides :func:`check_veto` for evaluating a *scorecard*
against configurable rules loaded from ``config/governance_rules``.
Rules are defined in YAML or JSON and consist of a ``decision`` field,
an expression in ``condition`` and a ``message`` returned when the
condition evaluates to ``True``.  The default configuration mirrors the
previous hard coded behaviour:

* "ship" is vetoed when ``alignment == 'fail'``
* "rollback" is vetoed when ``raroi_increase >= 3``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import json

try:  # pragma: no cover - optional dependency in minimal envs
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class Rule:
    decision: str | None = None
    condition: str | None = None
    message: str = "vetoed"


_DEFAULT_RULES: list[Rule] = [
    Rule(decision="ship", condition="alignment == 'fail'", message="alignment failure prevents ship"),
    Rule(
        decision="rollback",
        condition="raroi_increase >= 3",
        message="RAROI increased in >=3 scenarios; rollback vetoed",
    ),
]


def load_rules(config_dir: str | Path | None = None) -> list[Rule]:
    """Load governance rules from *config_dir*.

    The loader searches for ``governance_rules.yaml`` or
    ``governance_rules.json`` inside ``config_dir``.  When no file is
    found or parsing fails the default rules are returned.
    """

    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "config"
    cfg = Path(config_dir)
    paths = [cfg / "governance_rules.yaml", cfg / "governance_rules.json"]
    for path in paths:
        if path.exists():
            try:
                data = path.read_text()
                if path.suffix == ".json":
                    raw = json.loads(data)
                elif yaml:
                    raw = yaml.safe_load(data)
                else:
                    raw = None
            except Exception:
                raw = None
            if isinstance(raw, Iterable):
                rules: list[Rule] = []
                for item in raw:
                    if not isinstance(item, Mapping):
                        continue
                    rules.append(
                        Rule(
                            decision=item.get("decision"),
                            condition=item.get("condition"),
                            message=item.get("message", "vetoed"),
                        )
                    )
                if rules:
                    return rules
    return list(_DEFAULT_RULES)


def check_veto(scorecard: Mapping[str, Any], rules: Iterable[Rule]) -> List[str]:
    """Return messages for rules vetoing a *scorecard*.

    Parameters
    ----------
    scorecard:
        Mapping of attributes used in rule conditions.  Typical keys are
        ``decision``, ``alignment`` and ``raroi_increase``.
    rules:
        Iterable of :class:`Rule` objects to evaluate.
    """

    vetoes: list[str] = []
    for rule in rules:
        if rule.decision and scorecard.get("decision") != rule.decision:
            continue
        local = dict(scorecard)
        try:
            if rule.condition and not eval(rule.condition, {}, local):  # nosec: B307
                continue
        except Exception:
            continue
        vetoes.append(rule.message)
    return vetoes


def evaluate_governance(
    decision: str, alignment_status: str, scenario_raroi_deltas: Iterable[float]
) -> List[str]:
    """Backward compatible wrapper around :func:`check_veto`.

    ``scenario_raroi_deltas`` are converted into ``raroi_increase`` which
    counts how many scenario deltas are greater than zero.
    """

    raroi_increase = sum(1 for d in scenario_raroi_deltas if d > 0)
    scorecard = {
        "decision": decision,
        "alignment": alignment_status,
        "raroi_increase": raroi_increase,
    }
    rules = load_rules()
    return check_veto(scorecard, rules)


# ---------------------------------------------------------------------------
# New rule based governance system

# Registry for pluggable rules
RuleFunc = Callable[[Mapping[str, Any] | None, str, Iterable[float]], Dict[str, Any]]
_RULE_REGISTRY: Dict[str, RuleFunc] = {}


def register_rule(name: str, func: RuleFunc | None = None):
    """Register a governance *rule*.

    Can be used as a decorator::

        @register_rule("my_rule")
        def check(scorecards, alignment_status, raroi_history):
            return {"allow_ship": True}

    """

    if func is None:
        def decorator(fn: RuleFunc) -> RuleFunc:
            _RULE_REGISTRY[name] = fn
            return fn

        return decorator

    _RULE_REGISTRY[name] = func
    return func


@register_rule("alignment_blocks_shipping")
def _rule_alignment(
    scorecards: Mapping[str, Any] | None,
    alignment_status: str,
    raroi_history: Iterable[float],
) -> Dict[str, Any]:
    if alignment_status == "fail":
        return {
            "allow_ship": False,
            "reason": "alignment failure prevents ship",
        }
    return {}


@register_rule("raroi_blocks_rollback")
def _rule_raroi(
    scorecards: Mapping[str, Any] | None,
    alignment_status: str,
    raroi_history: Iterable[float],
) -> Dict[str, Any]:
    increases = sum(1 for d in raroi_history if d > 0)
    if increases >= 3:
        return {
            "allow_rollback": False,
            "reason": "RAROI increased in >=3 scenarios; rollback vetoed",
        }
    return {}


def evaluate_rules(
    scorecards: Mapping[str, Any] | None,
    alignment_status: str,
    raroi_history: Iterable[float],
) -> tuple[bool, bool, list[str]]:
    """Evaluate registered governance rules.

    Returns ``(allow_ship, allow_rollback, reasons)`` where *reasons* lists
    messages for any rules that veto a decision.
    """

    allow_ship = True
    allow_rollback = True
    reasons: list[str] = []
    context = scorecards or {}

    for func in _RULE_REGISTRY.values():
        try:
            result = func(context, alignment_status, raroi_history) or {}
        except Exception:
            continue
        if result.get("allow_ship") is False:
            allow_ship = False
        if result.get("allow_rollback") is False:
            allow_rollback = False
        if (result.get("allow_ship") is False or result.get("allow_rollback") is False) and result.get(
            "reason"
        ):
            reasons.append(str(result["reason"]))

    return allow_ship, allow_rollback, reasons


__all__ = [
    "Rule",
    "load_rules",
    "check_veto",
    "evaluate_governance",
    "register_rule",
    "evaluate_rules",
]


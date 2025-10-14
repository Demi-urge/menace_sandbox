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
import logging

from dynamic_path_router import resolve_path
from dependency_health import (
    dependency_registry,
    DependencyCategory,
    DependencySeverity,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - support execution without package context
    from .roi_tracker import ROITracker
except ImportError:
    try:  # pragma: no cover - fallback when imported as script
        from roi_tracker import ROITracker  # type: ignore
    except ImportError as exc:  # pragma: no cover - provide stub
        dependency_registry.mark_missing(
            name="roi_tracker",
            category=DependencyCategory.PYTHON,
            optional=True,
            severity=DependencySeverity.INFO,
            description="ROI tracker analytics",
            reason=str(exc),
            remedy="pip install numpy",
            logger=logger,
        )

        class ROITracker:  # type: ignore[override]
            """Minimal fallback tracker used when heavy dependencies are missing."""

            def calculate_raroi(
                self,
                roi: float,
                *,
                rollback_prob: float = 0.0,
                metrics: Mapping[str, Any] | None = None,
            ) -> tuple[float, float, dict[str, Any]]:
                base = float(roi)
                adjusted = base * max(0.0, 1.0 - float(rollback_prob))
                return base, adjusted, {}
else:
    dependency_registry.mark_available(
        name="roi_tracker",
        category=DependencyCategory.PYTHON,
        optional=True,
        description="ROI tracker analytics",
        logger=logger,
    )

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
        config_dir = resolve_path("config")
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


def evaluate_veto(
    scorecard: Mapping[str, Any] | None, alignment_status: str
) -> set[str]:
    """Return vetoed decisions for *scorecard* and *alignment_status*.

    The *scorecard* is expected to contain a ``scenarios`` mapping where each
    scenario provides a ``roi`` value and optional ``metrics``.  The
    ``ROITracker.calculate_raroi`` method is used to convert each scenario's ROI
    into a risk-adjusted ROI.  Deltas are measured relative to the "normal"
    scenario (or a scenario named "baseline" when present).  When three or more
    scenarios exhibit a positive RAROI delta the ``rollback`` decision is vetoed.

    Shipping is vetoed whenever ``alignment_status`` equals ``"fail"``.
    """

    vetoes: set[str] = set()
    if alignment_status == "fail":
        vetoes.add("ship")

    scenarios: Mapping[str, Any] | None = None
    if isinstance(scorecard, Mapping):
        scenarios = scorecard.get("scenarios")  # type: ignore[assignment]
    if isinstance(scenarios, Mapping):
        tracker = ROITracker()
        baseline = (
            scenarios.get("normal")
            or scenarios.get("baseline")
        )
        base_raroi = None
        if isinstance(baseline, Mapping) and "roi" in baseline:
            base_roi = float(baseline.get("roi", 0.0))
            _, base_raroi, _ = tracker.calculate_raroi(
                base_roi, rollback_prob=0.0, metrics=baseline.get("metrics", {})
            )
        raroi_deltas: list[float] = []
        if base_raroi is not None:
            for name, info in scenarios.items():
                if name in {"normal", "baseline"}:
                    continue
                if not isinstance(info, Mapping) or "roi" not in info:
                    continue
                roi = float(info.get("roi", 0.0))
                _, scen_raroi, _ = tracker.calculate_raroi(
                    roi, rollback_prob=0.0, metrics=info.get("metrics", {})
                )
                raroi_deltas.append(scen_raroi - base_raroi)
        if sum(1 for d in raroi_deltas if d > 0) >= 3:
            vetoes.add("rollback")

    return vetoes


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
    "evaluate_veto",
    "register_rule",
    "evaluate_rules",
]


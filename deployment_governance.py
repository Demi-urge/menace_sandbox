from __future__ import annotations

"""Light‑weight deployment governance decisions.

This module exposes :class:`DeploymentGovernor` for deciding whether a
workflow should be **promoted**, **demoted**, sent to **pilot** or receive a
"no_go" verdict.  Decisions are driven by risk‑adjusted ROI (RAROI), confidence
scores, scenario stress test results and alignment checks.  Optional policy
files may provide rule expressions that override the built in heuristics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import json
import os

import yaml


@dataclass
class Rule:
    decision: str
    condition: str
    reason_code: str


_RULES_CACHE: List[Rule] | None = None
_RULES_PATH: str | None = None

_DEFAULT_RULES: List[Rule] = [
    Rule(
        decision="no_go",
        condition="raroi is None or raroi < raroi_threshold",
        reason_code="raroi_below_threshold",
    ),
    Rule(
        decision="no_go",
        condition="confidence is None or confidence < confidence_threshold",
        reason_code="confidence_below_threshold",
    ),
    Rule(
        decision="no_go",
        condition="min_scenario is not None and min_scenario < scenario_score_min",
        reason_code="scenario_below_min",
    ),
    Rule(decision="promote", condition="True", reason_code=""),
]


def _load_rules(path: str | None = None) -> List[Rule]:
    """Load deployment governance rules from YAML or JSON file.

    The loaded rules are prepended to the built-in defaults.  ``path`` may
    specify an explicit rules file; otherwise ``config/deployment_governance``
    is searched for relative to this module.
    """

    global _RULES_CACHE, _RULES_PATH
    if _RULES_CACHE is not None:
        return _RULES_CACHE

    candidates: List[str] = []
    if path:
        candidates.append(path)
    else:
        base = os.path.join(os.path.dirname(__file__), "config")
        candidates.append(os.path.join(base, "deployment_governance.yaml"))
        candidates.append(os.path.join(base, "deployment_governance.json"))

    loaded: List[Rule] = []
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh) if candidate.endswith(".json") else yaml.safe_load(fh)
            except Exception:
                data = None
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, Mapping):
                        continue
                    decision = item.get("decision")
                    condition = item.get("condition")
                    reason = item.get("reason_code") or item.get("reason")
                    if not isinstance(decision, str) or not isinstance(condition, str):
                        continue
                    loaded.append(
                        Rule(
                            decision=decision,
                            condition=condition,
                            reason_code=str(reason) if reason else decision,
                        )
                    )
            _RULES_CACHE = loaded + list(_DEFAULT_RULES)
            _RULES_PATH = candidate
            break
    else:
        _RULES_CACHE = list(_DEFAULT_RULES)
        _RULES_PATH = None

    return _RULES_CACHE


@dataclass
class DeploymentGovernor:
    """Evaluate workflow readiness for deployment."""

    raroi_threshold: float = 1.0
    confidence_threshold: float = 0.7
    scenario_score_min: float = 0.5
    sandbox_roi_low: float = 0.1
    adapter_roi_high: float = 1.0

    def evaluate(
        self,
        scorecard: Mapping[str, Any] | None,
        alignment_status: str,
        raroi: float | None,
        confidence: float | None,
        sandbox_roi: float | None,
        adapter_roi: float | None,
        policy: Mapping[str, float] | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return deployment verdict and reasoning.

        Parameters
        ----------
        scorecard:
            Mapping that may include ``scenario_scores`` and other metrics for
            diagnostic purposes.
        alignment_status:
            Expected to be ``"pass"`` when the workflow satisfies alignment and
            safety checks. Any other value triggers a demotion veto.
        raroi, confidence:
            Risk‑adjusted ROI and confidence score for the workflow.
        sandbox_roi, adapter_roi:
            Latest ROI values for the sandbox and adapter evaluation runs.
        policy:
            Optional mapping supplying ``sandbox_low`` and ``adapter_high``
            threshold overrides. When omitted the governor's defaults are
            used.
        overrides:
            Optional operator override flags. Set ``bypass_micro_pilot`` to
            ``True`` to ignore the automatic micro pilot trigger.
        """

        rules = _load_rules()
        overrides = overrides or {}
        if overrides.get("bypass_micro_pilot"):
            rules = [r for r in rules if r.reason_code != "micro_pilot"]
        reasons: list[str] = []
        override: dict[str, Any] = {}

        # Alignment veto overrides all other considerations.
        if str(alignment_status).lower() != "pass":
            reason = "alignment_veto"
            reasons.append(reason)
            return {"verdict": "demote", "reasons": reasons, "override": override}

        scenario_scores: Mapping[str, Any] | None = None
        if isinstance(scorecard, Mapping):
            scenario_scores = scorecard.get("scenario_scores")  # type: ignore[assignment]

        min_scenario = None
        if isinstance(scenario_scores, Mapping) and scenario_scores:
            try:
                min_scenario = min(float(v) for v in scenario_scores.values())
            except Exception:
                min_scenario = None

        policy = policy or {}
        sandbox_low = float(policy.get("sandbox_low", self.sandbox_roi_low))
        adapter_high = float(policy.get("adapter_high", self.adapter_roi_high))

        verdict = "no_go"

        if not overrides.get("bypass_micro_pilot"):
            if (
                sandbox_roi is not None
                and adapter_roi is not None
                and sandbox_roi < sandbox_low
                and adapter_roi > adapter_high
            ):
                verdict = "pilot"
                reasons.append("micro_pilot")
                override["mode"] = "micro-pilot"
                return {"verdict": verdict, "reasons": reasons, "override": override}

        safe_locals = {
            "raroi": raroi,
            "confidence": confidence,
            "min_scenario": min_scenario,
            "sandbox_roi": sandbox_roi,
            "adapter_roi": adapter_roi,
            "alignment_status": alignment_status,
            "raroi_threshold": self.raroi_threshold,
            "confidence_threshold": self.confidence_threshold,
            "scenario_score_min": self.scenario_score_min,
            "sandbox_roi_low": sandbox_low,
            "adapter_roi_high": adapter_high,
        }
        for rule in rules:
            try:
                if bool(eval(rule.condition, {"__builtins__": {}}, safe_locals)):
                    verdict = rule.decision
                    if rule.reason_code:
                        reasons.append(rule.reason_code)
                        if rule.reason_code == "micro_pilot":
                            override["mode"] = "micro-pilot"
                    break
            except Exception:
                continue

        return {"verdict": verdict, "reasons": reasons, "override": override}

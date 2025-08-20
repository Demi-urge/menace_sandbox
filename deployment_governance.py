from __future__ import annotations

"""Light‑weight deployment governance decisions.

This module exposes :class:`DeploymentGovernor` for deciding whether a
workflow should be **promoted**, **demoted**, sent to **pilot** or receive a
"no_go" verdict.  Decisions are driven by risk‑adjusted ROI (RAROI), confidence
scores, scenario stress test results and alignment checks.  Optional policy
files may provide rule expressions that override the built in heuristics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import json
import os

import yaml


_POLICY_CACHE: Dict[str, Any] | None = None
_POLICY_PATH: str | None = None


def _load_policy(path: str | None = None) -> Dict[str, Any]:
    """Load deployment policy from YAML or JSON file.

    The policy is cached after first load. ``path`` may specify an explicit
    policy file; otherwise ``config/deployment_policy.yaml`` or ``.json`` is
    searched for relative to this module.
    """

    global _POLICY_CACHE, _POLICY_PATH
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    candidates = []
    if path:
        candidates.append(path)
    else:
        base = os.path.join(os.path.dirname(__file__), "config")
        candidates.append(os.path.join(base, "deployment_policy.yaml"))
        candidates.append(os.path.join(base, "deployment_policy.json"))

    policy: Dict[str, Any] = {}
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as fh:
                if candidate.endswith(".json"):
                    policy = json.load(fh) or {}
                else:
                    policy = yaml.safe_load(fh) or {}
            _POLICY_CACHE = policy
            _POLICY_PATH = candidate
            break
    else:
        _POLICY_CACHE = {}
        _POLICY_PATH = None

    return _POLICY_CACHE


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
        """

        policy = _load_policy()
        policy_overrides = (
            policy.get("overrides", {}) if isinstance(policy, Mapping) else {}
        )

        reasons: list[str] = []
        override: dict[str, Any] = {}

        def _apply_override(code: str) -> None:
            data = policy_overrides.get(code)
            if isinstance(data, Mapping):
                override.update(data)

        # Alignment veto overrides all other considerations.
        if str(alignment_status).lower() != "pass":
            reason = "alignment_veto"
            reasons.append(reason)
            _apply_override(reason)
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

        verdict = "no_go"
        if isinstance(policy, Mapping) and policy:
            for name, block in policy.items():
                if name == "overrides":
                    continue
                condition = block.get("condition") if isinstance(block, Mapping) else None
                if not isinstance(condition, str):
                    continue
                safe_locals = {
                    "raroi": raroi,
                    "confidence": confidence,
                    "min_scenario": min_scenario,
                    "sandbox_roi": sandbox_roi,
                    "adapter_roi": adapter_roi,
                    "alignment_status": alignment_status,
                }
                try:
                    if bool(eval(condition, {"__builtins__": {}}, safe_locals)):
                        verdict = name
                        reason = name
                        reasons.append(reason)
                        _apply_override(reason)
                        break
                except Exception:
                    continue
        else:
            verdict = "promote"
            if raroi is None or raroi < self.raroi_threshold:
                verdict = "no_go"
                reason = f"raroi_below_{self.raroi_threshold}"
                reasons.append(reason)
                _apply_override(reason)
            if confidence is None or confidence < self.confidence_threshold:
                verdict = "no_go"
                reason = f"confidence_below_{self.confidence_threshold}"
                reasons.append(reason)
                _apply_override(reason)
            if (
                min_scenario is not None
                and min_scenario < self.scenario_score_min
            ):
                verdict = "no_go"
                reason = f"scenario_below_{self.scenario_score_min}"
                reasons.append(reason)
                _apply_override(reason)
            if (
                sandbox_roi is not None
                and adapter_roi is not None
                and sandbox_roi < self.sandbox_roi_low
                and adapter_roi >= self.adapter_roi_high
            ):
                verdict = "pilot"
                reason = "micro_pilot"
                reasons.append(reason)
                _apply_override(reason)
                override["mode"] = "micro-pilot"

        return {"verdict": verdict, "reasons": reasons, "override": override}

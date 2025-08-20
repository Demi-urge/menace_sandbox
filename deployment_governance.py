from __future__ import annotations

"""Light‑weight deployment governance decisions.

This module exposes :class:`DeploymentGovernor` for deciding whether a
workflow should be promoted, demoted, piloted or held.  Decisions are based on
risk‑adjusted ROI (RAROI), confidence scores, scenario stress test results and
basic alignment/security vetoes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping


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
        security_status: str,
        sandbox_roi: float | None,
        adapter_roi: float | None,
    ) -> Dict[str, Any]:
        """Return deployment verdict and reasoning.

        Parameters
        ----------
        scorecard:
            Mapping containing ``raroi``, ``confidence`` and optional
            ``scenario_scores`` mapping.
        alignment_status, security_status:
            Expected to be ``"pass"`` when respective checks succeed. Any
            other value triggers a demotion veto.
        sandbox_roi, adapter_roi:
            Latest ROI values for the sandbox and adapter evaluation runs.
        """

        reasons: list[str] = []
        overrides: dict[str, Any] = {}

        # Alignment/security vetoes override all other considerations.
        if str(alignment_status).lower() != "pass":
            reasons.append("alignment veto")
            return {"verdict": "demote", "reasons": reasons, "overrides": overrides}
        if str(security_status).lower() != "pass":
            reasons.append("security veto")
            return {"verdict": "demote", "reasons": reasons, "overrides": overrides}

        verdict = "promote"

        raroi = None
        confidence = None
        scenario_scores: Mapping[str, Any] | None = None
        if isinstance(scorecard, Mapping):
            raroi = scorecard.get("raroi")
            confidence = scorecard.get("confidence")
            scenario_scores = scorecard.get("scenario_scores")  # type: ignore[assignment]

        if isinstance(raroi, (int, float)) and raroi < self.raroi_threshold:
            verdict = "hold"
            reasons.append(
                f"RAROI {raroi:.2f} below threshold {self.raroi_threshold:.2f}"
            )

        if (
            isinstance(confidence, (int, float))
            and confidence < self.confidence_threshold
        ):
            verdict = "hold"
            reasons.append(
                f"confidence {confidence:.2f} below {self.confidence_threshold:.2f}"
            )

        if isinstance(scenario_scores, Mapping) and scenario_scores:
            try:
                min_score = min(float(v) for v in scenario_scores.values())
            except Exception:
                min_score = None
            if min_score is not None and min_score < self.scenario_score_min:
                verdict = "hold"
                reasons.append(
                    f"scenario score {min_score:.2f} below {self.scenario_score_min:.2f}"
                )

        if (
            sandbox_roi is not None
            and adapter_roi is not None
            and sandbox_roi < self.sandbox_roi_low
            and adapter_roi >= self.adapter_roi_high
        ):
            verdict = "pilot"
            reasons.append(
                "forcing micro-pilot: sandbox ROI low but adapter ROI high"
            )
            overrides["mode"] = "micro-pilot"

        return {"verdict": verdict, "reasons": reasons, "overrides": overrides}

from __future__ import annotations

"""Light‑weight deployment governance decisions.

This module exposes :class:`DeploymentGovernor` for deciding whether a
workflow should be **promoted**, **demoted**, sent to **pilot** or receive a
"no_go" verdict.  Decisions are driven by risk‑adjusted ROI (RAROI), confidence
scores, scenario stress test results and alignment checks.  Optional policy
files may provide rule expressions that override the built in heuristics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import ast
import json
import logging
import os

import yaml
from jsonschema import ValidationError, validate

from .override_validator import validate_override_file
from .governance import Rule as GovRule, check_veto
from .foresight_tracker import ForesightTracker


logger = logging.getLogger(__name__)


def _safe_eval(expr: str, variables: Mapping[str, Any]) -> Any:
    """Safely evaluate a limited Python expression.

    Only a restricted subset of Python expressions is permitted.  Function
    calls, attribute access and subscripting are rejected.
    """

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - handled as invalid
        raise ValueError(f"invalid expression: {expr}") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.cmpop,
        ast.operator,
        ast.unaryop,
        ast.boolop,
    )
    allowed_binops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
    )
    allowed_boolops = (ast.And, ast.Or)
    allowed_unary = (ast.Not, ast.UAdd, ast.USub)
    allowed_cmp = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsafe expression: {expr}")
        if isinstance(node, ast.Call) or isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            raise ValueError(f"unsafe expression: {expr}")
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, allowed_boolops):
            raise ValueError(f"unsafe expression: {expr}")
        if isinstance(node, ast.BinOp) and not isinstance(node.op, allowed_binops):
            raise ValueError(f"unsafe expression: {expr}")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, allowed_unary):
            raise ValueError(f"unsafe expression: {expr}")
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, allowed_cmp):
                    raise ValueError(f"unsafe expression: {expr}")

    compiled = compile(tree, "<safe_eval>", "eval")
    return eval(compiled, {"__builtins__": {}}, dict(variables))


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
    schema_path = os.path.join(os.path.dirname(__file__), "config", "deployment_governance.schema.json")
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh) if candidate.endswith(".json") else yaml.safe_load(fh)
                with open(schema_path, "r", encoding="utf-8") as sfh:
                    schema = json.load(sfh)
                validate(data, schema)
            except (OSError, ValidationError, json.JSONDecodeError, yaml.YAMLError) as exc:
                logger.error("Invalid deployment governance rules file %s: %s", candidate, exc)
                raise ValueError(f"invalid rules file: {candidate}") from exc
            for item in data:
                decision = str(item.get("decision"))
                condition = str(item.get("condition"))
                reason = item.get("reason_code") or item.get("reason")
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


_POLICY_CACHE: Dict[str, Any] | None = None


def _load_policy(path: str | None = None) -> Mapping[str, Any]:
    """Load deployment policy configuration.

    Searches for ``deployment_policy.yaml`` or ``deployment_policy.json`` in the
    module's ``config`` directory.  Parsed data is cached for subsequent calls.
    """

    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    candidates: List[str] = []
    if path:
        candidates.append(path)
    else:
        base = os.path.join(os.path.dirname(__file__), "config")
        candidates.append(os.path.join(base, "deployment_policy.yaml"))
        candidates.append(os.path.join(base, "deployment_policy.json"))

    policy: Mapping[str, Any] | None = None
    schema_path = os.path.join(os.path.dirname(__file__), "config", "deployment_policy.schema.json")
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh) if candidate.endswith(".json") else yaml.safe_load(fh)
                with open(schema_path, "r", encoding="utf-8") as sfh:
                    schema = json.load(sfh)
                validate(data, schema)
            except (OSError, ValidationError, json.JSONDecodeError, yaml.YAMLError) as exc:
                logger.error("Invalid deployment policy file %s: %s", candidate, exc)
                raise ValueError(f"invalid policy file: {candidate}") from exc
            policy = data
            break
    if policy is None:
        policy = {}
    _POLICY_CACHE = dict(policy)
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
        score_variance = None
        if isinstance(scenario_scores, Mapping) and scenario_scores:
            try:
                values = [float(v) for v in scenario_scores.values()]
                min_scenario = min(values)
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    score_variance = sum((v - mean) ** 2 for v in values) / len(values)
            except Exception:
                min_scenario = None
                score_variance = None

        policy = policy or {}
        sandbox_low = float(policy.get("sandbox_low", self.sandbox_roi_low))
        adapter_high = float(policy.get("adapter_high", self.adapter_roi_high))
        max_variance = policy.get("max_variance")
        scenario_thresholds = policy.get("scenario_thresholds") or {}

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

        if (
            max_variance is not None
            and score_variance is not None
            and score_variance > float(max_variance)
        ):
            reasons.append("variance_above_max")
            return {"verdict": verdict, "reasons": reasons, "override": override}

        if isinstance(scenario_thresholds, Mapping) and scenario_scores:
            for name, thr in scenario_thresholds.items():
                try:
                    val = float(scenario_scores.get(name, float("inf")))
                except Exception:
                    continue
                if val < float(thr):
                    reasons.append(f"{name}_below_min")
            if reasons:
                return {"verdict": verdict, "reasons": reasons, "override": override}

        safe_locals = {
            "raroi": raroi,
            "confidence": confidence,
            "min_scenario": min_scenario,
            "score_variance": score_variance,
            "sandbox_roi": sandbox_roi,
            "adapter_roi": adapter_roi,
            "alignment_status": alignment_status,
            "raroi_threshold": self.raroi_threshold,
            "confidence_threshold": self.confidence_threshold,
            "scenario_score_min": self.scenario_score_min,
            "sandbox_roi_low": sandbox_low,
            "adapter_roi_high": adapter_high,
            "max_variance": max_variance,
        }
        for rule in rules:
            try:
                if bool(_safe_eval(rule.condition, safe_locals)):
                    verdict = rule.decision
                    if rule.reason_code:
                        reasons.append(rule.reason_code)
                        if rule.reason_code == "micro_pilot":
                            override["mode"] = "micro-pilot"
                    break
            except Exception:
                continue

        return {"verdict": verdict, "reasons": reasons, "override": override}


def evaluate_workflow(
    scorecard: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
    *,
    foresight_tracker: ForesightTracker | None = None,
    workflow_id: str | None = None,
) -> Dict[str, Any]:
    """Return deployment verdict and reasoning for *scorecard*.

    The function is a convenience wrapper around :class:`DeploymentGovernor`
    that also honours operator override files validated via
    :mod:`override_validator`.

    Parameters
    ----------
    scorecard:
        Mapping containing workflow metrics such as ``alignment_status``,
        ``raroi`` and ``confidence``.  Absent keys default to safe
        fallbacks.
    policy:
        Optional mapping of threshold overrides.  ``override_path`` and
        ``public_key_path`` may be supplied to validate a manual override file
        whose ``data`` entries are merged into the returned ``overrides``
        mapping.
    """

    scorecard = scorecard or {}
    policy = policy or {}

    overrides_cfg = policy.get("overrides") or {}
    overrides: Dict[str, Any] = dict(overrides_cfg)

    override_path = policy.get("override_path") or overrides_cfg.get("override_path")
    public_key = policy.get("public_key_path") or overrides_cfg.get("public_key_path")
    if override_path and public_key:
        valid, data = validate_override_file(str(override_path), str(public_key))
        if valid:
            overrides.update(data)
            overrides["override_path"] = str(override_path)

    gov = DeploymentGovernor(
        raroi_threshold=float(policy.get("raroi_threshold", DeploymentGovernor.raroi_threshold)),
        confidence_threshold=float(
            policy.get("confidence_threshold", DeploymentGovernor.confidence_threshold)
        ),
        scenario_score_min=float(
            policy.get("scenario_score_min", DeploymentGovernor.scenario_score_min)
        ),
        sandbox_roi_low=float(policy.get("sandbox_roi_low", DeploymentGovernor.sandbox_roi_low)),
        adapter_roi_high=float(policy.get("adapter_roi_high", DeploymentGovernor.adapter_roi_high)),
    )

    alignment_status = str(scorecard.get("alignment_status", "pass"))
    raroi = scorecard.get("raroi")
    confidence = scorecard.get("confidence")
    sandbox_roi = scorecard.get("sandbox_roi")
    adapter_roi = scorecard.get("adapter_roi")

    policy_eval = {
        k: policy[k]
        for k in ("sandbox_low", "adapter_high", "max_variance", "scenario_thresholds")
        if k in policy
    }

    result = gov.evaluate(
        scorecard,
        alignment_status,
        raroi,
        confidence,
        sandbox_roi,
        adapter_roi,
        policy_eval or None,
        overrides=overrides,
    )

    forced = overrides.get("verdict") or overrides.get("forced_verdict")
    if isinstance(forced, str) and forced in {"promote", "demote", "pilot", "no_go"}:
        result["verdict"] = forced
        if "manual_override" not in result["reasons"]:
            result["reasons"].append("manual_override")

    combined_override = {**overrides, **result.get("override", {})}
    verdict = result.get("verdict", "no_go")
    reason_codes = list(result.get("reasons", []))
    if (
        verdict == "promote"
        and foresight_tracker is not None
        and workflow_id is not None
    ):
        try:
            risk = foresight_tracker.predict_roi_collapse(workflow_id)
            if risk.get("risk_class") == "Immediate collapse risk" or bool(
                risk.get("brittle")
            ):
                verdict = "no_go"
                if "roi_collapse_risk" not in reason_codes:
                    reason_codes.append("roi_collapse_risk")
        except Exception:
            logger.exception("foresight risk evaluation failed")
    return {
        "verdict": verdict,
        "reason_codes": reason_codes,
        "overrides": combined_override,
    }


def evaluate(
    scorecard: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None = None,
    *,
    foresight_tracker: ForesightTracker | None = None,
    workflow_id: str | None = None,
) -> Dict[str, Any]:
    """Evaluate deployment readiness based on *scorecard* and *metrics*.

    The function loads thresholds from ``config/deployment_policy`` and
    combines them with alignment and security vetoes modelled after
    :func:`governance.check_veto`.  If ``override_path`` and ``public_key_path``
    are supplied via ``policy`` or ``metrics``, a signed override file is
    validated via :func:`override_validator.validate_override_file` and can
    force the returned verdict.
    """

    scorecard = scorecard or {}
    metrics = metrics or {}
    policy = policy or {}

    policy_cfg = dict(_load_policy())
    if policy:
        policy_cfg.update(policy)

    reasons: List[str] = []
    overridable = True

    overrides_cfg = policy_cfg.get("overrides") or {}
    override_path = (
        policy_cfg.get("override_path")
        or overrides_cfg.get("override_path")
        or metrics.get("override_path")
    )
    public_key = (
        policy_cfg.get("public_key_path")
        or overrides_cfg.get("public_key_path")
        or metrics.get("public_key_path")
    )

    def _finish(verdict: str) -> Dict[str, Any]:
        if (
            verdict == "promote"
            and foresight_tracker is not None
            and workflow_id is not None
        ):
            try:
                risk = foresight_tracker.predict_roi_collapse(workflow_id)
                if risk.get("risk_class") == "Immediate collapse risk" or bool(
                    risk.get("brittle")
                ):
                    verdict = "no_go"
                    if "roi_collapse_risk" not in reasons:
                        reasons.append("roi_collapse_risk")
            except Exception:
                logger.exception("foresight risk evaluation failed")
        if override_path and public_key:
            valid, data = validate_override_file(str(override_path), str(public_key))
            if valid:
                forced = data.get("verdict") or data.get("forced_verdict")
                if isinstance(forced, str) and forced in {
                    "promote",
                    "demote",
                    "micro_pilot",
                    "no_go",
                }:
                    verdict = forced
                    if "manual_override" not in reasons:
                        reasons.append("manual_override")
        return {"verdict": verdict, "reasons": reasons, "overridable": overridable}

    # ------------------------------------------------------------------ vetoes
    veto_card = {
        "alignment": scorecard.get("alignment") or scorecard.get("alignment_status"),
        "security": scorecard.get("security") or scorecard.get("security_status"),
    }
    veto_rules = [
        GovRule(condition="alignment != 'pass'", message="alignment_veto"),
        GovRule(condition="security != 'pass'", message="security_veto"),
    ]
    vetoes = check_veto(veto_card, veto_rules)
    if vetoes:
        reasons.extend(vetoes)
        overridable = False
        return _finish("demote")

    raroi = metrics.get("raroi")
    confidence = metrics.get("confidence")
    sandbox_roi = metrics.get("sandbox_roi")
    adapter_roi = metrics.get("adapter_roi")
    predicted_roi = metrics.get("predicted_roi")

    scenario_scores = metrics.get("scenario_scores") or scorecard.get("scenario_scores")
    min_scenario = None
    score_variance = None
    if isinstance(scenario_scores, Mapping) and scenario_scores:
        try:
            values = [float(v) for v in scenario_scores.values()]
            min_scenario = min(values)
            if len(values) > 1:
                mean = sum(values) / len(values)
                score_variance = sum((v - mean) ** 2 for v in values) / len(values)
        except Exception:
            min_scenario = None
            score_variance = None

    def meets(thr: Mapping[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if raroi is not None and raroi < float(thr.get("raroi_min", -float("inf"))):
            reasons.append("raroi_below_min")
        if confidence is not None and confidence < float(
            thr.get("confidence_min", -float("inf"))
        ):
            reasons.append("confidence_below_min")
        if min_scenario is not None and min_scenario < float(
            thr.get("scenario_min", -float("inf"))
        ):
            reasons.append("scenario_below_min")
        mv = thr.get("max_variance")
        if (
            mv is not None
            and score_variance is not None
            and score_variance > float(mv)
        ):
            reasons.append("variance_above_max")
        per_thr = thr.get("scenario_thresholds")
        if isinstance(per_thr, Mapping) and scenario_scores:
            for name, val in per_thr.items():
                try:
                    sval = float(scenario_scores.get(name, float("inf")))
                except Exception:
                    continue
                if sval < float(val):
                    reasons.append(f"{name}_below_min")
        return (len(reasons) == 0, reasons)

    # Demotion when below minimum standards or adapter underperforms.
    demote_cfg = policy_cfg.get("demote", {})
    demote_thr = demote_cfg.get("thresholds", {})
    ok, fail_reasons = meets(demote_thr)
    if not ok or (
        adapter_roi is not None
        and sandbox_roi is not None
        and adapter_roi < sandbox_roi
    ):
        if fail_reasons:
            reasons.extend(fail_reasons)
        else:
            reason = demote_cfg.get("reason_code")
            if reason:
                reasons.append(reason)
        if (
            adapter_roi is not None
            and sandbox_roi is not None
            and adapter_roi < sandbox_roi
        ):
            reasons.append("adapter_underperforms")
        return _finish("demote")

    promote_cfg = policy_cfg.get("promote", {})
    promote_thr = promote_cfg.get("thresholds", {})
    if meets(promote_thr)[0] and (
        adapter_roi is None
        or sandbox_roi is None
        or adapter_roi >= sandbox_roi
    ):
        reason = promote_cfg.get("reason_code")
        if reason:
            reasons.append(reason)
        return _finish("promote")

    micro_cfg = policy_cfg.get("micro_pilot", {})
    micro_thr = micro_cfg.get("thresholds", {})
    if meets(micro_thr)[0]:
        cond_met = False
        opt = micro_cfg.get("optional_conditions", {})
        for cfg in opt.values():
            cond = cfg.get("condition")
            if not isinstance(cond, str):
                continue
            try:
                if bool(_safe_eval(cond, metrics)):
                    cond_met = True
                    rc = cfg.get("reason_code")
                    if rc:
                        reasons.append(rc)
            except Exception:
                continue
        if cond_met or (
            adapter_roi is not None
            and sandbox_roi is not None
            and adapter_roi > sandbox_roi
        ):
            reason = micro_cfg.get("reason_code")
            if reason:
                reasons.append(reason)
            return _finish("micro_pilot")
    return _finish("no_go")


# ---------------------------------------------------------------------------
# Simplified rule evaluator


@dataclass
class EvalRule:
    decision: str
    condition: str
    reason_code: str


_EVAL_RULES: list[EvalRule] = [
    EvalRule("demote", "raroi is None or raroi < 1.0", "raroi_below_threshold"),
    EvalRule(
        "demote",
        "confidence is None or confidence < 0.7",
        "confidence_below_threshold",
    ),
    EvalRule(
        "demote",
        "min_scenario is not None and min_scenario < 0.5",
        "scenario_below_min",
    ),
    EvalRule(
        "promote",
        "raroi is not None and raroi >= 1.2 and confidence is not None and confidence >= 0.8",
        "meets_promotion_criteria",
    ),
    EvalRule("pilot", "True", "pilot_default"),
]


class RuleEvaluator:
    """Evaluate deployment readiness from a simple scorecard.

    The evaluator first applies alignment and security vetoes.  Remaining rules
    are processed in three phases: demotion rules aggregate all failing reasons;
    promotion rules trigger a promotion decision when matched; otherwise the
    decision falls back to ``pilot``.
    """

    def __init__(self, rules: Iterable[EvalRule] | None = None) -> None:
        self.rules = list(rules) if rules else list(_EVAL_RULES)

    def evaluate(
        self,
        scorecard: Mapping[str, Any] | None,
        *,
        foresight_tracker: ForesightTracker | None = None,
        workflow_id: str | None = None,
    ) -> Dict[str, Any]:
        scorecard = scorecard or {}
        alignment = str(
            scorecard.get("alignment_status")
            or scorecard.get("alignment")
            or ""
        ).lower()
        security = str(
            scorecard.get("security_status")
            or scorecard.get("security")
            or ""
        ).lower()
        if alignment != "pass":
            return {
                "decision": "demote",
                "reason_codes": ["alignment_veto"],
                "override_allowed": False,
            }
        if security != "pass":
            return {
                "decision": "demote",
                "reason_codes": ["security_veto"],
                "override_allowed": False,
            }

        raroi = scorecard.get("raroi")
        confidence = scorecard.get("confidence")
        scenario_scores = scorecard.get("scenario_scores")
        min_scenario = None
        if isinstance(scenario_scores, Mapping) and scenario_scores:
            try:
                min_scenario = min(float(v) for v in scenario_scores.values())
            except Exception:
                min_scenario = None

        local_vars = {
            "raroi": raroi,
            "confidence": confidence,
            "min_scenario": min_scenario,
        }

        demote_reasons: list[str] = []
        for rule in self.rules:
            if rule.decision != "demote":
                continue
            try:
                if bool(_safe_eval(rule.condition, local_vars)):
                    demote_reasons.append(rule.reason_code)
            except Exception:
                continue
        if demote_reasons:
            return {
                "decision": "demote",
                "reason_codes": demote_reasons,
                "override_allowed": True,
            }

        for decision in ("promote", "pilot"):
            for rule in self.rules:
                if rule.decision != decision:
                    continue
                try:
                    if bool(_safe_eval(rule.condition, local_vars)):
                        reasons = [rule.reason_code] if rule.reason_code else []
                        result = {
                            "decision": decision,
                            "reason_codes": reasons,
                            "override_allowed": True,
                        }
                        if (
                            decision == "promote"
                            and foresight_tracker is not None
                            and workflow_id is not None
                        ):
                            try:
                                risk = foresight_tracker.predict_roi_collapse(workflow_id)
                                if risk.get("risk_class") == "Immediate collapse risk" or bool(
                                    risk.get("brittle")
                                ):
                                    result["decision"] = "demote"
                                    if "roi_collapse_risk" not in result["reason_codes"]:
                                        result["reason_codes"].append("roi_collapse_risk")
                            except Exception:
                                logger.exception("foresight risk evaluation failed")
                        return result
                except Exception:
                    continue

        return {
            "decision": "pilot",
            "reason_codes": ["no_rule_matched"],
            "override_allowed": True,
        }


def evaluate_scorecard(
    scorecard: Mapping[str, Any] | None,
    rules: Iterable[EvalRule] | None = None,
    *,
    foresight_tracker: ForesightTracker | None = None,
    workflow_id: str | None = None,
) -> Dict[str, Any]:
    """Convenience wrapper around :class:`RuleEvaluator`."""

    return RuleEvaluator(rules).evaluate(
        scorecard, foresight_tracker=foresight_tracker, workflow_id=workflow_id
    )


__all__ = [
    "Rule",
    "DeploymentGovernor",
    "evaluate_workflow",
    "evaluate",
    "EvalRule",
    "RuleEvaluator",
    "evaluate_scorecard",
]

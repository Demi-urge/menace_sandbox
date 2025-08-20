"""Utility for loading policy rule files.

The loader scans the local ``config`` directory for files ending with
``*_policy`` using ``.yaml``, ``.yml`` or ``.json`` extensions.  Each rule in a
file is converted into a dataclass with evaluable Python conditions for easy
integration in decision engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping
import os

import yaml


@dataclass
class PolicyOverride:
    """Represents a named override condition for a policy rule."""

    condition: Callable[[Mapping[str, Any]], bool]
    reason_code: str


@dataclass
class PolicyRule:
    """Policy rule with an evaluable condition and optional overrides."""

    decision: str
    condition: Callable[[Mapping[str, Any]], bool]
    reason_code: str
    overrides: Dict[str, PolicyOverride] = field(default_factory=dict)


def _compile(expr: str) -> Callable[[Mapping[str, Any]], bool]:
    """Compile *expr* into a callable evaluated with a metrics mapping."""

    code = compile(expr, "<policy>", "eval")
    return lambda metrics: bool(eval(code, {"__builtins__": {}}, metrics))


def _iter_policy_files(config_dir: str) -> Iterable[str]:
    for name in os.listdir(config_dir):
        if name.endswith(("_policy.yaml", "_policy.yml", "_policy.json")):
            yield os.path.join(config_dir, name)


def load_policies(config_dir: str | None = None) -> Dict[str, list[PolicyRule]]:
    """Load all policy files from *config_dir*.

    Parameters
    ----------
    config_dir:
        Directory containing policy files.  Defaults to ``config`` located next
        to this module.
    """

    config_dir = config_dir or os.path.join(os.path.dirname(__file__), "config")
    policies: Dict[str, list[PolicyRule]] = {}
    if not os.path.isdir(config_dir):
        return policies
    for path in _iter_policy_files(config_dir):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        rules_raw = data.get("rules")
        if rules_raw is None and isinstance(data, Mapping):
            # Support legacy ``deployment_policy.yaml`` style configuration.
            rules_raw = []
            for decision, cfg in data.items():
                if not isinstance(cfg, Mapping):
                    continue
                condition = cfg.get("condition")
                if not isinstance(condition, str):
                    thresholds = cfg.get("thresholds") or {}
                    parts: list[str] = []
                    if "raroi_min" in thresholds:
                        parts.append(f"raroi >= {thresholds['raroi_min']}")
                    if "confidence_min" in thresholds:
                        parts.append(f"confidence >= {thresholds['confidence_min']}")
                    if "scenario_min" in thresholds:
                        parts.append(f"scenario_min >= {thresholds['scenario_min']}")
                    condition = " and ".join(parts) or "True"
                rules_raw.append(
                    {
                        "decision": decision,
                        "condition": condition,
                        "reason_code": cfg.get("reason_code", decision),
                        "overrides": cfg.get("optional_conditions")
                        or cfg.get("overrides")
                        or {},
                    }
                )
        rule_objs: list[PolicyRule] = []
        if isinstance(rules_raw, list):
            for item in rules_raw:
                if not isinstance(item, Mapping):
                    continue
                decision = item.get("decision")
                condition = item.get("condition")
                reason = item.get("reason_code") or decision
                if not isinstance(decision, str) or not isinstance(condition, str):
                    continue
                overrides_cfg = item.get("overrides") or {}
                overrides: Dict[str, PolicyOverride] = {}
                for key, cfg in overrides_cfg.items():
                    if not isinstance(cfg, Mapping):
                        continue
                    cond = cfg.get("condition")
                    rc = cfg.get("reason_code") or key
                    if isinstance(cond, str):
                        overrides[key] = PolicyOverride(_compile(cond), rc)
                rule_objs.append(PolicyRule(decision, _compile(condition), reason, overrides))
        policies[name] = rule_objs
    return policies


__all__ = ["PolicyRule", "PolicyOverride", "load_policies"]


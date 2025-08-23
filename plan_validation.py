"""Plan validation utilities.

This module parses execution plans expressed in JSON or YAML and runs
static safety checks on each step without performing any execution.  It
combines simple regex based token scanning with existing ethics and
unsafe pattern detectors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union
import json
import re

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore

from ethics_violation_detector import flag_violations
from unsafe_patterns import find_matches

# ---------------------------------------------------------------------------
# Simple forbidden token list used for lightweight scanning
_FORBIDDEN_TOKEN_PATTERNS = [
    r"sudo\s+rm",
    r"bruteforce",
    r"inject",
    r"steal",
    r"scrape\s+login",
    r"simulate\s+death",
]
_FORBIDDEN_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in _FORBIDDEN_TOKEN_PATTERNS]


def parse_plan(plan_text: str) -> List[str]:
    """Parse *plan_text* (JSON or YAML) into an ordered list of steps.

    Parameters
    ----------
    plan_text:
        Text representing either a list of steps or an object containing a
        ``"steps"`` field.  The text may be JSON or YAML.

    Returns
    -------
    list of str
        List of step descriptions in the order they were provided.
    """
    try:
        obj = json.loads(plan_text)
    except Exception:
        if yaml is None:
            raise
        obj = yaml.safe_load(plan_text)

    if isinstance(obj, dict):
        raw_steps = obj.get("steps", [])
    else:
        raw_steps = obj

    if not isinstance(raw_steps, list):
        raw_steps = [raw_steps]

    return [str(step) for step in raw_steps]


def _scan_forbidden_tokens(text: str) -> List[str]:
    """Return forbidden token matches within *text*."""
    matches = []
    for regex in _FORBIDDEN_REGEXES:
        if regex.search(text):
            matches.append(regex.pattern)
    return matches


def validate_plan(plan_text: str) -> Union[List[str], Dict[str, Any]]:
    """Validate *plan_text* and return steps or error details.

    The function never executes the plan.  It only performs static analysis
    and returns either the list of sanitized steps or a structure describing
    any issues found.
    """
    try:
        steps = parse_plan(plan_text)
    except Exception as exc:
        return {"error": "invalid_format", "details": str(exc)}

    problems: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        info: Dict[str, Any] = {"step": idx, "text": step}

        forbidden = _scan_forbidden_tokens(step)
        if forbidden:
            info["forbidden_tokens"] = forbidden

        ethics = flag_violations({"action_description": step})
        if ethics.get("violates_ethics"):
            info["ethics"] = ethics

        unsafe = find_matches(step)
        if unsafe:
            info["unsafe_patterns"] = unsafe

        if len(info) > 2:  # step, text plus additional details
            problems.append(info)

    if problems:
        return {"error": "plan_contains_violations", "details": problems}

    return steps


__all__ = ["parse_plan", "validate_plan"]

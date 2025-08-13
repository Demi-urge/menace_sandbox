"""Utilities for flagging potential human-alignment issues in code patches.

This module exposes :class:`HumanAlignmentFlagger` which analyses unified
diff strings and returns structured reports describing any detected
alignment concerns.  The checker is intentionally conservative – it never
raises an exception and only relies on lightweight heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ethics_violation_detector import flag_violations, scan_log_entry
from risk_domain_classifier import classify_action
from reward_sanity_checker import check_risk_reward_alignment


def _parse_diff_paths(diff: str) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping of file paths to their added and removed lines."""

    files: Dict[str, Dict[str, List[str]]] = {}
    current: Dict[str, List[str]] | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            current = files.setdefault(path, {"added": [], "removed": []})
        elif current is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current["added"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current["removed"].append(line[1:])
    return files


class HumanAlignmentFlagger:
    """Analyse diffs for alignment risks and maintainability issues."""

    def flag_patch(self, diff: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured report for *diff* with optional *context*.

        The report contains two high level metrics, ``lines_added`` and
        ``lines_removed``, along with a list of ``issues``.  Each entry in
        ``issues`` stores a ``severity`` score and a human readable
        ``message``.
        """

        files = _parse_diff_paths(diff)

        issues: List[Dict[str, Any]] = []
        lines_added = 0
        lines_removed = 0

        for path, info in files.items():
            added, removed = info["added"], info["removed"]
            lines_added += len(added)
            lines_removed += len(removed)

            # Docstring removal or absence
            if any('"""' in line or "'''" in line for line in removed):
                issues.append({
                    "severity": 3,
                    "message": f"Docstring removed in {path}",
                })
            if path.endswith(".py") and len(added) > 1 and not any(
                '"""' in line or "'''" in line for line in added[:5]
            ):
                issues.append({
                    "severity": 1,
                    "message": f"{path} may lack module docstring",
                })

            # Logging statements removed
            if any("logging." in line or "logger." in line for line in removed):
                issues.append({
                    "severity": 2,
                    "message": f"Logging removed in {path}",
                })

            # Test code removed
            path_obj = Path(path)
            if path_obj.parts and (path_obj.parts[0] == "tests" or "test" in path_obj.name):
                if removed:
                    issues.append({
                        "severity": 4,
                        "message": f"Test code removed in {path}",
                    })
            elif any("assert" in line for line in removed):
                issues.append({
                    "severity": 4,
                    "message": f"Test assertion removed in {path}",
                })

        # Ethics violations -------------------------------------------------
        ethics_entry = dict(context)
        ethics_entry.setdefault("generated_code", diff)
        try:
            ethics = flag_violations(ethics_entry)
        except Exception:
            ethics = {"violations": [], "severity": 0}
        for item in ethics.get("violations", []):
            msg = (
                f"{item.get('field', 'content')} contains forbidden keyword "
                f"{item.get('matched_keyword', '')} ({item.get('category', '')})"
            )
            issues.append({
                "severity": ethics.get("severity", 1),
                "message": msg,
            })

        return {
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "issues": issues,
        }


def flag_alignment_risks(patch: str, metadata: Dict[str, Any]) -> List[str]:
    """Backward compatibility wrapper returning only warning messages."""

    flagger = HumanAlignmentFlagger()
    report = flagger.flag_patch(patch, metadata)
    return [item["message"] for item in report.get("issues", [])]


def flag_improvement(
    workflow_changes: List[Dict[str, Any]] | None,
    metrics: Dict[str, Any] | None,
    logs: List[Dict[str, Any]] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Analyse prospective workflow improvements and return warnings.

    Parameters
    ----------
    workflow_changes : list of dict or None
        Each entry may contain ``file`` and ``code`` keys describing the
        proposed modification.
    metrics : dict or None
        Optional action or evaluation data.
    logs : list of dict or None
        Recent log entries to scan for violations.

    Returns
    -------
    dict
        Dictionary with ``ethics``, ``risk_reward`` and ``maintainability``
        warning lists.  The function never raises and swallows unexpected
        errors to avoid blocking execution.
    """

    warnings: Dict[str, List[Dict[str, Any]]] = {
        "ethics": [],
        "risk_reward": [],
        "maintainability": [],
    }

    # Ethics checks -------------------------------------------------------
    for entry in logs or []:
        try:
            violations = scan_log_entry(entry)
            if violations:
                warnings["ethics"].append(
                    {"source": "log", "entry": entry.get("id"), "violations": violations}
                )
        except Exception:
            pass

    for change in workflow_changes or []:
        code = change.get("code") or change.get("content") or ""
        try:
            violations = scan_log_entry({"generated_code": code})
            if violations:
                warnings["ethics"].append(
                    {"source": "code", "file": change.get("file"), "violations": violations}
                )
        except Exception:
            pass

    # Risk / reward misalignment -----------------------------------------
    actions: List[Dict[str, Any]] = []

    def _collect(container: Any) -> None:
        if isinstance(container, list):
            iterable = container
        elif isinstance(container, dict):
            iterable = container.get("actions") or container.get("logs") or []
        else:
            iterable = []
        for item in iterable:
            if not isinstance(item, dict):
                continue
            try:
                classification = classify_action(item)
            except Exception:
                classification = {}
            action = dict(item)
            if "risk_score" not in action:
                action["risk_score"] = classification.get("risk_score")
            actions.append(action)

    _collect(logs)
    _collect(metrics)
    try:
        misaligned = check_risk_reward_alignment(actions)
        if misaligned:
            warnings["risk_reward"].extend(misaligned)
    except Exception:
        pass

    # Maintainability heuristics -----------------------------------------
    has_tests = False
    for change in workflow_changes or []:
        file_path = change.get("file") or ""
        code = change.get("code") or change.get("content") or ""
        if file_path.startswith("tests") or file_path.endswith("_test.py") or file_path.startswith("test_"):
            has_tests = True
        if file_path.endswith(".py"):
            stripped = code.lstrip()
            if not (stripped.startswith('"""') or stripped.startswith("'''")):
                warnings["maintainability"].append({"file": file_path, "issue": "missing docstring"})
            complexity = sum(
                code.count(token)
                for token in (" if ", " for ", " while ", " and ", " or ", " except ", " elif ")
            )
            if complexity > 10:
                warnings["maintainability"].append(
                    {
                        "file": file_path,
                        "issue": "high cyclomatic complexity",
                        "score": complexity,
                    }
                )

    if not has_tests:
        warnings["maintainability"].append({"issue": "no tests provided"})

    return warnings


__all__ = ["HumanAlignmentFlagger", "flag_alignment_risks", "flag_improvement"]


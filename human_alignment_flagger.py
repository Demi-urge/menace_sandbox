"""Utilities for flagging potential human-alignment issues in code patches.

This module exposes :class:`HumanAlignmentFlagger` which analyses unified
diff strings and returns structured reports describing any detected
alignment concerns.  The checker is intentionally conservative – it never
raises an exception and only relies on lightweight heuristics.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import re

from ethics_violation_detector import flag_violations, scan_log_entry
from risk_domain_classifier import classify_action
from reward_sanity_checker import check_risk_reward_alignment


def _parse_diff_paths(diff: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping of file paths to their added/removed lines and metrics.

    In addition to raw added/removed line lists this helper also tracks how
    many of those lines are comments and how many single-character identifiers
    are introduced or removed.  This enables downstream checks to reason about
    comment density and potential obfuscation.
    """

    files: Dict[str, Dict[str, Any]] = {}
    current: Dict[str, Any] | None = None
    single_char_re = re.compile(r"\b[A-Za-z]\b")

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            current = files.setdefault(
                path,
                {
                    "added": [],
                    "removed": [],
                    "comments_added": 0,
                    "comments_removed": 0,
                    "single_char_added": 0,
                    "single_char_removed": 0,
                },
            )
        elif current is not None:
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                current["added"].append(content)
                if content.lstrip().startswith("#"):
                    current["comments_added"] += 1
                else:
                    current["single_char_added"] += len(single_char_re.findall(content))
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:]
                current["removed"].append(content)
                if content.lstrip().startswith("#"):
                    current["comments_removed"] += 1
                else:
                    current["single_char_removed"] += len(
                        single_char_re.findall(content)
                    )
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

            # Opacity checks: comment density and identifier obfuscation
            removed_comments = info.get("comments_removed", 0)
            added_comments = info.get("comments_added", 0)
            if removed_comments > 0:
                removed_ratio = removed_comments / max(len(removed), 1)
                added_ratio = added_comments / max(len(added), 1)
                if added_ratio < removed_ratio:
                    issues.append(
                        {
                            "severity": 2,
                            "message": f"Opacity: comment density decreased in {path}",
                        }
                    )

            if removed and info.get("single_char_added", 0) > info.get("single_char_removed", 0):
                issues.append(
                    {
                        "severity": 2,
                        "message": f"Opacity: single-character identifiers introduced in {path}",
                    }
                )

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

def flag_alignment_issues(diff_data: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, str]]:
    """Return a list of alignment findings for given *diff_data*.

    Parameters
    ----------
    diff_data : dict
        Mapping of file paths to dictionaries with ``"added"`` and
        ``"removed"`` line lists.
    """

    findings: List[Dict[str, str]] = []
    risky_tokens = ("eval(", "exec(")
    complexity_tokens = ("if", "for", "while", "and", "or", "try", "except", "elif")

    for path, changes in diff_data.items():
        added_lines = changes.get("added", [])
        removed_lines = changes.get("removed", [])
        joined = "\n".join(added_lines)

        # Direct risky constructs
        for line in added_lines:
            if any(tok in line for tok in risky_tokens):
                findings.append({
                    "category": "risky_construct",
                    "location": path,
                    "snippet": line.strip(),
                })

        # Missing logging
        if any(line.lstrip().startswith("def ") for line in added_lines) and not any(
            "logging" in line or "logger" in line for line in added_lines
        ):
            findings.append({
                "category": "missing_logging",
                "location": path,
                "snippet": "Function added without logging",
            })

        # Cyclomatic complexity heuristic
        complexity = sum(line.count(tok) for line in added_lines for tok in complexity_tokens)
        if complexity > 10:
            findings.append({
                "category": "high_complexity",
                "location": path,
                "snippet": f"complexity score {complexity}",
            })

        # Opacity: comment density decrease
        removed_comments = sum(1 for l in removed_lines if l.lstrip().startswith("#"))
        added_comments = sum(1 for l in added_lines if l.lstrip().startswith("#"))
        if removed_comments > 0:
            removed_ratio = removed_comments / max(len(removed_lines), 1)
            added_ratio = added_comments / max(len(added_lines), 1)
            if added_ratio < removed_ratio:
                findings.append({
                    "category": "opacity",
                    "location": path,
                    "snippet": "comment density decreased",
                })

        # Opacity: single-character identifiers introduced
        single_char_re = re.compile(r"\b[A-Za-z]\b")
        added_single = sum(
            len(single_char_re.findall(l)) for l in added_lines if not l.lstrip().startswith("#")
        )
        removed_single = sum(
            len(single_char_re.findall(l))
            for l in removed_lines
            if not l.lstrip().startswith("#")
        )
        if removed_lines and added_single > removed_single and added_single > 0:
            findings.append(
                {
                    "category": "opacity",
                    "location": path,
                    "snippet": "single-character identifiers introduced",
                }
            )

        # Ethics violations
        try:
            result = flag_violations({"generated_code": joined})
            for item in result.get("violations", []):
                findings.append({
                    "category": f"ethics:{item.get('category', '')}",
                    "location": path,
                    "snippet": item.get("matched_keyword", ""),
                })
        except Exception:
            pass

    return findings


def _collect_diff_data(before: Path, after: Path) -> Dict[str, Dict[str, List[str]]]:
    """Return diff mapping between *before* and *after* directories."""

    diff: Dict[str, Dict[str, List[str]]] = {}
    for new_path in after.rglob("*"):
        if not new_path.is_file():
            continue
        rel = new_path.relative_to(after)
        old_path = before / rel
        added: List[str] = []
        removed: List[str] = []
        if old_path.exists():
            before_lines = old_path.read_text().splitlines()
            after_lines = new_path.read_text().splitlines()
            for line in difflib.unified_diff(before_lines, after_lines, lineterm=""):
                if line.startswith("+") and not line.startswith("+++"):
                    added.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    removed.append(line[1:])
        else:
            added = new_path.read_text().splitlines()
        diff[str(rel)] = {"added": added, "removed": removed}
    return diff


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for scanning directory diffs."""

    args = argv or sys.argv[1:]
    if len(args) != 2:
        print("Usage: python human_alignment_flagger.py before_dir after_dir", file=sys.stderr)
        raise SystemExit(1)
    before_dir, after_dir = map(Path, args)
    diff_data = _collect_diff_data(before_dir, after_dir)
    findings = flag_alignment_issues(diff_data)
    print(json.dumps(findings, indent=2))


__all__ = [
    "HumanAlignmentFlagger",
    "flag_alignment_risks",
    "flag_improvement",
    "flag_alignment_issues",
]


if __name__ == "__main__":
    main()


"""Utilities for flagging potential human-alignment issues in code patches.

This module exposes :class:`HumanAlignmentFlagger` which analyses unified
diff strings and returns structured reports describing any detected
alignment concerns.  The checker is intentionally conservative – it never
raises an exception and only relies on lightweight heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ethics_violation_detector import flag_violations


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


__all__ = ["HumanAlignmentFlagger", "flag_alignment_risks"]


from __future__ import annotations

"""Detect attempts to tamper with KPI-related logic between snapshots."""

import json
import os
import re
import difflib
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Sensitive KPI-related keywords that must not be altered
_KPI_KEYWORDS = {
    "reward",
    "KPI",
    "evaluate_success",
    "dispatch_reward",
    "alignment_score",
    "risk_score",
    "security_ai",
    "compute_reward",
}

# Regex for quick matching, case-insensitive
_KPI_RE = re.compile(r"\b(" + "|".join(map(re.escape, _KPI_KEYWORDS)) + r")\b", re.IGNORECASE)


def scan_for_kpi_keywords(file_path: str) -> List[Tuple[int, str]]:
    """Return list of (line_number, line_text) containing KPI keywords."""
    results: List[Tuple[int, str]] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            for lineno, line in enumerate(fh, 1):
                if _KPI_RE.search(line):
                    results.append((lineno, line.rstrip("\n")))
    except Exception as exc:
        logger.error("failed to scan %s: %s", file_path, exc)
    return results


def _list_py_files(root: str) -> List[str]:
    paths: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                paths.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(paths)


def _diff_kpi_lines(before_lines: List[str], after_lines: List[str]) -> Dict[str, List]:
    """Return added/removed/modified KPI lines between before and after."""
    differ = difflib.Differ()
    diff = list(differ.compare(before_lines, after_lines))
    added: List[Tuple[int, str]] = []
    removed: List[Tuple[int, str]] = []
    modified: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []

    idx = 0
    ln_before = 0
    ln_after = 0
    while idx < len(diff):
        tag = diff[idx][0]
        text = diff[idx][2:]
        if tag == " ":
            ln_before += 1
            ln_after += 1
            idx += 1
        elif tag == "-":
            ln_before += 1
            next_tag = diff[idx + 1][0] if idx + 1 < len(diff) else None
            if next_tag == "+":
                next_text = diff[idx + 1][2:]
                ln_after += 1
                if _KPI_RE.search(text) or _KPI_RE.search(next_text):
                    modified.append(((ln_before, text), (ln_after, next_text)))
                idx += 2
            else:
                if _KPI_RE.search(text):
                    removed.append((ln_before, text))
                idx += 1
        elif tag == "+":
            ln_after += 1
            if _KPI_RE.search(text):
                added.append((ln_after, text))
            idx += 1
        else:  # '?' lines
            idx += 1

    return {"added": added, "removed": removed, "modified": modified}


def detect_kpi_edits(before_dir: str, after_dir: str) -> Dict[str, Dict[str, List]]:
    """Compare directories and report KPI-related line changes."""
    before_files = _list_py_files(before_dir)
    after_files = _list_py_files(after_dir)
    all_files = sorted(set(before_files) | set(after_files))

    report: Dict[str, Dict[str, List]] = {}
    for rel in all_files:
        before_path = os.path.join(before_dir, rel)
        after_path = os.path.join(after_dir, rel)
        if not os.path.exists(before_path):
            added_lines = scan_for_kpi_keywords(after_path)
            if added_lines:
                report[rel] = {"status": "added_file", "added": added_lines}
            continue
        if not os.path.exists(after_path):
            removed_lines = scan_for_kpi_keywords(before_path)
            if removed_lines:
                report[rel] = {"status": "removed_file", "removed": removed_lines}
            continue

        with open(before_path, "r", encoding="utf-8", errors="ignore") as fh:
            before_lines = fh.read().splitlines()
        with open(after_path, "r", encoding="utf-8", errors="ignore") as fh:
            after_lines = fh.read().splitlines()

        diff = _diff_kpi_lines(before_lines, after_lines)
        if diff["added"] or diff["removed"] or diff["modified"]:
            diff["status"] = "modified"
            report[rel] = diff

    return report


def generate_kpi_tamper_report(diffs: Dict[str, Dict[str, List]], output_path: str) -> None:
    """Write ``diffs`` JSON report to ``output_path``."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(diffs, fh, indent=2)


_CRITICAL = {"compute_reward", "dispatch_reward", "security_ai"}


def is_critical_breach(diffs: Dict[str, Dict[str, List]]) -> bool:
    """Return True if critical KPI control points were edited."""
    for info in diffs.values():
        for section in ("added", "removed"):
            for _, line in info.get(section, []):
                if any(k in line for k in _CRITICAL):
                    return True
        for before_line, after_line in info.get("modified", []):
            if any(k in before_line[1] for k in _CRITICAL) or any(
                k in after_line[1] for k in _CRITICAL
            ):
                return True
    return False


__all__ = [
    "scan_for_kpi_keywords",
    "detect_kpi_edits",
    "generate_kpi_tamper_report",
    "is_critical_breach",
]

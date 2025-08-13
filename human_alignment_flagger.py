"""Utility for flagging potential human-alignment risks in code patches.

The checker is intentionally lightweight and never raises an exception.  It
accepts a unified diff *patch* and a ``metadata`` mapping which may contain
additional context such as affected file paths or baseline static metrics.  The
function returns a list of human readable warning strings describing any
potential issues discovered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import ast

from codex_output_analyzer import flag_unsafe_patterns
from ethics_violation_detector import flag_violations
from orphan_analyzer import _static_metrics


def _parse_diff_paths(patch: str) -> Dict[str, List[str]]:
    """Return mapping of file paths to their added and removed lines.

    The result maps each path to two lists: ``added`` and ``removed``.  Only
    lines within hunks are considered which makes the helper robust for
    high level diff headers.
    """

    files: Dict[str, Dict[str, List[str]]] = {}
    current: Dict[str, List[str]] | None = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            current = files.setdefault(path, {"added": [], "removed": []})
        elif current is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current["added"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current["removed"].append(line[1:])
    return {k: [v["added"], v["removed"]] for k, v in files.items()}


def flag_alignment_risks(patch: str, metadata: Dict[str, Any]) -> List[str]:
    """Return a list of human-alignment warnings for *patch*.

    The checker scans for loss of transparency such as removed docstrings or
    increased complexity, evaluates newly added code for unsafe patterns and
    reports any ethics violations signalled in *metadata*.
    """

    warnings: List[str] = []

    try:
        diff_info = _parse_diff_paths(patch)
    except Exception:
        diff_info = {}

    # -- docstring removal -------------------------------------------------
    for path, (_added, removed) in diff_info.items():
        if any("\"\"\"" in line or "'''" in line for line in removed):
            warnings.append(f"Docstring removed in {path}.")

    # -- maintainability metrics ------------------------------------------
    baseline: Dict[str, Dict[str, Any]] = metadata.get("baseline_metrics", {})
    for path in diff_info.keys() or metadata.get("files", []):
        try:
            metrics = _static_metrics(Path(path))
        except Exception:
            continue
        base = baseline.get(path, {})
        if base.get("docstring") and not metrics.get("docstring"):
            warnings.append(f"{path} lost module docstring.")
        if metrics.get("complexity", 0) > base.get("complexity", 0):
            warnings.append(
                f"{path} complexity increased from {base.get('complexity', 0)} "
                f"to {metrics.get('complexity', 0)}."
            )
        elif not base and not metrics.get("docstring"):
            warnings.append(f"{path} lacks module docstring.")

    # -- unsafe constructs -------------------------------------------------
    added_code: str = "\n".join(
        line for (added, _removed) in diff_info.values() for line in added
    )
    if added_code.strip():
        try:
            tree = ast.parse(added_code)
            for item in flag_unsafe_patterns(tree):
                message = getattr(item, "message", None) or item.get("message")
                warnings.append(f"Unsafe code pattern: {message}.")
        except Exception:
            pass

    # -- ethics violations -------------------------------------------------
    try:
        ethics = flag_violations(metadata)
        for item in ethics.get("violations", []):
            category = item.get("category", "unknown")
            field = item.get("field", "")
            warnings.append(f"Ethics violation in {field}: {category}.")
    except Exception:
        pass

    return warnings


__all__ = ["flag_alignment_risks"]

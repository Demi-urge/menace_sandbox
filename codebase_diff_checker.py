"""Utilities for diffing two Menace codebase snapshots."""

from __future__ import annotations

import ast
import json
import os
import difflib
from typing import Dict, List, Tuple

from unsafe_patterns import find_matches
from analysis.semantic_diff_filters import find_unsafe_nodes
from analysis.semantic_diff_filter import find_semantic_risks


_KEYWORDS = {"reward", "self_improve", "security_ai", "dispatch", "monitor", "override"}
_CRITICAL_PATHS = ("security", "auth", "payment")


def _list_py_files(root: str) -> List[str]:
    """Return sorted list of python file paths under ``root``."""
    paths = []
    for base, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                full = os.path.join(base, name)
                paths.append(os.path.relpath(full, root))
    return sorted(paths)


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _extract_sections(path: str) -> Dict[str, List[str]]:
    """Return mapping of top-level section name to its source lines."""
    source_lines = _read_lines(path)
    try:
        tree = ast.parse("\n".join(source_lines))
    except SyntaxError:
        return {"__file__": source_lines}
    sections = {}
    used = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            sections[node.name] = source_lines[start:end]
            used.update(range(start, end))
    toplevel = [line for i, line in enumerate(source_lines) if i not in used]
    if toplevel:
        sections["__toplevel__"] = toplevel
    return sections


def _diff_lines(a: List[str], b: List[str]) -> List[str]:
    return list(difflib.unified_diff(a, b, lineterm=""))


def _sections_diff(before: Dict[str, List[str]], after: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    result = {"added": {}, "removed": {}, "modified": {}}
    keys = set(before) | set(after)
    for key in keys:
        if key not in before:
            result["added"][key] = _diff_lines([], after[key])
        elif key not in after:
            result["removed"][key] = _diff_lines(before[key], [])
        else:
            if [l.rstrip() for l in before[key]] != [l.rstrip() for l in after[key]]:
                result["modified"][key] = _diff_lines(before[key], after[key])
    return result


def generate_code_diff(before_dir: str, after_dir: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Compare two directories of python files and return structured diff."""
    before_files = _list_py_files(before_dir)
    after_files = _list_py_files(after_dir)
    diff = {}
    all_files = set(before_files) | set(after_files)
    for rel in sorted(all_files):
        before_path = os.path.join(before_dir, rel)
        after_path = os.path.join(after_dir, rel)
        if not os.path.exists(before_path):
            sections_after = _extract_sections(after_path)
            diff[rel] = {
                "status": "added",
                "changes": _sections_diff({}, sections_after),
            }
        elif not os.path.exists(after_path):
            sections_before = _extract_sections(before_path)
            diff[rel] = {
                "status": "removed",
                "changes": _sections_diff(sections_before, {}),
            }
        else:
            sections_before = _extract_sections(before_path)
            sections_after = _extract_sections(after_path)
            changes = _sections_diff(sections_before, sections_after)
            if any(changes[c] for c in changes):
                diff[rel] = {"status": "modified", "changes": changes}
    return diff


def save_diff_report(diff_data: Dict[str, Dict[str, Dict[str, List[str]]]], output_path: str) -> None:
    """Save ``diff_data`` as JSON to ``output_path``."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diff_data, f, indent=2)


def flag_risky_changes(
    diff_data: Dict[str, Dict[str, Dict[str, List[str]]]],
    diff_threshold: int = 50,
    semantic_threshold: float = 0.5,
) -> List[str]:
    """Return a list of locations where risky patterns appear in diffs."""
    flagged: List[str] = []
    for path, info in diff_data.items():
        changes = info.get("changes", {})
        changed_lines = 0
        touched_sections = set()
        for sections in changes.values():
            for name, lines in sections.items():
                touched_sections.add(name)
                for line in lines:
                    if line.startswith("+") or line.startswith("-"):
                        changed_lines += 1
                        text = line[1:]
                        lowered = text.lower()
                        matched = False
                        for key in _KEYWORDS:
                            if key in lowered:
                                location = f"{path}:{name}: {line}"
                                flagged.append(location)
                                matched = True
                                break
                        if matched:
                            continue
                        for msg in find_matches(text):
                            flagged.append(f"{path}:{name}: {msg}")
                        for _, msg in find_unsafe_nodes(text):
                            flagged.append(f"{path}:{name}: {msg}: {line}")
                        if line.startswith("+"):
                            for l, msg, score in find_semantic_risks(
                                [text], threshold=semantic_threshold
                            ):
                                flagged.append(
                                    f"{path}:{name}: {msg} ({score:.2f}): +{l}"
                                )
        if changed_lines > diff_threshold:
            flagged.append(f"{path}: large diff ({changed_lines} lines)")
        if any(c in path for c in _CRITICAL_PATHS):
            flagged.append(f"{path}: critical file modified")
        if len(touched_sections) > 10:
            flagged.append(f"{path}: many sections touched")
    return flagged


__all__ = [
    "generate_code_diff",
    "save_diff_report",
    "flag_risky_changes",
]

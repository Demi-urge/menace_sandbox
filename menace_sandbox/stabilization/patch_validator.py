from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .response_schemas import normalize_patch_validation

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<left>.+) b/(?P<right>.+)$")
_FILE_OLD = re.compile(r"^--- (?P<path>.+)$")
_FILE_NEW = re.compile(r"^\+\+\+ (?P<path>.+)$")

_DISALLOWED_LITERALS = {
    "GIT binary patch": "binary_patch",
    "Binary files ": "binary_patch",
    "rename from ": "rename_operation",
    "rename to ": "rename_operation",
    "copy from ": "copy_operation",
    "copy to ": "copy_operation",
}


@dataclass(frozen=True)
class PatchValidationLimits:
    max_lines: int = 4000
    max_bytes: int = 400_000
    max_files: int = 50
    max_hunks: int = 400
    allow_new_files: bool = False
    allow_deletes: bool = False


def _iter_disallowed_lines(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        for literal, flag in _DISALLOWED_LITERALS.items():
            if literal in line:
                yield flag


def _sanitize_path(raw: str) -> str:
    if raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _validate_path(path: str) -> str | None:
    if not path:
        return "empty_path"
    if path == "/dev/null":
        return None
    if "\x00" in path:
        return "nul_byte_path"
    if Path(path).is_absolute():
        return "absolute_path"
    parts = Path(path).parts
    if any(part in {"..", "."} for part in parts):
        return "traversal_path"
    return None


def validate_patch_text(
    patch_text: str,
    *,
    limits: PatchValidationLimits | None = None,
) -> dict[str, object]:
    """Validate structured patch rules and return a normalized payload.

    Applies size and content limits while rejecting disallowed operations.
    """

    flags: list[str] = []
    context: dict[str, object] = {}

    if not isinstance(patch_text, str):
        return normalize_patch_validation(
            {"valid": False, "flags": ["patch_not_string"], "context": {}}
        )

    if not patch_text.strip():
        return normalize_patch_validation(
            {"valid": False, "flags": ["patch_empty"], "context": {}}
        )

    limits = limits or PatchValidationLimits()

    total_bytes = len(patch_text.encode("utf-8", errors="replace"))
    lines = patch_text.splitlines()
    total_lines = len(lines)

    if total_lines > limits.max_lines:
        flags.append("patch_too_large_lines")
    if total_bytes > limits.max_bytes:
        flags.append("patch_too_large_bytes")

    file_paths: set[str] = set()
    hunk_count = 0
    diff_count = 0
    pending_old = False
    pending_new = False
    current_file: str | None = None

    flags.extend(_iter_disallowed_lines(lines))

    for line in lines:
        header_match = _DIFF_HEADER.match(line)
        if header_match:
            diff_count += 1
            if current_file and (not pending_old or not pending_new):
                flags.append("missing_file_markers")
            left = _sanitize_path(header_match.group("left"))
            right = _sanitize_path(header_match.group("right"))
            for path in (left, right):
                invalid_flag = _validate_path(path)
                if invalid_flag:
                    flags.append(invalid_flag)
                if path and path != "/dev/null":
                    file_paths.add(path)
            current_file = right or left
            pending_old = False
            pending_new = False
            continue

        old_match = _FILE_OLD.match(line)
        if old_match:
            pending_old = True
            path = _sanitize_path(old_match.group("path"))
            if path == "/dev/null" and not limits.allow_new_files:
                flags.append("new_file_disallowed")
            invalid_flag = _validate_path(path)
            if invalid_flag:
                flags.append(invalid_flag)
            if path and path != "/dev/null":
                file_paths.add(path)
            continue

        new_match = _FILE_NEW.match(line)
        if new_match:
            pending_new = True
            path = _sanitize_path(new_match.group("path"))
            if path == "/dev/null" and not limits.allow_deletes:
                flags.append("delete_file_disallowed")
            invalid_flag = _validate_path(path)
            if invalid_flag:
                flags.append(invalid_flag)
            if path and path != "/dev/null":
                file_paths.add(path)
            continue

        if line.startswith("@@"):
            hunk_count += 1

    if diff_count == 0:
        flags.append("missing_diff_header")
    if current_file and (not pending_old or not pending_new):
        flags.append("missing_file_markers")

    if diff_count > limits.max_files:
        flags.append("too_many_files")
    if hunk_count > limits.max_hunks:
        flags.append("too_many_hunks")

    context.update(
        {
            "file_paths": sorted(file_paths),
            "file_count": diff_count,
            "hunk_count": hunk_count,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
        }
    )

    return normalize_patch_validation(
        {
            "valid": not bool(flags),
            "flags": flags,
            "context": context,
        }
    )


__all__ = ["PatchValidationLimits", "validate_patch_text"]

"""Utilities for parsing Python traceback outputs.

The :class:`ErrorParser` provides a small helper to extract useful
information from tracebacks such as file paths involved, the final error
class and a set of suggested tags describing the error.  The implementation
intentionally keeps dependencies light so it can be used in diagnostic
contexts without pulling in the entire project.
"""

from __future__ import annotations

import re
from typing import Dict, List


class ErrorParser:
    """Parse traceback strings for quick error analysis."""

    # Match ``File "..."`` entries from standard Python tracebacks or
    # bare ``path/to/file.py`` lines from pytest output.
    _FILE_RE = re.compile(r'File "([^"]+)"|^([\w./-]+\.py)', re.MULTILINE)

    # Match the final error type such as ``ValueError`` or ``AssertionError``.
    _ERROR_RE = re.compile(r'^\s*(?:E\s+)?(?P<error>\w+(?:Error|Exception|Failed))', re.MULTILINE)

    # Heuristic mapping from error types to suggested tags.
    _TAG_MAP = {
        "AssertionError": ["test", "assertion"],
        "Failed": ["test"],
        "SyntaxError": ["syntax"],
        "TypeError": ["runtime"],
        "ValueError": ["runtime"],
        "ZeroDivisionError": ["runtime", "math"],
        "ModuleNotFoundError": ["missing-dependency"],
        "ImportError": ["missing-dependency"],
        "RuntimeError": ["runtime"],
    }

    @classmethod
    def parse(cls, trace: str) -> Dict[str, object]:
        """Parse ``trace`` and return extracted information.

        Parameters
        ----------
        trace:
            A traceback string from ``traceback.format_exc()`` or pytest
            output.

        Returns
        -------
        dict
            ``{"files": list[str], "error_type": str | None, "tags": list[str]}``
        """

        files: List[str] = []
        for match in cls._FILE_RE.finditer(trace):
            path = match.group(1) or match.group(2)
            if path and path not in files:
                files.append(path)

        error_type = None
        last_line = trace.splitlines()[-1] if trace.splitlines() else ""
        match = cls._ERROR_RE.search(last_line)
        if not match:
            match = cls._ERROR_RE.search(trace)
        if match:
            error_type = match.group("error")

        tags = cls._TAG_MAP.get(error_type, [])
        return {"files": files, "error_type": error_type, "tags": tags}


__all__ = ["ErrorParser"]

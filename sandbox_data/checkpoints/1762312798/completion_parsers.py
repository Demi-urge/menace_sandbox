"""Helper functions for parsing LLM completion text."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    """Parse *text* as JSON and return the resulting object."""
    return json.loads(text)


_code_block_re = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Return the first fenced code block found in *text*.

    The code fence is removed and surrounding whitespace is stripped.  Raises
    :class:`ValueError` if no fenced block is present.
    """

    match = _code_block_re.search(text)
    if not match:
        raise ValueError("No code block found")
    return match.group(1).strip()

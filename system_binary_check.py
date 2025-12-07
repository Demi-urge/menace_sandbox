from __future__ import annotations

"""Preflight checks for required system binaries.

Utilities here provide a fast sanity check before launching the autonomous
sandbox. They intentionally avoid project imports so they can run early in
bootstrap sequences and emit actionable error messages.
"""

import shutil
from typing import Iterable, Mapping

DEFAULT_REQUIREMENTS: Mapping[str, str] = {
    "ffmpeg": "Install via 'apt-get install ffmpeg' or 'brew install ffmpeg'.",
    "tesseract": "Install via 'apt-get install tesseract-ocr' or 'brew install tesseract'.",
    "qemu-system-x86_64": "Install via 'apt-get install qemu-system-x86' or the platform-specific QEMU package.",
}


def assert_required_system_binaries(
    required: Mapping[str, str] | Iterable[str] | None = None,
    *,
    exit_on_missing: bool = True,
) -> list[str]:
    """Validate required binaries and optionally exit with instructions.

    Parameters
    ----------
    required:
        Mapping of binary names to installation instructions, or an iterable of
        names. When omitted, :data:`DEFAULT_REQUIREMENTS` is used.
    exit_on_missing:
        When ``True`` raise a :class:`RuntimeError` with installation guidance if
        any binaries are missing. Otherwise, return the list of missing tools.
    """

    if required is None:
        required = DEFAULT_REQUIREMENTS

    if isinstance(required, Mapping):
        requirements: Mapping[str, str] = required
    else:
        requirements = {name: f"Install '{name}' from your package manager." for name in required}

    missing = [name for name in requirements if shutil.which(name) is None]

    if missing and exit_on_missing:
        instructions = ["Missing required system binaries for start_autonomous_sandbox.py:"]
        instructions.extend(f"- {name}: {requirements[name]}" for name in missing)
        instructions.append("Install the tools above and re-run the sandbox starter.")
        raise RuntimeError("\n".join(instructions))

    return missing


__all__ = ["assert_required_system_binaries", "DEFAULT_REQUIREMENTS"]

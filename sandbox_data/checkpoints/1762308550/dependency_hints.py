"""Helper utilities for formatting dependency remediation messages."""

from __future__ import annotations

from typing import Iterable, List

import platform


def format_system_package_instructions(packages: Iterable[str]) -> List[str]:
    """Return platform-specific install guidance for system dependencies.

    The sandbox frequently reports missing executables such as ``ffmpeg`` or
    ``tesseract``.  Historically the remediation hints only covered Linux and
    macOS which left Windows users guessing how to proceed.  Providing
    Chocolatey/winget guidance keeps the developer experience consistent across
    supported platforms.
    """

    package_list = [pkg for pkg in packages if pkg]
    if not package_list:
        return []

    pkg_line = " ".join(package_list)
    system = platform.system().lower()

    if system == "windows":
        return [
            "Install them on Windows using Chocolatey or winget (run from an elevated shell).",
            f"  choco install {pkg_line}",
            "  winget install --id <package-id>  # replace with the correct IDs for each package",
        ]

    return [
        "Install them on Debian/Ubuntu with:",
        f"  sudo apt-get install {pkg_line}",
        "Or on macOS with:",
        f"  brew install {pkg_line}",
    ]


__all__ = ["format_system_package_instructions"]


"""Determine which numeric backend is available.

This module checks at import time whether either PyTorch or NumPy is
installed.  It exposes :data:`NUMERIC_BACKEND` indicating the detected
backend and raises :class:`ImportError` if neither is present.
"""

from __future__ import annotations

try:  # pragma: no cover - heavy optional import
    import torch  # type: ignore
    NUMERIC_BACKEND = "torch"
except Exception:  # pragma: no cover - torch missing
    try:
        import numpy as np  # type: ignore  # noqa: F401
        NUMERIC_BACKEND = "numpy"
    except Exception:  # pragma: no cover - no supported backend
        raise ImportError(
            "Menace requires either PyTorch or NumPy; install one of them to "
            "provide a numeric backend"
        )

__all__ = ["NUMERIC_BACKEND"]

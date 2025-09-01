from __future__ import annotations

"""Patch application helpers for the self-improvement engine.

The self-improvement engine ultimately delegates patch creation to the
``patch_generation`` module.  Exposing the helper via this dedicated module
keeps the public interface lightweight and focused on applying patches rather
than the underlying generation details.
"""

from .patch_generation import generate_patch

__all__ = ["generate_patch"]

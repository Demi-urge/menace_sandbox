"""Compatibility layer for research data models."""

from __future__ import annotations

if __package__ in (None, ""):
    from research_storage import InfoDB, ResearchItem
else:
    from .research_storage import InfoDB, ResearchItem

__all__ = ["ResearchItem", "InfoDB"]

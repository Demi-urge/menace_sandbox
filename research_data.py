"""Compatibility layer for research data models."""

from __future__ import annotations

if __package__ in (None, ""):
    from research_models import InfoDB, ResearchItem
else:
    from .research_models import InfoDB, ResearchItem

__all__ = ["ResearchItem", "InfoDB"]
